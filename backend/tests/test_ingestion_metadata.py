"""Offline extraction/ledger tests. PostgreSQL concurrency remains in integration tests.

Use real SQLAlchemy queries and ledger tables in SQLite through a small async
facade. SQLite's lock function is a no-op; these tests do not assert PG locking.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APIStatusError
from pydantic import SecretStr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import QuotaReservation, UsageRecord, User, UserQuota, utcnow
from app.services.ingestion.metadata import AccountedMetadataExtractor
from app.services.parsing.metadata import MetadataExtractionSkipped

VALID = '{"candidates":[{"field":"year","value":"2024","evidence":"Published in 2024."}]}'
TEXT = "Published in 2024."


class AsyncSessionFacade:
    def __init__(self, db):
        self.db = db

    async def scalar(self, *args, **kwargs):
        return self.db.scalar(*args, **kwargs)

    async def scalars(self, *args, **kwargs):
        return self.db.scalars(*args, **kwargs)

    async def execute(self, *args, **kwargs):
        return self.db.execute(*args, **kwargs)

    async def get(self, *args, **kwargs):
        return self.db.get(*args, **kwargs)

    def add(self, value):
        self.db.add(value)

    async def commit(self):
        self.db.commit()


class Model:
    def __init__(self):
        self.replies = [VALID]
        self.calls = []
        self.usage = {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}

    async def ainvoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, BaseException):
            raise reply
        return AIMessage(content=reply, usage_metadata=self.usage)


@pytest.fixture
def extraction(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.connection.create_function("pg_advisory_xact_lock", 1, lambda _: 0)
    for model in (User, UserQuota, UsageRecord, QuotaReservation):
        model.__table__.create(engine)
    owner, version = uuid4(), uuid4()
    with Session(engine) as db:
        db.add(
            User(
                id=owner, email="a@example.org", password_hash="unused", email_verified_at=utcnow()
            )
        )
        db.commit()

    @asynccontextmanager
    async def sessions():
        with Session(engine) as db:
            yield AsyncSessionFacade(db)

    async def validate(db):
        if context.invalid:
            raise MetadataExtractionSkipped("论文元数据已手动编辑，已跳过快速模型提取")

    async def no_sleep(_):
        pass

    model = Model()
    monkeypatch.setattr("app.services.ingestion.metadata.chat_model", lambda *a, **kw: model)
    monkeypatch.setattr("app.services.ingestion.metadata.asyncio.sleep", no_sleep)
    settings = Settings(_env_file=None, tracing_enabled=False)
    context = SimpleNamespace(
        owner=owner,
        version=version,
        engine=engine,
        model=model,
        settings=settings,
        invalid=False,
        sessions=sessions,
    )

    def factory(user_id=owner, version_id=version):
        return AccountedMetadataExtractor(
            settings, sessions, None, user_id, version_id, tmp_path, validate
        )

    context.factory = factory
    context.extractor = factory()
    context.path = tmp_path / "metadata-extraction.json"
    yield context
    engine.dispose()


def records(context):
    with Session(context.engine) as db:
        return list(db.scalars(select(UsageRecord).order_by(UsageRecord.created_at)))


async def test_success_is_version_accounted_and_restored_without_new_calls(extraction):
    c = extraction
    assert (await c.extractor(TEXT)).candidates[0].value == "2024"
    first = records(c)
    assert len(first) == 1
    row = first[0]
    assert (row.user_id, row.version_id, row.node) == (c.owner, c.version, "metadata_extract")
    assert (row.amount, row.estimated, row.outcome) == (20, False, "confirmed")
    assert row.provider == c.settings.llm_fast_base_url.rstrip("/")
    assert row.model == c.settings.llm_fast_model
    assert (await c.factory()(TEXT)).candidates[0].value == "2024"
    assert len(c.model.calls) == len(records(c)) == 1
    assert c.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "limit", ["user_daily_model_tokens", "global_daily_model_tokens", "override"]
)
async def test_quota_is_checked_before_any_external_request(extraction, limit):
    c = extraction
    if limit == "override":
        with Session(c.engine) as db:
            db.add(UserQuota(user_id=c.owner, daily_model_tokens=1))
            db.commit()
    else:
        setattr(c.settings, limit, 1)
    with pytest.raises(MetadataExtractionSkipped, match="额度不足"):
        await c.extractor(TEXT)
    assert c.model.calls == [] and records(c) == []


async def test_existing_reservations_are_included_in_budget(extraction):
    c = extraction
    with Session(c.engine) as db:
        db.add(
            QuotaReservation(
                user_id=c.owner,
                version_id=c.version,
                metric="model_tokens",
                amount=c.settings.user_daily_model_tokens,
            )
        )
        db.commit()
    with pytest.raises(MetadataExtractionSkipped, match="额度不足"):
        await c.extractor(TEXT)
    assert c.model.calls == [] and records(c) == []


async def test_missing_usage_is_estimated_with_prompt_and_output_allowance(extraction):
    c = extraction
    c.model.usage = None
    await c.extractor(TEXT)
    row = records(c)[0]
    assert row.estimated and row.outcome == "confirmed"
    assert row.amount > 3000 + len(TEXT.encode())


async def test_malformed_responses_are_paid_and_retry_count_survives_restart(extraction):
    c = extraction
    c.model.replies = ["invalid", "invalid"]
    for extractor in (c.extractor, c.factory()):
        with pytest.raises(MetadataExtractionSkipped, match="次数已耗尽"):
            await extractor(TEXT)
    assert len(c.model.calls) == len(records(c)) == 2
    assert sum(r.amount for r in records(c)) == 40


async def test_second_attempt_has_its_own_budget_check(extraction):
    c = extraction
    c.model.replies = ["invalid", VALID]
    # First call fits exactly, then its real usage leaves too little for a retry.
    from app.services.parsing.metadata import metadata_prompt

    messages = (await metadata_prompt().ainvoke({"text": TEXT})).to_messages()
    c.settings.user_daily_model_tokens = (
        sum(len(str(m.content).encode()) + 16 for m in messages) + 3000
    )
    with pytest.raises(MetadataExtractionSkipped, match="额度不足"):
        await c.extractor(TEXT)
    assert len(c.model.calls) == len(records(c)) == 1
    assert records(c)[0].amount == 20


async def test_invalid_then_valid_response_is_separately_accounted(extraction):
    c = extraction
    c.model.replies = ["invalid", VALID]
    assert (await c.extractor(TEXT)).candidates[0].value == "2024"
    assert len(c.model.calls) == len(records(c)) == 2


async def test_explicit_rejection_releases_charge_and_does_not_retry(extraction):
    c = extraction
    c.model.replies = [
        APIStatusError(
            "private detail",
            response=httpx.Response(401, request=httpx.Request("POST", "https://model.invalid")),
            body=None,
        )
    ]
    with pytest.raises(MetadataExtractionSkipped, match="拒绝"):
        await c.extractor(TEXT)
    row = records(c)[0]
    assert row.amount == 0 and row.outcome == "rejected"
    with pytest.raises(MetadataExtractionSkipped, match="拒绝"):
        await c.factory()(TEXT)
    assert len(c.model.calls) == 1
    assert "private detail" not in c.path.read_text()


async def test_unknown_outcomes_remain_charged_and_bounded(extraction):
    c = extraction
    c.model.replies = [httpx.ReadTimeout("private error")]
    with pytest.raises(MetadataExtractionSkipped, match="次数已耗尽"):
        await c.extractor(TEXT)
    assert len(c.model.calls) == 2
    assert all(r.amount > 3000 and r.estimated and r.outcome == "uncertain" for r in records(c))
    with pytest.raises(MetadataExtractionSkipped):
        await c.factory()(TEXT)
    assert len(c.model.calls) == 2


async def test_cancellation_before_response_keeps_usage_and_remaining_attempt_bound(extraction):
    c = extraction
    c.model.replies = [asyncio.CancelledError(), VALID]
    with pytest.raises(asyncio.CancelledError):
        await c.extractor(TEXT)
    assert records(c)[0].outcome == "uncertain"
    assert (await c.factory()(TEXT)).candidates[0].value == "2024"
    assert len(c.model.calls) == len(records(c)) == 2


async def test_response_saved_before_settlement_recovers_without_duplicate_call(
    extraction, monkeypatch
):
    c = extraction

    async def interrupted(*args):
        raise asyncio.CancelledError()

    monkeypatch.setattr(c.extractor.accounting, "settle", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await c.extractor(TEXT)
    assert records(c)[0].outcome == "uncertain"
    assert (await c.factory()(TEXT)).candidates[0].value == "2024"
    assert len(c.model.calls) == len(records(c)) == 1
    assert records(c)[0].amount == 20 and records(c)[0].outcome == "confirmed"


@pytest.mark.parametrize("change", ["owner", "version", "input", "model"])
async def test_private_result_is_bound_to_owner_version_input_and_model(extraction, change):
    c = extraction
    await c.extractor(TEXT)
    extractor, text = c.factory(), TEXT
    if change == "owner":
        extractor = c.factory(user_id=uuid4())
    elif change == "version":
        extractor = c.factory(version_id=uuid4())
    elif change == "input":
        text += "Different source."
    else:
        c.settings.llm_fast_model = "different-model"
    with pytest.raises(MetadataExtractionSkipped, match="已改变"):
        await extractor(text)
    assert len(c.model.calls) == 1


async def test_disabled_owner_prevents_calls(extraction):
    c = extraction
    with Session(c.engine) as db:
        user = db.get(User, c.owner)
        user.status = "disabled"
        db.commit()
    with pytest.raises(MetadataExtractionSkipped, match="账户或论文状态"):
        await c.extractor(TEXT)
    assert c.model.calls == [] and records(c) == []


async def test_validation_runs_before_usage_intent_and_does_not_charge_manual_edits(extraction):
    c = extraction
    c.invalid = True
    with pytest.raises(MetadataExtractionSkipped, match="手动编辑"):
        await c.extractor(TEXT)
    assert c.model.calls == [] and records(c) == []


async def test_reconciliation_cannot_update_another_owners_record(extraction):
    c = extraction
    await c.extractor(TEXT)
    row = records(c)[0]
    with pytest.raises(ValueError, match="owner"):
        await c.extractor.accounting.settle(uuid4(), row.id, 9999)
    assert records(c)[0].amount == 20
    assert UUID(json.loads(c.path.read_text())["attempts"][0]["record_id"]) == row.id


async def test_configured_fast_client_json_request_and_vendor_usage(extraction, monkeypatch):
    from app.services.providers.clients import chat_model

    c = extraction
    c.settings.llm_fast_base_url = "https://fast-model.invalid/v1"
    c.settings.llm_fast_api_key = SecretStr("unit-test")
    client = chat_model(c.settings, "fast", max_tokens=3000)
    # Mock the SDK boundary: current SDK releases use httpx2, while respx
    # intercepts httpx. No transport or proxy should run in this offline test.
    body = {
        "id": "unit-response",
        "model": c.settings.llm_fast_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": VALID},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    parse = AsyncMock(return_value=SimpleNamespace(parse=lambda: body, headers={}))
    monkeypatch.setattr(client.root_async_client.chat.completions.with_raw_response, "parse", parse)
    monkeypatch.setattr("app.services.ingestion.metadata.chat_model", lambda *a, **kw: client)
    assert (await c.extractor(TEXT)).candidates[0].value == "2024"
    request = parse.call_args.kwargs
    assert request["model"] == c.settings.llm_fast_model
    assert request["extra_body"]["max_tokens"] == 3000
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"]["enable_thinking"] is False
    assert parse.await_count == 1 and records(c)[0].amount == 20


async def test_table_summaries_use_accounted_fast_model_and_restore_cache(extraction, tmp_path):
    from app.services.chunking.chunker import chunk_document
    from app.services.ingestion.chunk_summaries import enrich_table_summaries
    from tests.test_chunking import block, document

    c = extraction
    c.model.replies = ['{"summary":"Table 1 compares accuracy: Ours 95.2, Baseline 93.1."}']
    doc = document(
        [
            block(
                "| Method | Accuracy |\n|---|---|\n| Ours | 95.2 |\n| Baseline | 93.1 |", "table"
            ),
            block("Table 1: accuracy comparison", "caption"),
        ]
    )
    ctx = {"settings": c.settings, "session_factory": c.sessions, "engine": None}

    async def validate(db):
        pass

    drafts = chunk_document(doc, "Paper")
    original = drafts[0].text
    await enrich_table_summaries(ctx, c.owner, c.version, tmp_path, drafts, validate)
    assert drafts[0].chunk_metadata["summary_method"] == "fast_model"
    assert drafts[0].text == original and "Baseline 93.1" in drafts[0].embedding_input
    assert records(c)[0].node == "chunk_summary" and records(c)[0].amount == 20
    again = chunk_document(doc, "Paper")
    await enrich_table_summaries(ctx, c.owner, c.version, tmp_path, again, validate)
    assert again[0].embedding_input == drafts[0].embedding_input
    assert len(c.model.calls) == 1


@pytest.mark.parametrize("failure", ["quota", "fabricated_number", "malformed"])
async def test_table_summary_failure_preserves_complete_table_and_extractive_index(
    extraction,
    tmp_path,
    failure,
):
    from app.services.chunking.chunker import chunk_document
    from app.services.ingestion.chunk_summaries import enrich_table_summaries
    from tests.test_chunking import block, document

    c = extraction
    if failure == "quota":
        c.settings.user_daily_model_tokens = 1
    c.model.replies = [
        '{"summary":"Ours achieves 999.5 accuracy."}' if failure != "malformed" else "{}"
    ]
    drafts = chunk_document(
        document([block("| Method | Accuracy |\n|---|---|\n| Ours | 95.2 |", "table")]), "Paper"
    )
    before = drafts[0].embedding_input

    async def validate(db):
        pass

    await enrich_table_summaries(
        {"settings": c.settings, "session_factory": c.sessions, "engine": None},
        c.owner,
        c.version,
        tmp_path,
        drafts,
        validate,
    )
    assert drafts[0].chunk_metadata["summary_method"] == "extractive"
    assert drafts[0].embedding_input == before
    assert "| Ours | 95.2 |" in drafts[0].text and drafts[0].indexable
    if failure == "quota":
        assert c.model.calls == [] and records(c) == []
