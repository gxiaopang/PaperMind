"""Offline memory workflow tests with real SQL queries against SQLite.

These test scope predicates, provenance, CRUD and summarization without providers.
PostgreSQL migrations/advisory locks require the separate Docker chat suite.
"""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import Identity, require_identity
from app.api.v1.memory import router
from app.db.models import (
    Base,
    ChatRun,
    Chunk,
    Citation,
    Collection,
    Conversation,
    MemoryEntry,
    Message,
    Paper,
    PaperVersion,
    ParseJob,
    ResearchProject,
    User,
    utcnow,
)
from app.db.session import get_db
from app.graph.context import node_payload
from app.graph.state import RAGState
from app.schemas.memory import MemoryInput, ResearchState
from app.services.memory.consolidation import consolidate
from app.services.memory.history import (
    compact_text,
    estimate_tokens,
    history_tokens,
    load_memory,
    trim_history,
)
from app.services.memory.store import (
    evidence_map,
    evidence_status,
    recall,
    save_candidates,
    select_memories,
    validate_memory,
)


@compiles(JSONB, "sqlite")
def compile_jsonb(element, compiler, **kwargs):
    return "JSON"


class OfflineSession:
    """Awaitable facade for SQLAlchemy's synchronous SQLite test session."""

    def __init__(self, engine):
        self.sync = Session(engine, expire_on_commit=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.sync.close()

    async def get(self, *args, **kwargs):
        return self.sync.get(*args, **kwargs)

    async def scalar(self, *args, **kwargs):
        return self.sync.scalar(*args, **kwargs)

    async def scalars(self, *args, **kwargs):
        return self.sync.scalars(*args, **kwargs)

    async def execute(self, *args, **kwargs):
        return self.sync.execute(*args, **kwargs)

    async def commit(self):
        self.sync.commit()

    async def flush(self):
        self.sync.flush()

    def add(self, item):
        self.sync.add(item)


@pytest.fixture
def memory_db():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        owner = User(email="memory@example.org", password_hash="test", email_verified_at=utcnow())
        other = User(email="stranger@example.org", password_hash="test", email_verified_at=utcnow())
        db.add_all([owner, other])
        db.flush()
        a = ResearchProject(user_id=owner.id, name="低秩适配")
        b = ResearchProject(user_id=owner.id, name="图神经网络")
        foreign = ResearchProject(user_id=other.id, name="另一个用户的项目")
        db.add_all([a, b, foreign])
        db.flush()
        conv = Conversation(user_id=owner.id, project_id=a.id)
        second = Conversation(user_id=owner.id, project_id=a.id)
        unrelated = Conversation(user_id=owner.id, project_id=b.id)
        db.add_all([conv, second, unrelated])
        db.commit()
    yield SimpleNamespace(
        engine=engine,
        sessions=lambda: OfflineSession(engine),
        owner=owner,
        other=other,
        a=a,
        b=b,
        foreign=foreign,
        conv=conv,
        second=second,
        unrelated=unrelated,
        settings=SimpleNamespace(
            memory_recent_token_budget=2000, memory_long_term_token_budget=1600
        ),
    )
    engine.dispose()


def add_turn(db, env, conv, question="研究低秩适配", answer="需要核对消融实验。"):
    run = ChatRun(
        user_id=env.owner.id,
        conversation_id=conv.id,
        idempotency_key=uuid4().hex,
        request_hash="test",
        question=question,
        mode="fast",
        scope={},
        config={},
        status="completed",
    )
    db.add(run)
    db.flush()
    q = Message(
        user_id=env.owner.id,
        conversation_id=conv.id,
        run_id=run.id,
        seq=conv.next_seq,
        role="user",
        content=question,
    )
    a = Message(
        user_id=env.owner.id,
        conversation_id=conv.id,
        run_id=run.id,
        seq=conv.next_seq + 1,
        role="assistant",
        content=answer,
    )
    conv.next_seq += 2
    db.add_all([q, a])
    db.flush()
    return run, q, a


def add_evidence(db, env, message):
    collection = Collection(user_id=env.owner.id, name="论文库")
    db.add(collection)
    db.flush()
    paper = Paper(
        user_id=env.owner.id,
        collection_id=collection.id,
        title="LoRA",
        sha256=uuid4().hex,
        file_path="/test/paper.pdf",
        size_bytes=10,
        page_count=1,
    )
    db.add(paper)
    db.flush()
    job = ParseJob(user_id=env.owner.id, paper_id=paper.id, result_path="/test", config={})
    db.add(job)
    db.flush()
    version = PaperVersion(
        user_id=env.owner.id,
        paper_id=paper.id,
        parse_job_id=job.id,
        operation="parse",
        embedding_provider="test",
        embedding_model="test",
        embedding_dim=1024,
    )
    db.add(version)
    db.flush()
    paper.active_version_id = version.id
    chunk = Chunk(
        user_id=env.owner.id,
        collection_id=collection.id,
        paper_id=paper.id,
        version_id=version.id,
        ordinal=0,
        kind="text",
        text="在该实验设置下减少训练参数。",
        embedding_input="test",
        section_path=["method"],
        locations=[{"page_idx": 0}],
        page_start=0,
        page_end=0,
        token_count=15,
        content_hash="test",
        indexable=True,
    )
    db.add(chunk)
    db.flush()
    citation = Citation(
        user_id=env.owner.id,
        message_id=message.id,
        number=1,
        paper_id=paper.id,
        version_id=version.id,
        chunk_id=chunk.id,
        source_text=chunk.text,
        locations=chunk.locations,
    )
    db.add(citation)
    db.flush()
    return paper, version, chunk, citation


def test_long_answer_preserves_question_and_is_explicitly_compacted():
    history = [
        {"role": "user", "content": "这个方法为什么有效？"},
        {"role": "assistant", "content": "证据" * 10000},
    ]
    chosen = trim_history(history, 128)
    assert [m["role"] for m in chosen] == ["user", "assistant"]
    assert chosen[0]["content"] == history[0]["content"]
    assert "已压缩" in chosen[1]["content"] and history_tokens(chosen) <= 128
    assert len(history[1]["content"]) == 20000


def test_recent_history_keeps_complete_turns_and_chinese_capacity():
    old = [
        {"role": "user", "content": "旧问题" * 60},
        {"role": "assistant", "content": "旧回答" * 60},
    ]
    latest = [
        {"role": "user", "content": "为什么？"},
        {"role": "assistant", "content": "这是回答。"},
    ]
    assert trim_history(old + latest, 128) == latest
    assert trim_history(latest, 10) == []
    assert estimate_tokens("中文" * 500) == 1000
    assert compact_text("汉字😀and equations " * 100, 120).endswith("equations ")
    assert estimate_tokens(compact_text("汉字😀and equations " * 100, 120)) <= 120


async def test_recall_isolates_users_projects_and_conversations(memory_db):
    e = memory_db
    with Session(e.engine) as db:
        db.add_all(
            [
                MemoryEntry(
                    user_id=e.owner.id,
                    level="project",
                    project_id=e.a.id,
                    kind="goal",
                    content="研究低秩适配",
                    status="active",
                ),
                MemoryEntry(
                    user_id=e.owner.id,
                    level="project",
                    project_id=e.b.id,
                    kind="goal",
                    content="另一个课题",
                    status="active",
                ),
                MemoryEntry(
                    user_id=e.other.id,
                    level="project",
                    project_id=e.foreign.id,
                    kind="goal",
                    content="其他用户秘密",
                    status="active",
                ),
                MemoryEntry(
                    user_id=e.owner.id,
                    level="conversation",
                    conversation_id=e.conv.id,
                    project_id=e.a.id,
                    kind="goal",
                    content="临时讨论",
                    status="active",
                ),
                MemoryEntry(
                    user_id=e.owner.id,
                    level="user",
                    kind="preference",
                    content="用中文解释",
                    status="active",
                ),
                MemoryEntry(
                    user_id=e.owner.id,
                    level="project",
                    project_id=e.a.id,
                    kind="goal",
                    content="已纠正观点",
                    status="superseded",
                ),
            ]
        )
        db.commit()
    async with e.sessions() as db:
        conv = await db.get(Conversation, e.second.id)
        recalled = await recall(db, e.settings, conv, {}, "继续上次研究")
    assert {m["content"] for m in recalled} == {
        "研究低秩适配",
        "用中文解释",
        "研究项目：低秩适配。",
    }


async def test_evidence_update_deletion_and_scope_revoke_recall(memory_db):
    e = memory_db
    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        _, _, answer = add_turn(db, e, conv)
        paper, version, chunk, citation = add_evidence(db, e, answer)
        entry = MemoryEntry(
            user_id=e.owner.id,
            level="project",
            project_id=e.a.id,
            kind="conclusion",
            content="低秩适配实验判断",
            status="active",
            epistemic="evidence_linked",
            source_message_id=answer.id,
            citation_ids=[str(citation.id)],
        )
        db.add(entry)
        db.commit()
    async with e.sessions() as db:
        conv = await db.get(Conversation, e.second.id)
        assert (
            len(
                [
                    m
                    for m in await recall(db, e.settings, conv, {}, "低秩适配")
                    if m.get("citation_ids")
                ]
            )
            == 1
        )
        assert not [
            m
            for m in await recall(db, e.settings, conv, {"paper_ids": []}, "低秩适配")
            if m.get("citation_ids")
        ]
        live = await db.get(Chunk, chunk.id)
        live.text = "解析已更新"
        await db.commit()
        assert evidence_status(entry, await evidence_map(db, e.owner.id, [entry])) == "outdated"
        assert not [
            m for m in await recall(db, e.settings, conv, {}, "低秩适配") if m.get("citation_ids")
        ]
        live.text = citation.source_text
        p = await db.get(Paper, paper.id)
        p.deleted_at = utcnow()
        await db.commit()
        assert evidence_status(entry, await evidence_map(db, e.owner.id, [entry])) == "unavailable"
        assert not [
            m for m in await recall(db, e.settings, conv, {}, "低秩适配") if m.get("citation_ids")
        ]


async def test_candidates_require_user_origin_and_tombstones_do_not_resurrect(memory_db):
    e = memory_db
    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        _, question, answer = add_turn(db, e, conv)
        db.commit()
    state = ResearchState(
        goal=[{"content": "研究低秩适配", "source_seq": question.seq}],
        next_step=[{"content": "助手建议不能当作已接受决策", "source_seq": answer.seq}],
    )
    async with e.sessions() as db:
        conv = await db.get(Conversation, e.conv.id)
        await save_candidates(db, conv, state, [question, answer])
        await db.commit()
        entries = list(await db.scalars(select(MemoryEntry)))
        assert len(entries) == 1 and entries[0].status == "candidate"
        entries[0].deleted_at = utcnow()
        await db.commit()
        await save_candidates(db, conv, state, [question, answer])
        await db.commit()
        assert len(list(await db.scalars(select(MemoryEntry)))) == 1
        assert not [
            m
            for m in await recall(db, e.settings, conv, {}, "继续研究")
            if not m["id"].startswith("project:")
        ]


async def test_token_pressure_consolidates_before_history_is_lost(memory_db):
    e = memory_db
    e.settings.memory_recent_token_budget = 128
    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        for _ in range(3):
            run, _, _ = add_turn(db, e, conv, answer="很长的回答" * 100)
        db.commit()
    calls = []

    class Models:
        async def invoke(self, run, node, payload, config, schema):
            calls.append(payload)
            seq = payload["history"][0]["seq"]
            return schema.model_validate(
                {
                    "summary": "用户在研究注意力。" * 180,
                    "research_state": {
                        "goal": [{"content": "研究低秩适配", "source_seq": seq}],
                        "focus": [{"content": "伪造的来源必须被丢弃", "source_seq": 99999}],
                    },
                }
            )

    services = SimpleNamespace(sessions=e.sessions, settings=e.settings, models=Models())
    assert await consolidate(services, run, {}, pressure_only=True)
    memory = await load_memory(e.sessions, e.settings, e.owner.id, e.conv.id, {}, "为什么？")
    assert memory["summary"].endswith("注意力。") and len(memory["summary"].encode()) > 4000
    assert memory["memory_coverage"]["summarized_seq"] == 4
    assert memory["research_state"]["goal"] and not memory["research_state"].get("focus")
    assert [m["role"] for m in memory["history"]] == ["user", "assistant"]
    assert history_tokens(memory["history"]) <= 128
    assert len(calls) == 1


async def test_failed_summary_preserves_raw_history_and_summary_watermark(memory_db):
    e = memory_db
    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        for _ in range(4):
            run, _, _ = add_turn(db, e, conv)
        db.commit()

    class Models:
        async def invoke(self, *args, **kwargs):
            raise TimeoutError("offline provider failure")

    with pytest.raises(TimeoutError):
        await consolidate(
            SimpleNamespace(sessions=e.sessions, settings=e.settings, models=Models()), run, {}
        )
    async with e.sessions() as db:
        conv = await db.get(Conversation, e.conv.id)
        assert conv.summarized_seq == 0 and conv.summary == ""
        assert len(list(await db.scalars(select(Message)))) == 8


def test_recall_budget_and_corrected_memory_precedence():
    entries = [
        dict(
            id=str(i),
            content="低秩适配" * 30,
            kind="note",
            pinned=False,
            status="active",
            updated_at=f"2026-11-{i:02d}",
        )
        for i in range(1, 20)
    ]
    entries.append(
        dict(
            id="correction",
            content="纠正：研究目标改为注意力",
            kind="correction",
            pinned=True,
            status="active",
            updated_at="2026-11-02",
        )
    )
    chosen = select_memories(entries, "低秩适配", 400)
    assert chosen[0]["id"] == "correction"
    assert sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) + 8 for m in chosen) <= 400


async def test_memory_crud_candidate_confirmation_and_cross_user_access(memory_db):
    e = memory_db
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    current_user = [e.owner]

    async def db_dependency():
        async with e.sessions() as db:
            yield db

    async def identity_dependency():
        return Identity(current_user[0], None)

    app.dependency_overrides[get_db] = db_dependency
    app.dependency_overrides[require_identity] = identity_dependency
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/memories",
            json={
                "level": "project",
                "project_id": str(e.a.id),
                "kind": "goal",
                "content": "低秩适配目标",
                "status": "candidate",
            },
        )
        assert response.status_code == 201, response.text
        mid = response.json()["id"]
        assert (
            await client.patch(f"/api/v1/memories/{mid}", json={"status": "active", "pinned": True})
        ).json()["pinned"]
        assert (
            await client.patch(
                f"/api/v1/memories/{mid}",
                json={"content": "用户修正后的目标", "epistemic": "hypothesis"},
            )
        ).json()["epistemic"] == "hypothesis"
        current_user[0] = e.other
        assert (
            await client.patch(f"/api/v1/memories/{mid}", json={"pinned": False})
        ).status_code == 404
        assert (await client.get(f"/api/v1/memories?project_id={e.a.id}")).status_code == 404
        current_user[0] = e.owner
        assert (await client.delete(f"/api/v1/memories/{mid}")).status_code == 204
        assert (await client.get("/api/v1/memories")).json() == []
        assert (
            await client.patch(f"/api/v1/memories/{mid}", json={"status": "active"})
        ).status_code == 404


async def test_invalid_source_links_and_level_shapes_are_rejected(memory_db):
    e = memory_db
    async with e.sessions() as db:
        with pytest.raises(HTTPException) as exc:
            await validate_memory(
                db, e.owner.id, MemoryInput(level="user", project_id=e.a.id, content="invalid")
            )
        assert exc.value.status_code == 422
        with pytest.raises(HTTPException) as exc:
            await validate_memory(
                db,
                e.owner.id,
                MemoryInput(level="project", project_id=e.foreign.id, content="invalid"),
            )
        assert exc.value.status_code == 404
        with pytest.raises(HTTPException) as exc:
            await validate_memory(
                db,
                e.owner.id,
                MemoryInput(
                    level="project", project_id=e.a.id, content="invalid", source_message_id=uuid4()
                ),
            )
        assert exc.value.status_code == 404


def test_planning_receives_research_state_without_promoting_it_to_evidence():
    state = RAGState(
        question="继续验证",
        research_state={"goal": [{"content": "低秩适配", "source_seq": 1}]},
        memories=[{"content": "假设", "epistemic": "hypothesis", "status": "candidate"}],
    )
    plan = node_payload("plan_paper_tools", state, max_search_probes=3)
    verify = node_payload("verify_grounding", state, max_search_probes=3)
    assert (
        plan["research_state"] == state["research_state"] and plan["memories"] == state["memories"]
    )
    assert "memories" not in verify and verify["sources"] == []


async def test_project_switch_and_deletion_reset_context_and_block_active_runs(memory_db):
    from app.api.v1.chat import router as chat_router

    e = memory_db
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.state.session_factory = e.sessions
    app.state.settings = e.settings

    async def db_dependency():
        async with e.sessions() as db:
            yield db

    async def identity_dependency():
        return Identity(e.owner, None)

    app.dependency_overrides[get_db] = db_dependency
    app.dependency_overrides[require_identity] = identity_dependency
    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        run, _, _ = add_turn(db, e, conv)
        conv.summary = "旧课题的上下文"
        conv.research_state = {"goal": [{"content": "旧目标", "source_seq": 1}]}
        db.commit()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.patch(
            f"/api/v1/conversations/{e.conv.id}", json={"project_id": str(e.b.id)}
        )
        assert response.status_code == 200, response.text
        state = (await client.get(f"/api/v1/conversations/{e.conv.id}/memory")).json()
        assert (
            state["summary"] == ""
            and state["research_state"] == {}
            and state["summarized_seq"] == 2
        )
        assert (await client.get(f"/api/v1/conversations/{e.conv.id}/messages")).json()[0][
            "content"
        ]
        with Session(e.engine) as db:
            db.get(ChatRun, run.id).status = "running"
            db.commit()
        response = await client.patch(
            f"/api/v1/conversations/{e.conv.id}", json={"project_id": str(e.a.id)}
        )
        assert response.status_code == 409
        with Session(e.engine) as db:
            db.get(ChatRun, run.id).status = "completed"
            db.commit()
        assert (await client.delete(f"/api/v1/projects/{e.b.id}")).status_code == 204
        assert (await client.get(f"/api/v1/conversations/{e.conv.id}/memory")).json()[
            "project_id"
        ] is None


async def test_concurrent_manual_state_edit_wins_over_slow_summary(memory_db):
    e = memory_db
    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        for _ in range(3):
            run, _, _ = add_turn(db, e, conv)
        db.commit()

    class Models:
        async def invoke(self, *args, schema, **kwargs):
            with Session(e.engine) as db:
                item = db.get(Conversation, e.conv.id)
                item.summary = "用户手动修正后的摘要"
                item.memory_revision += 1
                db.commit()
            return schema.model_validate({"summary": "旧模型输出", "research_state": {}})

    assert not await consolidate(
        SimpleNamespace(sessions=e.sessions, settings=e.settings, models=Models()), run, {}
    )
    async with e.sessions() as db:
        item = await db.get(Conversation, e.conv.id)
        assert item.summary == "用户手动修正后的摘要" and item.summarized_seq == 0


async def test_short_project_session_is_resumable_before_summary_threshold(memory_db):
    from app.services.memory.store import record_project_question

    e = memory_db
    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        _, question, _ = add_turn(db, e, conv, question="下次继续比较LoRA与Adapter的消融实验")
        db.commit()
    async with e.sessions() as db:
        conv = await db.get(Conversation, e.conv.id)
        await record_project_question(db, conv, question)
        await db.commit()
        second = await db.get(Conversation, e.second.id)
        recalled = await recall(db, e.settings, second, {}, "继续上次研究")
        episode = next(m for m in recalled if m["kind"] == "discussed")
        assert "LoRA与Adapter" in episode["content"] and episode["status"] == "candidate"
        # Current-session messages are already in history; don't inject them twice.
        assert not [
            m
            for m in await recall(db, e.settings, conv, {}, "继续上次研究")
            if m["kind"] == "discussed"
        ]


async def test_reading_position_tracks_version_and_can_be_forgotten(memory_db):
    e = memory_db
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    async def db_dependency():
        async with e.sessions() as db:
            yield db

    async def identity_dependency():
        return Identity(e.owner, None)

    app.dependency_overrides[get_db] = db_dependency
    app.dependency_overrides[require_identity] = identity_dependency
    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        _, _, answer = add_turn(db, e, conv)
        paper, version, _, _ = add_evidence(db, e, answer)
        paper.page_count = 10
        db.commit()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        endpoint = f"/api/v1/papers/{paper.id}/reading-position"
        response = await client.put(endpoint, json={"version_id": str(version.id), "page_idx": 4})
        assert response.status_code == 200 and response.json()["position"]["page_idx"] == 4
        assert (await client.get(endpoint)).json()["current"]
        async with e.sessions() as db:
            conv = await db.get(Conversation, e.second.id)
            positions = [
                m
                for m in await recall(db, e.settings, conv, {}, "继续阅读")
                if m.get("reading_anchor")
            ]
            assert len(positions) == 1 and positions[0]["reading_anchor"]["page_idx"] == 4
        assert (
            await client.put(endpoint, json={"version_id": str(uuid4()), "page_idx": 4})
        ).status_code == 409
        assert (
            await client.put(endpoint, json={"version_id": str(version.id), "page_idx": 10})
        ).status_code == 422
        with Session(e.engine) as db:
            db.get(Paper, paper.id).active_version_id = None
            db.commit()
        assert not (await client.get(endpoint)).json()["current"]
        async with e.sessions() as db:
            conv = await db.get(Conversation, e.second.id)
            assert not [
                m
                for m in await recall(db, e.settings, conv, {}, "继续阅读")
                if m.get("reading_anchor")
            ]
        items = (await client.get("/api/v1/memories")).json()
        position = next(m for m in items if m["reading_anchor"])
        assert position["evidence_status"] == "outdated"
        assert (await client.delete(f"/api/v1/memories/{position['id']}")).status_code == 204
        assert (await client.get(endpoint)).json()["position"] is None


def test_summary_output_budget_matches_provider_contract():
    from pydantic import ValidationError

    from app.core.config import Settings
    from app.graph.prompts import Summary
    from app.services.chat.llm import ChatModels

    settings = Settings(
        _env_file=None,
        app_env="test",
        tracing_enabled=False,
        llm_api_key="test-only-no-network",
        llm_fast_api_key="test-only-no-network",
    )
    models = ChatModels(settings, None, None)
    assert models.fast.extra_body["max_tokens"] == 1024
    assert models.summarizer.extra_body["max_tokens"] == 4096
    with pytest.raises(ValidationError):
        Summary.model_validate(
            {
                "summary": "中" * 2300,
                "research_state": {"goal": [{"content": "中" * 200, "source_seq": 1}]},
            }
        )


async def test_core_memory_curate_turn_dedup_undo_and_session_search(memory_db):
    from app.services.memory.tools import curate_user_turn

    e = memory_db
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    async def db_dependency():
        async with e.sessions() as db:
            yield db

    async def identity_dependency():
        return Identity(e.owner, None)

    app.dependency_overrides[get_db] = db_dependency
    app.dependency_overrides[require_identity] = identity_dependency

    with Session(e.engine, expire_on_commit=False) as db:
        conv = db.get(Conversation, e.conv.id)
        run1, q1, _ = add_turn(
            db,
            e,
            conv,
            question="我的研究目标是对比LoRA与DoRA在低资源微调下的收敛稳定性",
            answer="我们可以先检索LoRA与DoRA的实验章节。",
        )
        run2, q2, _ = add_turn(
            db,
            e,
            conv,
            question="我的研究目标是对比LoRA与DoRA在低资源微调下的收敛稳定性！",
            answer="好的，已记录。",
        )
        run3, q3, _ = add_turn(
            db,
            e,
            conv,
            question="这个架构的注意力复杂度是多少？",
            answer="是O(N^2)。",
        )
        db.commit()

    async with e.sessions() as db:
        conv = await db.get(Conversation, e.conv.id)
        events1 = await curate_user_turn(db, conv, q1, run1.id)
        assert len(events1) == 1
        assert events1[0]["action"] == "add"
        # Near-duplicate statement is merged without creating duplicate entries
        events2 = await curate_user_turn(db, conv, q2, run2.id)
        assert events2 == []
        # Ordinary question does NOT spam candidate memories
        events3 = await curate_user_turn(db, conv, q3, run3.id)
        assert events3 == []
        await db.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Check memories list and filters
        memories = (await client.get(f"/api/v1/memories?project_id={e.a.id}&kind=goal")).json()
        assert len(memories) == 1
        assert memories[0]["status"] == "active"

        # Check core memory was auto-populated for the project
        core = (await client.get(f"/api/v1/memory/core?project_id={e.a.id}")).json()
        assert "LoRA与DoRA" in core["project"]["content"]
        assert core["profile"]["char_limit"] == 1500
        assert core["project"]["char_limit"] == 3000

        # Update user profile core memory and verify capacity limit enforcement
        res = await client.put(
            "/api/v1/memory/core?level=user",
            json={"content": "- 偏好严谨数学推导与消融表格对比"},
        )
        assert res.status_code == 200 and "数学推导" in res.json()["content"]
        over_limit = await client.put(
            "/api/v1/memory/core?level=user",
            json={"content": "超" * 1501},
        )
        assert over_limit.status_code == 422

        # Cross-session transcript search finds the earlier turn from another conversation
        hits = (
            await client.get(
                f"/api/v1/memory/sessions/search?q=DoRA&project_id={e.a.id}&project_only=true"
            )
        ).json()
        assert len(hits) >= 1 and "DoRA" in hits[0]["snippet"]

        # Load memory for second conversation in the same project includes core_memory & session_recall
        loaded = await load_memory(
            e.sessions, e.settings, e.owner.id, e.second.id, {}, "继续对比DoRA收敛稳定性"
        )
        assert "数学推导" in loaded["core_memory"]["profile"]
        assert "LoRA与DoRA" in loaded["core_memory"]["project"]["notebook"]
        assert len(loaded["session_recall"]) >= 1
        verify_payload = node_payload(
            "verify_grounding",
            RAGState(
                question="继续对比",
                core_memory=loaded["core_memory"],
                session_recall=loaded["session_recall"],
                memories=loaded["memories"],
            ),
            max_search_probes=3,
        )
        assert "core_memory" not in verify_payload
        assert "session_recall" not in verify_payload

        # Undo the auto-created memory event
        undo_res = await client.post(f"/api/v1/memory/events/{events1[0]['id']}/undo")
        assert undo_res.status_code == 200 and undo_res.json()["undone_at"] is not None
        after_undo = (await client.get(f"/api/v1/memories?project_id={e.a.id}&kind=goal")).json()
        assert after_undo == []

        # Test bulk operations and cleanup-discussed
        m_a = (
            await client.post(
                "/api/v1/memories",
                json={
                    "level": "project",
                    "project_id": str(e.a.id),
                    "kind": "discussed",
                    "content": "旧自动问题噪声",
                    "status": "candidate",
                },
            )
        ).json()
        m_b = (
            await client.post(
                "/api/v1/memories",
                json={
                    "level": "project",
                    "project_id": str(e.a.id),
                    "kind": "conclusion",
                    "content": "待批量确认的研究发现",
                    "status": "candidate",
                },
            )
        ).json()
        bulk_res = await client.post(
            "/api/v1/memories/bulk",
            json={"ids": [m_b["id"]], "action": "confirm"},
        )
        assert bulk_res.status_code == 200 and bulk_res.json()["updated"] == 1
        clean_res = await client.post(f"/api/v1/memories/cleanup-discussed?project_id={e.a.id}")
        assert clean_res.status_code == 200 and clean_res.json()["cleaned"] == 1
        remaining = (await client.get(f"/api/v1/memories?project_id={e.a.id}")).json()
        assert [item["id"] for item in remaining] == [m_b["id"]]
        assert remaining[0]["status"] == "active"
        assert m_a["id"] not in [item["id"] for item in remaining]
