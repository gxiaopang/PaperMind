"""Current real-model tool/chat smoke; all writes use disposable Postgres/Redis."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from m3_eval import import_corpus, require_reviewed, usage_rows  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.security import hash_token  # noqa: E402
from app.db.models import Session, User, utcnow  # noqa: E402
from app.eval.retrieval import load_dataset  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.chat.store import CONTEXT_STRATEGY  # noqa: E402
from app.workers.chat import execute, shutdown, startup  # noqa: E402

CASES = [
    dict(
        id="fast_formula",
        paper="attention",
        mode="fast",
        question="Scaled Dot-Product Attention 为什么除以根号 d_k？请用两句话解释并引用。",
        status="answered",
    ),
    dict(
        id="agentic_section",
        paper="lora",
        mode="agentic",
        question="请直接读取 LoRA 的 4.1 LOW-RANK-PARAMETRIZED UPDATE 小节全文，说明冻结和训练哪些参数，用两句话回答并引用。",
        status="answered",
        tool="read_paper_section",
    ),
    dict(
        id="follow_up",
        paper="lora",
        mode="fast",
        question="它在部署推理时会增加延迟吗？请用两句话回答并引用。",
        status="answered",
    ),
    dict(
        id="table",
        paper="attention",
        mode="fast",
        question="请精确调阅 Table 2 的完整表格及正文分析，base 和 big 的 EN-DE BLEU 各是多少？只用一句话回答并引用。",
        status="answered",
        tool="inspect_element",
    ),
    dict(
        id="formula_element",
        paper="lora",
        mode="fast",
        question="请精确调阅 Equation (3)，给出 LoRA 的前向计算公式及矩阵含义，用两句话回答并引用。",
        status="answered",
        tool="inspect_element",
    ),
    dict(
        id="idea_boundaries",
        paper="lora",
        mode="agentic",
        question="基于 LoRA 提出一个研究思路，先读取方法全文和局限全文；区分原文事实与建议，三句话以内并引用。",
        status="answered",
        boundary=True,
    ),
    dict(
        id="multi_probe",
        paper="all",
        mode="fast",
        question="比较 Transformer 和 LoRA 的隐性缺点、计算开销和失败案例。分探针核对，区分论文归属，只回答原文支持的两点并引用。",
        status="answered",
        multi_probe=True,
    ),
    dict(
        id="unsupported",
        paper="attention",
        mode="agentic",
        question="Transformer 这篇论文报告的 GSM8K pass@1 是多少？没有报告就明确说明，不能用 BLEU 替代。",
        status="abstain",
    ),
    dict(
        id="empty_scope",
        paper="lora",
        mode="fast",
        question="论文的训练结果是什么？",
        status="insufficient",
        scope={"paper_ids": []},
    ),
    dict(id="greeting", paper="attention", mode="fast", question="你好", status="chitchat"),
]


class ObservedModels:
    """Observe real calls without replacing providers, results or tool choices."""

    def __init__(self, models):
        self.models, self.runs = models, {}

    async def invoke(self, run, node, payload, *args, **kwargs):
        record = self.runs.setdefault(
            str(run.id), dict(plans=[], reports=[], history_messages=0, maps=0)
        )
        record["history_messages"] = max(
            record["history_messages"], len(payload.get("history", []))
        )
        record["maps"] = max(record["maps"], len(payload.get("paper_maps", [])))
        if payload.get("tool_reports"):
            # One record per retrieval attempt; only public tool arguments and
            # statuses, not hidden model text or chain-of-thought, are retained.
            record["reports"] = [
                {k: v for k, v in report.items() if k != "parents"}
                for report in payload["tool_reports"]
            ]
        try:
            result = await self.models.invoke(run, node, payload, *args, **kwargs)
        except Exception as exc:
            body = getattr(exc, "body", None)
            detail = body.get("error", body) if isinstance(body, dict) else {}
            detail = detail if isinstance(detail, dict) else {}
            message = str(detail.get("message", ""))
            settings = getattr(self.models, "settings", None)
            if settings:
                for key in (settings.llm_api_key, settings.llm_fast_api_key):
                    secret = key.get_secret_value()
                    if secret:
                        message = message.replace(secret, "[REDACTED]")
            message = message[:400]
            record.setdefault("failures", []).append(
                {
                    "node": node,
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                    "provider_error_code": detail.get("code"),
                    "provider_error_message": message,
                    "validation_issues": getattr(exc, "issues", []),
                }
            )
            raise
        if node == "plan_paper_tools":
            record["plans"].append(result.model_dump(mode="json"))
        return result


def assess(case, result, observation, events, allowed_papers, max_probes):
    message = result.get("message") or {}
    citations = message.get("citations", [])
    used = {int(n) for n in re.findall(r"\[(\d+)\]", message.get("content", ""))}
    numbers = {c["number"] for c in citations}
    reports = observation.get("reports", [])
    checks = {
        "completed": result["status"] == "completed",
        "citation_numbers_valid": used == numbers
        and all(not c["deleted"] and c["text"] for c in citations),
        "citation_scope_valid": all(c["paper_id"] in allowed_papers for c in citations),
        "bounded_retrieval": result["retrieval_count"] <= (3 if case["mode"] == "agentic" else 1),
        "bounded_generation": result["generation_count"] <= 2,
        "terminal_sse": any(e.get("kind") == "done" for e in events),
        "probe_bounds": all(
            len(call["args"].get("queries", [])) <= max_probes
            for plan in observation.get("plans", [])
            for call in plan["calls"]
        ),
    }
    status = result["result_status"]
    if case["status"] == "abstain":
        checks["abstention_smoke"] = status in {"insufficient", "safe_fallback"} or (
            status == "answered"
            and bool(
                re.search(
                    r"未报告|没有报告|未提供|证据不足|无法给出|未提及", message.get("content", "")
                )
            )
        )
    else:
        checks["expected_status"] = status == case["status"]
    if case["status"] == "answered":
        checks["has_citations"] = bool(citations)
        checks["map_mounted"] = observation.get("maps", 0) > 0
    if case.get("tool"):
        checks["target_tool_used"] = any(
            r["tool"] == case["tool"] and r["status"] == "ok" for r in reports
        )
    if case.get("boundary"):
        checks["method_and_limitations_attempted"] = {"method", "limitations"} <= {
            r.get("arguments", {}).get("section_std")
            for r in reports
            if r["tool"] == "read_paper_section" and r.get("arguments", {}).get("mode") == "full"
        }
    if case.get("multi_probe"):
        checks["multiple_search_probes"] = any(
            len(r.get("arguments", {}).get("queries", [])) > 1
            for r in reports
            if r["tool"] == "search_paper_chunks"
        )
    if case["id"] == "follow_up":
        checks["history_loaded"] = observation.get("history_messages", 0) >= 2
    return checks


async def smoke(settings, dataset_path, cache_path, output, selected=None):
    dataset, corpus = load_dataset(dataset_path)
    require_reviewed(dataset)
    cache = json.loads(cache_path.read_bytes())
    identity = dict(
        provider=settings.embedding_base_url.rstrip("/"),
        model=settings.embedding_model,
        dimension=settings.embedding_dim,
    )
    if cache["identity"] != identity:
        raise ValueError("Embedding cache identity mismatch")
    missing = [
        r["content_hash"]
        for p in corpus.values()
        for r in p["chunks"]
        if r["indexable"] and r["content_hash"] not in cache["vectors"]
    ]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} current corpus vectors; finish retrieval evaluation first"
        )
    report = dict(
        status="running",
        dataset=dataset["version"],
        dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        execution_source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "scripts/pre_m6_live.py",
                ROOT / "backend/app/services/chat/llm.py",
                ROOT / "backend/app/services/chat/store.py",
                ROOT / "backend/app/graph/nodes.py",
                ROOT / "backend/app/graph/context.py",
                ROOT / "backend/app/graph/prompts.py",
                *sorted((ROOT / "backend/app/services/retrieval").glob("*.py")),
            ]
        },
        strategy=CONTEXT_STRATEGY,
        models=dict(main=settings.llm_model, fast=settings.llm_fast_model),
        cases=[],
        scope="Real providers and production graph; smoke only. No Faithfulness score or independent human answer-support acceptance. Abstention language check is lexical, not semantic grading.",
    )

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        output.chmod(0o600)

    save()
    app = create_app(settings)
    started = time.monotonic()
    async with app.router.lifespan_context(app):
        owner, foreign, csrf, token = uuid4(), uuid4(), uuid4().hex, uuid4().hex
        async with app.state.session_factory() as db:
            for uid, email in [
                (owner, "current-smoke@example.org"),
                (foreign, "isolation-smoke@example.org"),
            ]:
                db.add(
                    User(id=uid, email=email, password_hash="no-login", email_verified_at=utcnow())
                )
            await db.flush()
            db.add(
                Session(
                    user_id=owner,
                    token_hash=hash_token(token),
                    csrf_hash=hash_token(csrf),
                    expires_at=utcnow() + timedelta(hours=2),
                )
            )
            await db.commit()
        paper_ids, _ = await import_corpus(
            app.state.session_factory, dataset, corpus, identity, cache["vectors"], owner, foreign
        )
        ctx = {"redis": app.state.redis}
        await startup(ctx)
        observed = ObservedModels(ctx["services"].models)
        ctx["services"].models = observed
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url=settings.origin,
                timeout=600,
                headers={"Origin": settings.origin, "X-CSRF-Token": csrf},
                cookies={"pm_session": token, "pm_csrf": csrf},
            ) as client:
                conversations = {}
                for name in ["attention", "lora", "all"]:
                    response = await client.post(
                        "/api/v1/conversations",
                        json={
                            "scope": {
                                "paper_ids": [str(p) for p in paper_ids.values()]
                                if name == "all"
                                else [str(paper_ids[name])]
                            }
                        },
                    )
                    response.raise_for_status()
                    conversations[name] = response.json()["id"]
                for case in CASES:
                    if selected and case["id"] not in selected:
                        continue
                    print(json.dumps({"starting": case["id"], "mode": case["mode"]}), flush=True)
                    began = time.monotonic()
                    body = dict(
                        idempotency_key=case["id"], mode=case["mode"], question=case["question"]
                    )
                    if "scope" in case:
                        body["scope"] = case["scope"]
                    response = await client.post(
                        f"/api/v1/conversations/{conversations[case['paper']]}/runs", json=body
                    )
                    response.raise_for_status()
                    rid = response.json()["run_id"]
                    await execute(ctx, str(owner), rid)
                    response = await client.get(f"/api/v1/chat/runs/{rid}")
                    response.raise_for_status()
                    result = response.json()
                    replay = await client.get(f"/api/v1/chat/runs/{rid}/events")
                    replay.raise_for_status()
                    events = [
                        {"kind": block.splitlines()[1].removeprefix("event: ")}
                        for block in replay.text.strip().split("\n\n")
                        if len(block.splitlines()) >= 2
                    ]
                    observation = observed.runs.get(rid, {})
                    allowed = (
                        {str(p) for p in paper_ids.values()}
                        if case["paper"] == "all"
                        else {str(paper_ids[case["paper"]])}
                    )
                    if case.get("scope") == {"paper_ids": []}:
                        allowed = set()
                    checks = assess(
                        case,
                        result,
                        observation,
                        events,
                        allowed,
                        settings.retrieval_max_subqueries,
                    )
                    row = dict(
                        case=case["id"],
                        mode=case["mode"],
                        passed=all(checks.values()),
                        checks=checks,
                        result=result,
                        observation=observation,
                        elapsed_s=round(time.monotonic() - began, 2),
                    )
                    report["cases"].append(row)
                    save()
                    print(
                        json.dumps(
                            {k: row[k] for k in ["case", "passed", "checks", "elapsed_s"]},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if any(
                        failure.get("http_status") == 429
                        or failure.get("error_type", "").endswith("RateLimitError")
                        for failure in observation.get("failures", [])
                    ):
                        report["stopped_on_provider_rate_limit"] = True
                        break
            report.update(
                status="provider_rate_limited"
                if report.get("stopped_on_provider_rate_limit")
                else "passed"
                if all(r["passed"] for r in report["cases"])
                else "failed",
                elapsed_s=round(time.monotonic() - started, 2),
                usage=await usage_rows(app.state.session_factory),
            )
        except Exception as exc:
            # Preserve completed cases without publishing provider bodies or credentials.
            report.update(
                status="interrupted",
                error_type=type(exc).__name__,
                elapsed_s=round(time.monotonic() - started, 2),
            )
            raise
        finally:
            await shutdown(ctx)
            save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-paid", action="store_true", required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/datasets/m3-gold-v7.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "reports/m3-local/embeddings.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/m3-local/pre-m6-live-v7.json"
    )
    parser.add_argument("--cases", nargs="+", choices=[c["id"] for c in CASES])
    args = parser.parse_args()
    data, _ = load_dataset(args.dataset)
    require_reviewed(data)
    if not args.cache.is_file():
        parser.error("Missing reviewed-corpus embedding cache")
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve prior attempts")
    if args.cases and "follow_up" in args.cases and "agentic_section" not in args.cases:
        parser.error("follow_up requires agentic_section in the same run")
    with (
        PostgresContainer(
            "paradedb/paradedb:0.26.3-pg16",
            username="current_smoke",
            password="test-only",
            dbname="current_smoke",
        ) as pg,
        RedisContainer("redis:7.4.6-alpine") as redis,
    ):
        values = dict(
            DATABASE_URL=pg.get_connection_url().replace(
                "postgresql+psycopg2://", "postgresql+asyncpg://"
            ),
            REDIS_URL=f"redis://{redis.get_container_host_ip()}:{redis.get_exposed_port(6379)}/0",
            APP_ENV="test",
            PUBLIC_BASE_URL="http://localhost:8080",
            MAIL_BACKEND="memory",
            TRACING_ENABLED="false",
            USER_DAILY_MODEL_TOKENS="10000000",
            GLOBAL_DAILY_MODEL_TOKENS="20000000",
        )
        env = {**os.environ, **values}
        for command in [
            [str(ROOT / "backend/.venv/bin/alembic"), "upgrade", "head"],
            [sys.executable, "-m", "app.cli", "checkpoints"],
        ]:
            subprocess.run(command, cwd=ROOT / "backend", env=env, check=True)
        os.environ.update(values)
        get_settings.cache_clear()
        report = asyncio.run(smoke(Settings(), args.dataset, args.cache, args.output, args.cases))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
