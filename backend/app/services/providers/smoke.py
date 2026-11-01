"""Explicit, small live probes. No calls run during normal tests or application startup."""

import argparse
import asyncio
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import aiosmtplib
import httpx
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.observability import setup_logging, tracing_client
from app.services.providers.clients import chat_model, embeddings


class Probe(BaseModel):
    ok: Literal[True]


async def run(only: set[str] | None = None) -> dict[str, Any]:
    settings = get_settings()
    setup_logging()
    report: dict[str, Any] = {"checked_at": datetime.now(UTC).isoformat(), "checks": {}}
    checks = report["checks"]

    async def record(name: str, operation: Any) -> None:
        started = time.monotonic()
        try:
            checks[name] = {"status": "passed", **await operation()}
        except Exception as exc:
            # Do not persist error strings: HTTP/SMTP exceptions may embed credentials or content.
            info: dict[str, Any] = {"status": "failed", "error_type": type(exc).__name__}
            if isinstance(exc, httpx.HTTPStatusError):
                info["http_status"] = exc.response.status_code
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                info["http_status"] = status
            checks[name] = info
        checks[name]["elapsed_seconds"] = round(time.monotonic() - started, 2)

    async with httpx.AsyncClient(timeout=45, follow_redirects=False) as http:
        for kind in ("main", "fast"):
            if only and kind not in only and not any(name.startswith(kind + "_") for name in only):
                continue
            model = chat_model(settings, kind, max_tokens=32)  # type: ignore[arg-type]
            model_id = settings.llm_model if kind == "main" else settings.llm_fast_model
            base = settings.llm_base_url if kind == "main" else settings.llm_fast_base_url
            key = settings.llm_api_key if kind == "main" else settings.llm_fast_api_key

            async def available(
                base: str = base, key: Any = key, model_id: str = model_id
            ) -> dict[str, Any]:
                response = await http.get(
                    f"{base.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {key.get_secret_value()}"},
                )
                response.raise_for_status()
                ids = {row["id"] for row in response.json()["data"]}
                if model_id not in ids:
                    raise ValueError("Configured model absent; no fallback allowed")
                return {"model": model_id, "available": True}

            await record(f"{kind}_model", available)
            if checks[f"{kind}_model"]["status"] != "passed":
                for capability in ("text", "json", "stream"):
                    checks[f"{kind}_{capability}"] = {
                        "status": "blocked",
                        "reason": "configured_model_unavailable",
                    }
                continue

            async def text_call(model: Any = model, model_id: str = model_id) -> dict[str, Any]:
                answer = await model.ainvoke("Reply with the single word OK.")
                if not answer.content:
                    raise ValueError("Empty completion")
                return {"model": model_id, "usage": answer.usage_metadata}

            async def json_call(model: Any = model) -> dict[str, Any]:
                result = await model.with_structured_output(Probe, method="json_mode").ainvoke(
                    'Return JSON only: {"ok": true}'
                )
                if not isinstance(result, Probe) or not result.ok:
                    raise ValueError("Invalid structured result")
                return {"method": "json_mode", "schema_valid": True}

            async def stream_call(model: Any = model) -> dict[str, Any]:
                chunks = 0
                async for part in model.astream("Reply with the single word OK."):
                    if part.content:
                        chunks += 1
                if not chunks:
                    raise ValueError("No content chunks")
                return {"content_chunks": chunks}

            for capability, operation in (
                ("text", text_call),
                ("json", json_call),
                ("stream", stream_call),
            ):
                if not only or kind in only or f"{kind}_{capability}" in only:
                    await record(f"{kind}_{capability}", operation)

        async def embedding_call() -> dict[str, Any]:
            vectors = await embeddings(settings).aembed_documents(
                ["PaperMind M0 embedding compatibility check."]
            )
            dimension = len(vectors[0]) if len(vectors) == 1 else 0
            if dimension != settings.embedding_dim or not all(math.isfinite(x) for x in vectors[0]):
                raise ValueError("Embedding dimension mismatch or invalid values")
            return {
                "model": settings.embedding_model,
                "actual_dimension": dimension,
                "expected_dimension": settings.embedding_dim,
            }

        async def rerank_call() -> dict[str, Any]:
            response = await http.post(
                f"{settings.rerank_base_url.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {settings.rerank_api_key.get_secret_value()}"},
                json={
                    "model": settings.rerank_model,
                    "query": "research paper",
                    "documents": ["A scientific research paper.", "A cooking recipe."],
                    "top_n": 1,
                },
            )
            response.raise_for_status()
            result = response.json()["results"]
            if (
                len(result) != 1
                or result[0]["index"] != 0
                or not math.isfinite(result[0]["relevance_score"])
            ):
                raise ValueError("Invalid rerank result")
            return {"model": settings.rerank_model, "response_valid": True}

        async def mineru_call() -> dict[str, Any]:
            # Authenticated read against a deliberately nonexistent batch; no PDF is submitted.
            response = await http.get(
                f"{settings.mineru_api_base.rstrip('/')}/extract-results/batch/00000000-0000-0000-0000-000000000000",
                headers={"Authorization": f"Bearer {settings.mineru_api_token.get_secret_value()}"},
            )
            response.raise_for_status()
            data = response.json()
            return {
                "scope": "read_only_endpoint_reachable",
                "api_code": data.get("code"),
                "note": "Parse permission and a real PDF are verified in M1",
            }

        if not only or "embedding" in only:
            await record("embedding", embedding_call)
        if not only or "rerank" in only:
            await record("rerank", rerank_call)
        if not only or "mineru" in only:
            await record("mineru", mineru_call)

    async def smtp_call() -> dict[str, Any]:
        client = aiosmtplib.SMTP(
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            use_tls=settings.smtp_use_ssl,
            start_tls=settings.smtp_starttls,
            timeout=20,
        )
        async with client:
            if settings.smtp_username:
                await client.login(
                    settings.smtp_username, settings.smtp_password.get_secret_value()
                )
        return {"scope": "tls_and_authentication_only", "email_sent": False}

    async def trace_call() -> dict[str, Any]:
        client = tracing_client(settings)
        if client is None:
            raise ValueError("Tracing disabled")
        try:
            if not await asyncio.to_thread(client.auth_check):
                raise ValueError("Invalid Langfuse credentials")
            with client.start_as_current_observation(
                name="papermind-m0-smoke",
                as_type="span",
                metadata={"milestone": "M0", "contains_user_content": False},
            ) as span:
                trace_id = span.trace_id
                span.update(output={"status": "ok"})
            await asyncio.to_thread(client.flush)
            # Check persistence through HTTP instead of treating an asynchronous flush as proof.
            async with httpx.AsyncClient(
                timeout=20,
                auth=(
                    settings.langfuse_public_key.get_secret_value(),
                    settings.langfuse_secret_key.get_secret_value(),
                ),
            ) as client_http:
                for _ in range(12):
                    response = await client_http.get(
                        f"{settings.langfuse_base_url.rstrip('/')}/api/public/v2/observations",
                        params={
                            "traceId": trace_id,
                            "fields": "core",
                            "limit": 10,
                            "fromStartTime": (datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
                            "toStartTime": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
                        },
                    )
                    if response.status_code == 200 and any(
                        row.get("traceId") == trace_id for row in response.json().get("data", [])
                    ):
                        return {
                            "trace_id": trace_id,
                            "persisted": True,
                            "read_api": "v2/observations",
                        }
                    if response.status_code not in {200, 404, 429}:
                        response.raise_for_status()
                    await asyncio.sleep(3)
                response.raise_for_status()
            raise ValueError("Trace not readable")
        finally:
            await asyncio.to_thread(client.shutdown)

    if not only or "smtp" in only:
        await record("smtp", smtp_call)
    if not only or "langfuse" in only:
        await record("langfuse", trace_call)
    report["passed"] = all(row["status"] == "passed" for row in checks.values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicit live M0 checks; consumes a few model tokens"
    )
    parser.add_argument("--output", type=Path, default=Path("../reports/providers-local.json"))
    parser.add_argument(
        "--only", help="Comma separated: main,fast,embedding,rerank,mineru,smtp,langfuse"
    )
    parser.add_argument("--merge", action="store_true", help="Preserve earlier successful checks")
    args = parser.parse_args()
    report = asyncio.run(run(set(args.only.split(",")) if args.only else None))
    if args.merge and args.output.exists():
        previous = json.loads(args.output.read_text())
        report["checks"] = {**previous["checks"], **report["checks"]}
        report["passed"] = all(row["status"] == "passed" for row in report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
