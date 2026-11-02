import asyncio
import json
import logging
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

import structlog
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import (
    MaskOtelSpansFunction,
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanPatch,
    TraceContext,
)
from opentelemetry import trace

from app.core.config import Settings

log = structlog.get_logger()
CONTENT_DISABLED = "[content disabled]"
MAX_CONTENT_CHARS = 8192
# Only server-produced diagnostic metadata is exported. Graph state, documents,
# tool arguments and memory must never become metadata by accident.
SAFE_METADATA = frozenset(
    {
        "run_id",
        "prompt_version",
        "mode",
        "task_kind",
        "attempt",
        "resumed",
        "operation_id",
        "variant",
        "intent",
        "query_count",
        "scope_count",
        "candidate_count",
        "document_count",
        "context_bytes",
        "channel_counts",
        "failed_count",
        "degraded",
        "tool",
        "status",
        "error_type",
        "http_status",
        "retrieval_count",
        "generation_count",
        "retrieval_retries",
        "generation_retries",
        "recoveries",
        "result_status",
        "citation_validation",
        "grounding",
        "sources_revoked",
        "citation_count",
        "tokens",
        "tokens_estimated",
        "safe_fallback",
        "duration_seconds",
        "milestone",
        "contains_user_content",
    }
)
_SECRET_KEY = re.compile(
    r"(?:password|secret|api[_-]?key|authorization|cookie|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(r"\bBearer\s+[^\s\"']+|\b(?:sk|pk)-[A-Za-z0-9_-]{12,}", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]" if _SECRET_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _EMAIL.sub("[email]", _SECRET_VALUE.sub("[redacted]", value))
    return value


def export_mask(*, content_enabled: bool) -> MaskOtelSpansFunction:
    """Filter final attributes, including metadata propagated by callbacks."""

    def mask(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
        patches = {}
        for identifier, span in params.spans.items():
            deleted = []
            replaced: dict[str, Any] = {}
            for key, value in span.attributes.items():
                if key.startswith(("langfuse.observation.metadata", "langfuse.trace.metadata")):
                    field = key.split(".metadata.", 1)[-1]
                    if field not in SAFE_METADATA:
                        deleted.append(key)
                elif key in {
                    "langfuse.observation.status_message",
                    "exception.message",
                    "exception.stacktrace",
                }:
                    replaced[key] = "[redacted]"
                elif key in {
                    "langfuse.observation.input",
                    "langfuse.observation.output",
                    "langfuse.trace.input",
                    "langfuse.trace.output",
                    "gen_ai.system_instructions",
                    "gen_ai.tool.call.arguments",
                    "gen_ai.tool.call.result",
                } or key.startswith(
                    ("gen_ai.prompt.", "gen_ai.completion.", "gen_ai.input.", "gen_ai.output.")
                ):
                    if not content_enabled:
                        replaced[key] = CONTENT_DISABLED
                    elif isinstance(value, str):
                        try:
                            decoded = json.loads(value)
                        except (ValueError, TypeError):
                            decoded = value
                        replaced[key] = json.dumps(_redact(decoded), ensure_ascii=False)[
                            :MAX_CONTENT_CHARS
                        ]
            if deleted or replaced:
                patches[identifier] = OtelSpanPatch(
                    delete_attributes=tuple(deleted), set_attributes=replaced
                )
        return MaskOtelSpansResult(span_patches=patches)

    return mask


def run_trace_id(user_id: UUID | str, run_id: UUID | str, kind: str = "chat") -> str:
    return Langfuse.create_trace_id(seed=f"papermind:{kind}:{user_id}:{run_id}")


_client: ContextVar[Any] = ContextVar("papermind_tracing_client", default=None)


class SafeCallbackHandler(CallbackHandler):
    # Keep OTEL context changes in the invoking task, rather than an executor.
    run_inline = True

    def _get_error_level_and_status_message(
        self, error: BaseException
    ) -> tuple[Literal["DEFAULT", "ERROR"], str]:
        # SDK also copies this message into OTEL Status.description, which the
        # export masking hook cannot change. Preserve interrupt classification
        # while keeping provider response bodies out of both representations.
        level, _ = super()._get_error_level_and_status_message(error)
        return level, type(error).__name__


class Observation:
    """Best-effort telemetry: SDK failures must not fail a business operation."""

    def __init__(self, span: Any = None):
        self.span = span

    def update(self, **metadata: Any) -> None:
        if self.span is not None:
            try:
                kwargs: dict[str, Any] = {"metadata": metadata}
                if metadata.get("status") in {"error", "failed"}:
                    kwargs.update(
                        level="ERROR", status_message=metadata.get("error_type", "failed")
                    )
                self.span.update(**kwargs)
            except Exception as exc:
                log.warning("tracing_update_failed", error_type=type(exc).__name__)


@contextmanager
def observation(name: str, **metadata: Any) -> Iterator[Observation]:
    client = _client.get()
    stack = ExitStack()
    span = None
    if client is not None:
        try:
            span = stack.enter_context(
                client.start_as_current_observation(name=name, as_type="span", metadata=metadata)
            )
        except Exception as exc:
            log.warning("tracing_start_failed", error_type=type(exc).__name__)
    result = Observation(span)
    try:
        yield result
    except BaseException as exc:
        result.update(
            status="interrupted" if isinstance(exc, asyncio.CancelledError) else "error",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        try:
            # Do not forward business exceptions to SDK context managers: their
            # automatic exception events could include private provider bodies.
            stack.close()
        except Exception as exc:
            log.warning("tracing_end_failed", error_type=type(exc).__name__)


@contextmanager
def run_trace(
    client: Any, run: Any, *, kind: str = "chat", attempt: int = 1
) -> Iterator[Observation]:
    token = _client.set(client)
    stack = ExitStack()
    root = None
    metadata = {
        "run_id": str(run.id),
        "mode": run.mode,
        "task_kind": kind,
        "attempt": attempt,
        "prompt_version": run.config.get("prompt_version"),
    }
    try:
        stack.enter_context(
            structlog.contextvars.bound_contextvars(
                run_id=str(run.id), trace_id=run_trace_id(run.user_id, run.id, kind)
            )
        )
        if client is not None:
            try:
                root = stack.enter_context(
                    client.start_as_current_observation(
                        name=f"papermind.{kind}.attempt",
                        as_type="span",
                        trace_context={"trace_id": run_trace_id(run.user_id, run.id, kind)},
                        metadata=metadata,
                    )
                )
                stack.enter_context(
                    propagate_attributes(
                        user_id=str(run.user_id),
                        session_id=str(run.conversation_id),
                        trace_name=f"papermind.{kind}",
                        metadata=metadata,
                    )
                )
            except Exception as exc:
                log.warning("tracing_start_failed", error_type=type(exc).__name__)
        result = Observation(root)
        try:
            yield result
        except BaseException as exc:
            result.update(
                status="interrupted" if isinstance(exc, asyncio.CancelledError) else "error",
                error_type=type(exc).__name__,
            )
            raise
    finally:
        try:
            stack.close()
        except Exception as exc:
            log.warning("tracing_end_failed", error_type=type(exc).__name__)
        _client.reset(token)


def trace_context(run: Any, kind: str = "chat") -> TraceContext:
    tid = run_trace_id(run.user_id, run.id, kind)
    current = trace.get_current_span().get_span_context()
    context: TraceContext = {"trace_id": tid}
    if current.is_valid and f"{current.trace_id:032x}" == tid:
        context["parent_span_id"] = f"{current.span_id:016x}"
    return context


def publish_result(client: Any, run: Any, metrics: dict[str, Any], root: Observation) -> None:
    root.update(**metrics)
    if client is None:
        return
    tid = run_trace_id(run.user_id, run.id)
    for name in (
        "result_status",
        "citation_validation",
        "grounding",
        "degraded",
        "retrieval_retries",
        "generation_retries",
    ):
        value = metrics[name]
        try:
            client.create_score(
                trace_id=tid,
                name=name,
                value=int(value) if isinstance(value, bool) else value,
                data_type="BOOLEAN"
                if isinstance(value, bool)
                else "CATEGORICAL"
                if isinstance(value, str)
                else "NUMERIC",
                score_id=str(uuid5(NAMESPACE_URL, f"papermind:{tid}:{name}")),
            )
        except Exception as exc:
            log.warning("tracing_score_failed", error_type=type(exc).__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # HTTP client debug logs can contain account tokens or provider response bodies.
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "aiosmtplib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def tracing_client(settings: Settings) -> Langfuse | None:
    if not settings.tracing_enabled:
        return None

    try:
        return Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            mask_otel_spans=export_mask(content_enabled=settings.trace_content_enabled),
            environment=settings.app_env,
        )
    except Exception as exc:
        log.warning("tracing_initialization_failed", error_type=type(exc).__name__)
        return None
