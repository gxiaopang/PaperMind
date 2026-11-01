import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID, uuid4

import httpx
from fastapi import HTTPException
from langchain_core.prompts import ChatPromptTemplate
from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.services.parsing.metadata import (
    MetadataExtraction,
    MetadataExtractionSkipped,
    metadata_prompt,
)
from app.services.parsing.pipeline import private_json
from app.services.providers.clients import chat_model
from app.services.retrieval.accounting import RetrievalAccounting

ExtractionModel = TypeVar("ExtractionModel", bound=BaseModel)

MAX_OUTPUT_TOKENS = 3000
MAX_ATTEMPTS = 2


class AccountedMetadataExtractor:
    """Optional metadata extraction within an ingestion task's existing provider slot.

    The caller serializes the private parse directory. Persist responses before
    settling usage so a crash can recover both without issuing another request.
    Unknown calls consume an attempt and retain their conservative token charge.
    """

    def __init__(
        self,
        settings: Settings,
        sessions: async_sessionmaker[AsyncSession],
        engine: AsyncEngine,
        user_id: UUID,
        version_id: UUID,
        output: Path,
        validate: Callable[[AsyncSession], Awaitable[None]],
        *,
        cache_name: str = "metadata-extraction.json",
    ):
        self.settings = settings
        self.user_id, self.version_id = user_id, version_id
        self.path = output / cache_name
        self.validate = validate
        self.accounting = RetrievalAccounting(sessions, engine, settings)

    async def __call__(self, text: str) -> MetadataExtraction:
        return await self.extract(text, metadata_prompt(), MetadataExtraction)

    async def extract(
        self,
        text: str,
        prompt: ChatPromptTemplate,
        schema: type[ExtractionModel],
        *,
        node: str = "metadata_extract",
        max_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> ExtractionModel:
        messages = (await prompt.ainvoke({"text": text})).to_messages()
        identity = {
            "user_id": str(self.user_id),
            "version_id": str(self.version_id),
            "provider": self.settings.llm_fast_base_url.rstrip("/"),
            "model": self.settings.llm_fast_model,
            "messages": [m.model_dump(mode="json") for m in messages],
            "max_tokens": max_tokens,
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        state: dict[str, Any] = {"fingerprint": fingerprint, "attempts": []}
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state.get("fingerprint") != fingerprint:
                raise MetadataExtractionSkipped("元数据提取配置或输入已改变，保留已有字段")

        # A received response may have been saved immediately before a crash in
        # SQL settlement. Reconciliation never makes a new provider request.
        for attempt in state["attempts"]:
            if attempt["status"] == "response":
                await self.accounting.settle(
                    self.user_id, UUID(attempt["record_id"]), attempt["usage"]
                )
                try:
                    return schema.model_validate_json(attempt["content"])
                except ValidationError:
                    pass
            elif attempt["status"] == "skipped":
                raise MetadataExtractionSkipped(attempt["warning"])

        # UTF-8 bytes conservatively bound input tokens; include all messages,
        # chat framing and the explicit output ceiling, not only the paper text.
        amount = sum(len(str(m.content).encode()) + 16 for m in messages) + max_tokens
        while len(state["attempts"]) < MAX_ATTEMPTS:
            attempt = {"record_id": str(uuid4()), "status": "started"}
            state["attempts"].append(attempt)
            private_json(self.path, state)
            usage: list[int | None] = [None]

            async def call(
                attempt: dict[str, Any] = attempt, usage: list[int | None] = usage
            ) -> str:
                response = await chat_model(self.settings, "fast", max_tokens=max_tokens).ainvoke(
                    messages, response_format={"type": "json_object"}
                )
                value = response.usage_metadata["total_tokens"] if response.usage_metadata else None
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage[0] = value
                content = response.content if isinstance(response.content, str) else ""
                attempt.update(status="response", content=content, usage=usage[0])
                private_json(self.path, state)
                return content

            def current_usage(usage: list[int | None] = usage) -> int | None:
                return usage[0]

            try:
                content = await self.accounting.call(
                    self.user_id,
                    self.version_id,
                    node,
                    amount,
                    call,
                    current_usage,
                    provider=self.settings.llm_fast_base_url,
                    model=self.settings.llm_fast_model,
                    version_id=self.version_id,
                    record_id=UUID(attempt["record_id"]),
                    validate=self.validate,
                )
            except HTTPException as error:
                warning = (
                    "今日模型 token 额度不足，已跳过快速模型元数据提取"
                    if error.status_code == 429
                    else "账户或论文状态不可用，已跳过快速模型元数据提取"
                )
                attempt.update(status="skipped", warning=warning)
                private_json(self.path, state)
                raise MetadataExtractionSkipped(warning) from None
            except (APIStatusError, httpx.HTTPStatusError) as error:
                status = (
                    error.status_code
                    if isinstance(error, APIStatusError)
                    else error.response.status_code
                )
                if status < 500 and status != 429:
                    warning = "快速模型拒绝元数据提取请求，保留已有字段"
                    attempt.update(status="skipped", warning=warning)
                    private_json(self.path, state)
                    raise MetadataExtractionSkipped(warning) from None
            except (APIConnectionError, httpx.TransportError):
                pass
            else:
                try:
                    return schema.model_validate_json(content)
                except ValidationError:
                    # A malformed response was still a paid call. Retry through
                    # the same accounting path, subject to the remaining budget.
                    pass
            if len(state["attempts"]) < MAX_ATTEMPTS:
                await asyncio.sleep(1)
        raise MetadataExtractionSkipped("快速模型元数据提取次数已耗尽，保留已有字段")
