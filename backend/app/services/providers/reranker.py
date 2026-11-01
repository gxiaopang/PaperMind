import math
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict, Field

from app.core.config import Settings


class RerankCompressor(BaseDocumentCompressor):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    settings: Settings = Field(exclude=True)
    last_usage: int | None = None

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks = None
    ) -> Sequence[Document]:
        raise NotImplementedError("Use acompress_documents")

    async def acompress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks = None
    ) -> Sequence[Document]:
        self.last_usage = None
        if not documents:
            return []
        async with httpx.AsyncClient(timeout=self.settings.rerank_timeout_s) as client:
            response = await client.post(
                f"{self.settings.rerank_base_url.rstrip('/')}/rerank",
                headers={
                    "Authorization": f"Bearer {self.settings.rerank_api_key.get_secret_value()}"
                },
                json={
                    "model": self.settings.rerank_model,
                    "query": query,
                    "documents": [
                        d.metadata.get("embedding_input", d.page_content) for d in documents
                    ],
                    "top_n": min(self.settings.rerank_top_n, len(documents)),
                    "return_documents": False,
                },
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        usage = (data.get("usage") or {}).get("total_tokens")
        if type(usage) is int and usage >= 0:
            self.last_usage = usage
        else:
            # SiliconFlow reports classic rerank usage under meta.tokens rather
            # than the OpenAI-style usage.total_tokens used by some gateways.
            meta = data.get("meta") or {}
            tokens = meta.get("tokens") or meta.get("billed_units") or {}
            input_tokens, output_tokens = tokens.get("input_tokens"), tokens.get("output_tokens", 0)
            if (
                type(input_tokens) is int
                and input_tokens >= 0
                and type(output_tokens) is int
                and output_tokens >= 0
            ):
                self.last_usage = input_tokens + output_tokens
        rows = data.get("results")
        expected = min(self.settings.rerank_top_n, len(documents))
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError("Invalid rerank count")
        result, seen = [], set()
        for row in rows:
            index, score = row.get("index"), row.get("relevance_score")
            if (
                type(index) is not int
                or not 0 <= index < len(documents)
                or index in seen
                or type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("Invalid rerank index or score")
            seen.add(index)
            doc = documents[index].model_copy(deep=True)
            doc.metadata["rerank_score"] = float(score)
            result.append(doc)
        return sorted(result, key=lambda d: -d.metadata["rerank_score"])
