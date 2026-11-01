from typing import Literal

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings


def chat_model(
    settings: Settings, kind: Literal["main", "fast"] = "main", *, max_tokens: int = 2048
) -> ChatOpenAI:
    main = kind == "main"
    return ChatOpenAI(
        model=settings.llm_model if main else settings.llm_fast_model,
        api_key=settings.llm_api_key if main else settings.llm_fast_api_key,
        base_url=settings.llm_base_url if main else settings.llm_fast_base_url,
        temperature=0,
        timeout=settings.llm_timeout_s if main else settings.llm_fast_timeout_s,
        max_retries=0,
        # These compatible providers document max_tokens, whereas ChatOpenAI rewrites
        # its max_tokens field to OpenAI's max_completion_tokens. Extra body keeps
        # the provider's documented wire contract.
        extra_body={
            "max_tokens": max_tokens,
            **({"thinking": {"type": "disabled"}} if main else {"enable_thinking": False}),
        },
    )


def embeddings(
    settings: Settings, *, http_async_client: httpx.AsyncClient | None = None
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.embedding_api_key,
        openai_api_base=settings.embedding_base_url,
        check_embedding_ctx_length=False,
        http_async_client=http_async_client,
        max_retries=0,
        request_timeout=60,
    )
