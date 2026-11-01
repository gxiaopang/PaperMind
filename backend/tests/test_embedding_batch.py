import json

import httpx
import pytest
import respx

from app.core.config import Settings
from app.services.parsing.errors import ParsingError
from app.services.providers.embedding_batch import AccountedEmbeddings


@pytest.fixture
async def client():
    settings = Settings(_env_file=None, tracing_enabled=False, embedding_api_key="test")
    instance = AccountedEmbeddings(settings)
    try:
        yield instance
    finally:
        await instance.aclose()


@respx.mock
async def test_embeddings_keep_string_inputs_sort_results_and_report_actual_usage(client):
    route = respx.post("https://api.siliconflow.cn/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "model": "BAAI/bge-m3",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.2] * 1024},
                    {"object": "embedding", "index": 0, "embedding": [0.1] * 1024},
                ],
                "usage": {"prompt_tokens": 20, "total_tokens": 20},
            },
        )
    )
    values = await client.aembed_documents(["Title\nMethods\n\n正文", "Title\nResults\n\ntext"])
    assert values[0][0] == 0.1 and values[1][0] == 0.2
    assert client.last_usage == 20
    assert json.loads(route.calls[0].request.content)["input"] == [
        "Title\nMethods\n\n正文",
        "Title\nResults\n\ntext",
    ]
    assert client.client.check_embedding_ctx_length is False


@respx.mock
async def test_embeddings_without_usage_are_explicitly_estimated(client):
    respx.post("https://api.siliconflow.cn/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 1024}],
            },
        )
    )
    assert len((await client.aembed_documents(["text"]))[0]) == 1024
    assert client.last_usage is None


@respx.mock
async def test_embeddings_reject_duplicate_indices(client):
    respx.post("https://api.siliconflow.cn/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}] * 2,
            },
        )
    )
    with pytest.raises(ParsingError, match="索引"):
        await client.aembed_documents(["a", "b"])
