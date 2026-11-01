import httpx

from app.core.config import Settings
from app.services.parsing.errors import ParsingError
from app.services.providers.clients import embeddings


class AccountedEmbeddings:
    """Use the configured OpenAIEmbeddings transport while retaining vendor usage.

    Chunking owns input budgets; the compatible API receives strings unchanged.
    The SDK response is retained because aembed_documents returns only vectors.
    """

    def __init__(self, settings: Settings):
        self.http = httpx.AsyncClient()
        self.client = embeddings(settings, http_async_client=self.http)
        self.last_usage: int | None = None

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.last_usage = None
        response = await self.client.async_client.create(
            input=texts, model=self.client.model, encoding_format="float"
        )
        result = response if isinstance(response, dict) else response.model_dump()
        data = result.get("data", [])
        if len(data) != len(texts) or sorted(row.get("index", -1) for row in data) != list(
            range(len(texts))
        ):
            raise ParsingError("Embedding 返回条目索引无效", code="invalid_embedding")
        amount = (result.get("usage") or {}).get("total_tokens")
        if isinstance(amount, int) and not isinstance(amount, bool) and amount >= 0:
            self.last_usage = amount
        return [row["embedding"] for row in sorted(data, key=lambda row: row["index"])]

    async def aclose(self) -> None:
        await self.http.aclose()
