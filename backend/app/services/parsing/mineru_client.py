import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import Settings
from app.services.parsing.errors import ParsingError, SubmissionUncertain


class UploadTicket(BaseModel):
    batch_id: str
    upload_url: str = Field(repr=False)
    data_id: str


class RemoteResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file_name: str = ""
    data_id: str | None = None
    state: Literal["waiting-file", "pending", "running", "converting", "done", "failed"]
    full_zip_url: str | None = Field(default=None, repr=False)
    err_msg: str = Field(default="", repr=False)
    extract_progress: dict[str, Any] | None = None


Progress = Callable[[RemoteResult], Awaitable[None]]


def check_remote_url(url: str, allowed_hosts: tuple[str, ...]) -> None:
    try:
        parts = urlsplit(url)
        valid = (
            parts.scheme == "https"
            and parts.hostname in allowed_hosts
            and parts.port in {None, 443}
            and not parts.username
            and not parts.password
            and not parts.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ParsingError("MinerU 文件地址不在 HTTPS 主机白名单中", code="unsafe_url")


class MinerUClient:
    """Cloud v4 batch upload API. Auth is only sent to the configured API origin.

    The caller must persist the ticket BEFORE upload. Retrying a known PUT/GET is
    safe; an ambiguous POST is never automatically repeated.
    """

    def __init__(self, settings: Settings, http: httpx.AsyncClient):
        self.settings = settings
        self.http = http
        self.base = settings.mineru_api_base.rstrip("/")
        parsed = urlsplit(self.base)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ParsingError("MINERU_API_BASE 必须是 HTTPS API 地址", code="configuration")
        self.token = settings.mineru_api_token.get_secret_value()
        if not self.token:
            raise ParsingError("未配置 MinerU Token", code="configuration")

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self.settings.mineru_http_attempts):
            try:
                response = await self.http.request(
                    method,
                    url,
                    follow_redirects=False,
                    timeout=self.settings.mineru_request_timeout_s,
                    **kwargs,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                # DNS/TCP/TLS and pool failures happen before an HTTP request
                # can be sent, so even a POST is safe to retry. Read/write
                # failures below may have reached the server and stay uncertain.
                if attempt + 1 == self.settings.mineru_http_attempts:
                    raise ParsingError(
                        "无法连接 MinerU，请检查网络或代理后重试", code="network", retryable=True
                    ) from None
            except httpx.TransportError:
                if method == "POST":
                    raise SubmissionUncertain() from None
                if attempt + 1 == self.settings.mineru_http_attempts:
                    raise ParsingError(
                        "MinerU 网络连接失败", code="network", retryable=True
                    ) from None
            else:
                if response.status_code in {401, 403}:
                    raise ParsingError("MinerU Token 无效、过期或权限不足", code="authentication")
                if response.status_code == 402:
                    raise ParsingError("MinerU 服务额度不足", code="quota_exhausted")
                if method == "POST" and response.status_code >= 500:
                    raise SubmissionUncertain()
                if response.status_code != 429 and response.status_code < 500:
                    if not 200 <= response.status_code < 300:
                        raise ParsingError(
                            "MinerU HTTP 请求被拒绝", code=f"http_{response.status_code}"
                        )
                    return response
                if attempt + 1 == self.settings.mineru_http_attempts:
                    raise ParsingError(
                        "MinerU 服务繁忙或限流", code="rate_or_service", retryable=True
                    )
            await asyncio.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")

    async def _api(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._request(
            method,
            f"{self.base}{endpoint}",
            headers={"Authorization": f"Bearer {self.token}"},
            **kwargs,
        )
        try:
            body = response.json()
            code = str(body["code"])
            if code != "0":
                messages = {
                    "A0202": "MinerU Token 无效",
                    "A0211": "MinerU Token 已过期",
                    "-60018": "MinerU 每日解析额度已用尽",
                    "-60019": "MinerU 额度已用尽",
                    "-60012": "MinerU 任务不存在，请核对任务记录",
                    "-500": "MinerU 请求参数错误",
                    "-10002": "MinerU 请求参数格式错误",
                    "-60002": "MinerU 无法识别文件格式",
                    "-60003": "MinerU 无法读取文件，请检查 PDF 是否损坏",
                    "-60004": "MinerU 收到空文件",
                    "-60005": "PDF 超过 MinerU 文件大小限制",
                    "-60006": "PDF 超过 MinerU 页数限制",
                    "-60007": "MinerU 模型服务暂时不可用",
                    "-60009": "MinerU 任务队列已满",
                    "-60013": "无权访问该 MinerU 任务",
                }
                raise ParsingError(
                    messages.get(code, f"MinerU 业务请求失败（{code}）"),
                    code=code,
                    retryable=code in {"-10001", "-60007", "-60009"},
                )
            data = body["data"]
            if not isinstance(data, dict):
                raise ValueError
            return data
        except (ValueError, KeyError, TypeError):
            if method == "POST":
                raise SubmissionUncertain() from None
            raise ParsingError("MinerU 响应格式无效", code="invalid_response") from None

    async def request_upload(self, filename: str, data_id: str) -> UploadTicket:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", data_id):
            raise ParsingError("data_id 格式无效", code="invalid_input")
        if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            raise ParsingError("上传文件名必须是 PDF basename", code="invalid_input")
        data = await self._api(
            "POST",
            "/file-urls/batch",
            json={
                "files": [{"name": filename, "data_id": data_id}],
                "model_version": self.settings.mineru_model_version,
                "enable_formula": self.settings.mineru_enable_formula,
                "enable_table": self.settings.mineru_enable_table,
                "language": self.settings.mineru_language,
            },
        )
        try:
            ticket = UploadTicket(
                batch_id=data["batch_id"], upload_url=data["file_urls"][0], data_id=data_id
            )
            return ticket
        except (KeyError, IndexError, TypeError, ValidationError):
            raise SubmissionUncertain() from None

    async def upload(self, ticket: UploadTicket, path: Path) -> None:
        check_remote_url(ticket.upload_url, self.settings.mineru_upload_hosts)
        size = (await asyncio.to_thread(path.stat)).st_size
        # Reopen the stream on every retry; never attach the API bearer token.
        for attempt in range(self.settings.mineru_http_attempts):

            async def chunks() -> AsyncIterator[bytes]:
                with path.open("rb") as source:
                    while block := await asyncio.to_thread(source.read, 1024 * 1024):
                        yield block

            try:
                response = await self.http.put(
                    ticket.upload_url,
                    content=chunks(),
                    headers={"Content-Length": str(size)},
                    timeout=self.settings.mineru_request_timeout_s,
                    follow_redirects=False,
                )
                if response.status_code in {200, 201, 204}:
                    return
                if response.status_code < 500 and response.status_code != 429:
                    raise ParsingError("MinerU 上传失败或上传地址已过期", code="upload_rejected")
            except httpx.TransportError:
                pass
            if attempt + 1 < self.settings.mineru_http_attempts:
                await asyncio.sleep(min(2**attempt, 8))
        raise ParsingError("MinerU 上传网络失败", code="upload_network", retryable=True)

    async def result(self, batch_id: str, data_id: str) -> RemoteResult:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", batch_id):
            raise ParsingError("batch_id 格式无效", code="invalid_input")
        data = await self._api("GET", f"/extract-results/batch/{batch_id}")
        try:
            rows = data["extract_result"]
            if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
                raise ValueError
            matches = [r for r in rows if r.get("data_id") == data_id]
            if len(matches) != 1:
                raise ValueError
            result = RemoteResult.model_validate(matches[0])
            if result.state == "done" and not result.full_zip_url:
                raise ValueError
            return result
        except (KeyError, TypeError, ValueError):
            raise ParsingError("MinerU 结果缺失或任务标识不匹配", code="invalid_response") from None

    async def poll(
        self, batch_id: str, data_id: str, progress: Progress | None = None
    ) -> RemoteResult:
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.settings.mineru_poll_timeout_s):
                while True:
                    result = await self.result(batch_id, data_id)
                    if progress:
                        await progress(result)
                    if result.state == "failed":
                        raise ParsingError(
                            "MinerU 远端解析失败，可核对原始任务记录", code="remote_failed"
                        )
                    if result.state == "done":
                        return result
                    await asyncio.sleep(self.settings.mineru_poll_interval_s)
        except TimeoutError:
            raise ParsingError(
                f"MinerU 轮询超时（{int(time.monotonic() - started)} 秒），保留批次可恢复",
                code="poll_timeout",
                retryable=True,
            ) from None
