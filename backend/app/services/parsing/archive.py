import asyncio
import json
import shutil
import stat
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from app.core.config import Settings
from app.services.parsing.errors import ParsingError
from app.services.parsing.mineru_client import check_remote_url


async def download_archive(
    http: httpx.AsyncClient, url: str, destination: Path, settings: Settings
) -> None:
    check_remote_url(url, settings.mineru_download_hosts)
    limit = settings.mineru_max_archive_mb * 1024**2
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for attempt in range(settings.mineru_http_attempts):
        temporary = destination.with_suffix(".part")
        try:
            async with http.stream(
                "GET", url, follow_redirects=False, timeout=settings.mineru_request_timeout_s
            ) as response:
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.ReadError("retryable download")
                if response.status_code != 200:
                    raise ParsingError(
                        "MinerU 结果包下载被拒绝或地址已过期", code="download_rejected"
                    )
                length = response.headers.get("content-length")
                if length:
                    try:
                        declared_size = int(length)
                    except ValueError:
                        raise ParsingError(
                            "结果下载长度格式无效", code="invalid_response"
                        ) from None
                    if declared_size < 0 or declared_size > limit:
                        raise ParsingError("MinerU 结果包超过下载限制", code="archive_limit")
                size = 0
                with temporary.open("wb") as target:
                    temporary.chmod(0o600)
                    async for block in response.aiter_bytes():
                        size += len(block)
                        if size > limit:
                            raise ParsingError("MinerU 结果包超过下载限制", code="archive_limit")
                        target.write(block)
                temporary.replace(destination)
                return
        except httpx.TransportError:
            if attempt + 1 == settings.mineru_http_attempts:
                raise ParsingError(
                    "MinerU 结果下载网络失败", code="download_network", retryable=True
                ) from None
        finally:
            temporary.unlink(missing_ok=True)
        await asyncio.sleep(min(2**attempt, 8))


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or ":" in name
        or path.is_absolute()
        or ".." in path.parts
        or any(part.startswith(".") for part in path.parts)
    ):
        raise ParsingError("结果包包含不安全路径", code="unsafe_archive")
    return path


def extract_archive(archive: Path, destination: Path, settings: Settings) -> None:
    """Validate every member before writing; publish a complete fresh directory."""
    if destination.exists():
        raise ParsingError("解压目录已存在", code="destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if len(entries) > settings.mineru_max_archive_entries:
                raise ParsingError("结果包条目过多", code="archive_limit")
            expanded = 0
            names: set[str] = set()
            for entry in entries:
                path = safe_member(entry.filename)
                name = str(path)
                mode = entry.external_attr >> 16
                if name in names or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                    raise ParsingError("结果包存在重复路径或特殊文件", code="unsafe_archive")
                names.add(name)
                expanded += entry.file_size
                if (
                    entry.flag_bits & 1
                    or expanded > settings.mineru_max_expanded_mb * 1024**2
                    or entry.file_size > max(1024**2, entry.compress_size * 1000)
                ):
                    raise ParsingError("结果包加密或展开大小超过限制", code="archive_limit")
            actual = 0
            for entry in entries:
                target = temporary / str(safe_member(entry.filename))
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with source.open(entry) as content, target.open("xb") as output:
                    target.chmod(0o600)
                    while block := content.read(1024 * 1024):
                        actual += len(block)
                        if actual > settings.mineru_max_expanded_mb * 1024**2:
                            raise ParsingError("结果包展开大小超过限制", code="archive_limit")
                        output.write(block)
        temporary.rename(destination)
    except ParsingError:
        raise
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        RuntimeError,
        NotImplementedError,
        zlib.error,
        EOFError,
    ):
        raise ParsingError("MinerU 结果包损坏或格式不受支持", code="invalid_archive") from None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_json(path: Path) -> Any:
    if path.stat().st_size > 32 * 1024**2:
        raise ParsingError("解析 JSON 超过大小限制", code="result_limit")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise ParsingError("解析 JSON 格式无效", code="invalid_result") from None
