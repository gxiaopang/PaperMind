import io
import stat
import zipfile

import httpx
import pytest
import respx

from app.services.parsing.archive import download_archive, extract_archive
from app.services.parsing.errors import ParsingError
from tests.test_mineru_client import ZIP, settings


def zip_bytes(entries):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in entries:
            archive.writestr(name, body)
    return target.getvalue()


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/etc/passwd",
        "nested/../../escape",
        "nested\\escape",
        "C:/escape",
        ".hidden/escape",
    ],
)
def test_zip_path_traversal_is_rejected_before_any_write(tmp_path, name):
    source = tmp_path / "result.zip"
    source.write_bytes(zip_bytes([("valid.txt", "not published"), (name, "unsafe")]))
    destination = tmp_path / "raw"
    with pytest.raises(ParsingError, match="路径"):
        extract_archive(source, destination, settings())
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600, stat.S_IFCHR | 0o600])
def test_zip_special_files_are_rejected(tmp_path, mode):
    source = tmp_path / "result.zip"
    with zipfile.ZipFile(source, "w") as archive:
        entry = zipfile.ZipInfo("link")
        entry.create_system = 3
        entry.external_attr = mode << 16
        archive.writestr(entry, "/etc/passwd")
    with pytest.raises(ParsingError):
        extract_archive(source, tmp_path / "raw", settings())
    assert not (tmp_path / "raw").exists()


def test_zip_entry_count_expansion_limit_duplicate_and_corruption(tmp_path):
    source = tmp_path / "result.zip"
    destination = tmp_path / "raw"
    source.write_bytes(zip_bytes([("a", "x"), ("b", "x")]))
    with pytest.raises(ParsingError):
        extract_archive(source, destination, settings(mineru_max_archive_entries=1))
    source.write_bytes(zip_bytes([("a", b"x" * (2 * 1024**2))]))
    with pytest.raises(ParsingError):
        extract_archive(source, destination, settings(mineru_max_expanded_mb=1))
    with pytest.warns(UserWarning):
        source.write_bytes(zip_bytes([("a", "x"), ("a", "y")]))
    with pytest.raises(ParsingError):
        extract_archive(source, destination, settings())
    source.write_bytes(b"not a zip")
    with pytest.raises(ParsingError):
        extract_archive(source, destination, settings())
    assert not destination.exists()


def test_zip_valid_nested_files_are_private_and_cannot_overwrite(tmp_path):
    source = tmp_path / "result.zip"
    source.write_bytes(
        zip_bytes([("nested/content_list.json", "[]"), ("nested/images/a.png", b"asset")])
    )
    destination = tmp_path / "raw"
    extract_archive(source, destination, settings())
    assert (destination / "nested/images/a.png").read_bytes() == b"asset"
    assert (destination / "nested/content_list.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ParsingError):
        extract_archive(source, destination, settings())


@respx.mock
async def test_download_stream_limits_and_refuses_redirects(tmp_path):
    route = respx.get(ZIP).respond(302, headers={"Location": "http://127.0.0.1/private"})
    async with httpx.AsyncClient() as http:
        with pytest.raises(ParsingError):
            await download_archive(http, ZIP, tmp_path / "result.zip", settings())
        route.respond(200, headers={"Content-Length": str(2 * 1024**2)})
        with pytest.raises(ParsingError):
            await download_archive(
                http, ZIP, tmp_path / "result.zip", settings(mineru_max_archive_mb=1)
            )
        route.respond(200, content=b"x" * (2 * 1024**2), headers={"Content-Length": "0"})
        with pytest.raises(ParsingError):
            await download_archive(
                http, ZIP, tmp_path / "result.zip", settings(mineru_max_archive_mb=1)
            )
    assert not list(tmp_path.iterdir())


@respx.mock
async def test_download_retry_writes_complete_file_without_auth(tmp_path, monkeypatch):
    async def sleep(_):
        pass

    monkeypatch.setattr("app.services.parsing.archive.asyncio.sleep", sleep)
    route = respx.get(ZIP)
    route.side_effect = [httpx.Response(503), httpx.Response(200, content=b"complete")]
    async with httpx.AsyncClient() as http:
        await download_archive(http, ZIP, tmp_path / "result.zip", settings())
    assert (tmp_path / "result.zip").read_bytes() == b"complete"
    assert "authorization" not in route.calls[-1].request.headers
    assert not (tmp_path / "result.part").exists()
