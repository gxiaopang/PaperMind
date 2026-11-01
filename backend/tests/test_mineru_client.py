import asyncio
import json

import httpx
import pytest
import respx

from app.core.config import Settings
from app.services.parsing.errors import ParsingError, SubmissionUncertain
from app.services.parsing.mineru_client import MinerUClient, check_remote_url

BASE = "https://mineru.net/api/v4"
UPLOAD = "https://mineru.oss-cn-shanghai.aliyuncs.com/api-upload/test?signature=private"
ZIP = "https://cdn-mineru.openxlab.org.cn/pdf/result.zip?signature=private"


def settings(**kwargs):
    return Settings(
        _env_file=None, tracing_enabled=False, mineru_api_token="secret-token", **kwargs
    )


def result(state, **kwargs):
    return {
        "code": 0,
        "data": {
            "batch_id": "batch-1",
            "extract_result": [
                {"data_id": "paper-1", "file_name": "paper.pdf", "state": state, **kwargs}
            ],
        },
    }


@respx.mock
async def test_upload_contract_auth_scope_and_all_poll_states(tmp_path):
    applied = respx.post(f"{BASE}/file-urls/batch").respond(
        json={"code": 0, "data": {"batch_id": "batch-1", "file_urls": [UPLOAD]}}
    )
    upload = respx.put(UPLOAD).respond(200)
    poll = respx.get(f"{BASE}/extract-results/batch/batch-1")
    poll.side_effect = [
        httpx.Response(200, json=result(s, **({"full_zip_url": ZIP} if s == "done" else {})))
        for s in ["waiting-file", "pending", "running", "converting", "done"]
    ]
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-test")
    async with httpx.AsyncClient() as http:
        client = MinerUClient(settings(mineru_poll_interval_s=0.001), http)
        ticket = await client.request_upload(path.name, "paper-1")
        assert "signature" not in repr(ticket)
        await client.upload(ticket, path)
        states = []

        async def progress(value):
            states.append(value.state)

        parsed = await client.poll(ticket.batch_id, "paper-1", progress)
    assert parsed.full_zip_url == ZIP
    assert states == ["waiting-file", "pending", "running", "converting", "done"]
    assert applied.calls[0].request.headers["authorization"] == "Bearer secret-token"
    assert json.loads(applied.calls[0].request.content) == {
        "files": [{"name": "paper.pdf", "data_id": "paper-1"}],
        "model_version": "vlm",
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
    }
    assert "authorization" not in upload.calls[0].request.headers
    assert "content-type" not in upload.calls[0].request.headers
    assert upload.calls[0].request.content == b"%PDF-test"


@pytest.mark.parametrize("code", ["A0202", "A0211", -500, -60018, -60019])
@respx.mock
async def test_permanent_business_errors_not_retried(code):
    route = respx.post(f"{BASE}/file-urls/batch").respond(json={"code": code, "msg": "private"})
    async with httpx.AsyncClient() as http:
        with pytest.raises(ParsingError) as exc:
            await MinerUClient(settings(), http).request_upload("paper.pdf", "paper-1")
    assert exc.value.code == str(code)
    assert not exc.value.retryable and "private" not in str(exc.value)
    assert route.call_count == 1


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadTimeout("private signed URL"),
        httpx.ReadError("private signed URL"),
        httpx.WriteTimeout("private signed URL"),
        httpx.WriteError("private signed URL"),
        httpx.Response(503),
        httpx.Response(200, text="bad json"),
    ],
)
@respx.mock
async def test_ambiguous_submission_is_never_repeated(failure):
    route = respx.post(f"{BASE}/file-urls/batch")
    route.side_effect = failure
    async with httpx.AsyncClient() as http:
        with pytest.raises(SubmissionUncertain) as exc:
            await MinerUClient(settings(), http).request_upload("paper.pdf", "paper-1")
    assert route.call_count == 1
    assert "private" not in str(exc.value)


@pytest.mark.parametrize(
    "error_type", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout]
)
@respx.mock
async def test_post_connection_failure_retries_without_uncertain_submission(
    error_type, monkeypatch
):
    async def sleep(_):
        pass

    monkeypatch.setattr("app.services.parsing.mineru_client.asyncio.sleep", sleep)
    route = respx.post(f"{BASE}/file-urls/batch")
    route.side_effect = error_type("private connection details")
    async with httpx.AsyncClient() as http:
        with pytest.raises(ParsingError) as exc:
            await MinerUClient(settings(), http).request_upload("paper.pdf", "paper-1")
    assert exc.value.code == "network" and exc.value.retryable
    assert "private" not in str(exc.value)
    assert route.call_count == 3


@respx.mock
async def test_post_recovers_after_connection_failure(monkeypatch):
    async def sleep(_):
        pass

    monkeypatch.setattr("app.services.parsing.mineru_client.asyncio.sleep", sleep)
    route = respx.post(f"{BASE}/file-urls/batch")
    route.side_effect = [
        httpx.ConnectError("TLS handshake failed"),
        httpx.Response(
            200, json={"code": 0, "data": {"batch_id": "batch-1", "file_urls": [UPLOAD]}}
        ),
    ]
    async with httpx.AsyncClient() as http:
        ticket = await MinerUClient(settings(), http).request_upload("paper.pdf", "paper-1")
    assert ticket.batch_id == "batch-1" and route.call_count == 2


@respx.mock
async def test_safe_reads_retry_bounded_and_resume_without_post(monkeypatch):
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("app.services.parsing.mineru_client.asyncio.sleep", sleep)
    route = respx.get(f"{BASE}/extract-results/batch/batch-1")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(502),
        httpx.Response(200, json=result("done", full_zip_url=ZIP)),
    ]
    async with httpx.AsyncClient() as http:
        parsed = await MinerUClient(settings(), http).poll("batch-1", "paper-1")
    assert parsed.state == "done" and route.call_count == 3
    assert sleeps == [1, 2]


@respx.mock
async def test_post_429_is_explicit_rejection_and_can_retry(monkeypatch):
    async def sleep(_):
        pass

    monkeypatch.setattr("app.services.parsing.mineru_client.asyncio.sleep", sleep)
    route = respx.post(f"{BASE}/file-urls/batch")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(
            200, json={"code": 0, "data": {"batch_id": "batch-1", "file_urls": [UPLOAD]}}
        ),
    ]
    async with httpx.AsyncClient() as http:
        await MinerUClient(settings(), http).request_upload("paper.pdf", "paper-1")
    assert route.call_count == 2


@pytest.mark.parametrize(
    "body,code",
    [
        (result("failed", err_msg="signed secret"), "remote_failed"),
        (result("new-state"), "invalid_response"),
        (result("done"), "invalid_response"),
        (
            {"code": 0, "data": {"extract_result": [{"data_id": "another", "state": "done"}]}},
            "invalid_response",
        ),
    ],
)
@respx.mock
async def test_failed_and_malformed_results(body, code):
    respx.get(f"{BASE}/extract-results/batch/batch-1").respond(json=body)
    async with httpx.AsyncClient() as http:
        with pytest.raises(ParsingError) as exc:
            await MinerUClient(settings(), http).poll("batch-1", "paper-1")
    assert exc.value.code == code and "secret" not in str(exc.value)


@respx.mock
async def test_poll_timeout_includes_inflight_request():
    async def slow(_):
        await asyncio.sleep(1)
        return httpx.Response(200, json=result("pending"))

    respx.get(f"{BASE}/extract-results/batch/batch-1").mock(side_effect=slow)
    async with httpx.AsyncClient() as http:
        with pytest.raises(ParsingError) as exc:
            await MinerUClient(settings(mineru_poll_timeout_s=0.01), http).poll(
                "batch-1", "paper-1"
            )
    assert exc.value.code == "poll_timeout" and exc.value.retryable


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn-mineru.openxlab.org.cn/file",
        "https://127.0.0.1/file",
        "https://cdn-mineru.openxlab.org.cn.evil.test/file",
        "https://user@cdn-mineru.openxlab.org.cn/file",
        "https://cdn-mineru.openxlab.org.cn:444/file",
        "file:///etc/passwd",
    ],
)
def test_url_allowlist_rejects_unsafe_targets(url):
    with pytest.raises(ParsingError, match="白名单"):
        check_remote_url(url, settings().mineru_download_hosts)
