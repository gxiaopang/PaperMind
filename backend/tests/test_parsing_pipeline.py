import json

import httpx
import pytest
import respx

from app.services.parsing.errors import ParsingError, SubmissionUncertain
from app.services.parsing.metadata import MetadataExtraction
from app.services.parsing.pdf import inspect_pdf
from app.services.parsing.pipeline import parse_pdf
from tests.test_mineru_archive import zip_bytes
from tests.test_mineru_client import BASE, UPLOAD, ZIP, settings
from tests.test_parsing_pdf import make_pdf


def setup_routes(path, *, post_failure=None, put_failure=None):
    digest = inspect_pdf(path, settings()).sha256
    post = respx.post(f"{BASE}/file-urls/batch")
    if post_failure:
        post.mock(side_effect=post_failure)
    else:
        post.respond(json={"code": 0, "data": {"batch_id": "known-batch", "file_urls": [UPLOAD]}})
    put = respx.put(UPLOAD)
    if put_failure:
        put.mock(side_effect=put_failure)
    else:
        put.respond(200)

    def body(state):
        return {
            "code": 0,
            "data": {"extract_result": [{"data_id": digest, "state": state, "full_zip_url": ZIP}]},
        }

    poll = respx.get(f"{BASE}/extract-results/batch/known-batch")
    poll.side_effect = [
        httpx.Response(200, json=body("waiting-file")),
        httpx.Response(200, json=body("done")),
    ]
    content = [
        {
            "type": "text",
            "text": "Test paper",
            "text_level": 1,
            "page_idx": 0,
            "bbox": [100, 200, 500, 600],
        }
    ]
    download = respx.get(ZIP).respond(
        content=zip_bytes([("content_list.json", json.dumps(content)), ("full.md", "# Test paper")])
    )
    return post, put, poll, download, body


@respx.mock
async def test_pipeline_is_durable_private_and_completed_jobs_do_not_call_cloud(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    output = tmp_path / "job"
    post, put, poll, download, _ = setup_routes(path)
    report = await parse_pdf(path, output, settings(), use_llm=False)
    manifest = json.loads((output / "job.json").read_text())
    assert manifest["stage"] == "complete" and "upload_url" not in manifest
    assert (output / "metadata.json").exists() and (output / "document.json").exists()
    assert (output / "job.json").stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    assert report["pages"] == 1 and report["blocks"]["title"] == 1
    assert await parse_pdf(path, output, settings(), use_llm=False) == report
    assert (
        await parse_pdf(path, output, settings(), batch_id="known-batch", use_llm=False) == report
    )
    assert [r.call_count for r in [post, put, poll, download]] == [1, 1, 2, 1]
    with pytest.raises(ParsingError, match="其他 PDF 或解析配置"):
        await parse_pdf(path, output, settings(mineru_language="en"), use_llm=False)


@respx.mock
async def test_ambiguous_post_survives_restart_and_requires_explicit_reconciliation(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    output = tmp_path / "job"
    post, put, poll, _, body = setup_routes(path, post_failure=httpx.ReadTimeout("secret"))
    with pytest.raises(SubmissionUncertain):
        await parse_pdf(path, output, settings(), use_llm=False)
    assert json.loads((output / "job.json").read_text())["stage"] == "submission_uncertain"
    with pytest.raises(ParsingError, match="核对远端"):
        await parse_pdf(path, output, settings(), use_llm=False)
    assert post.call_count == 1
    poll.side_effect = None
    poll.respond(json=body("done"))
    await parse_pdf(path, output, settings(), batch_id="known-batch", use_llm=False)
    assert post.call_count == 1 and put.call_count == 0


@respx.mock
async def test_connection_failure_can_resume_same_job_without_reconciliation(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    output = tmp_path / "job"
    post, put, _, _, _ = setup_routes(path, post_failure=httpx.ConnectError("TLS failure"))
    with pytest.raises(ParsingError) as exc:
        await parse_pdf(path, output, settings(mineru_http_attempts=1), use_llm=False)
    manifest = json.loads((output / "job.json").read_text())
    assert exc.value.code == "network" and exc.value.retryable
    assert manifest["stage"] == "rejected" and not manifest.get("batch_id")
    assert put.call_count == 0

    post.side_effect = None
    post.respond(json={"code": 0, "data": {"batch_id": "known-batch", "file_urls": [UPLOAD]}})
    await parse_pdf(path, output, settings(), use_llm=False)
    assert json.loads((output / "job.json").read_text())["stage"] == "complete"
    assert post.call_count == 2 and put.call_count == 1


@respx.mock
async def test_lost_upload_response_recovers_by_polling_saved_batch(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    output = tmp_path / "job"
    post, put, poll, _, body = setup_routes(path, put_failure=httpx.ReadTimeout("secret"))
    with pytest.raises(ParsingError) as exc:
        await parse_pdf(path, output, settings(mineru_http_attempts=1), use_llm=False)
    assert exc.value.code == "upload_network"
    state = json.loads((output / "job.json").read_text())
    assert state["batch_id"] == "known-batch" and state["stage"] == "uploading"
    poll.side_effect = None
    poll.respond(json=body("done"))
    await parse_pdf(path, output, settings(), use_llm=False)
    assert post.call_count == 1 and put.call_count == 1


@respx.mock
async def test_download_failure_resumes_saved_task_without_resubmission(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    output = tmp_path / "job"
    post, put, poll, download, body = setup_routes(path)
    archive = download.return_value
    download.respond(503)
    with pytest.raises(ParsingError) as exc:
        await parse_pdf(path, output, settings(mineru_http_attempts=1), use_llm=False)
    assert exc.value.code == "download_network"
    poll.side_effect = None
    poll.respond(json=body("done"))
    download.mock(return_value=archive)
    await parse_pdf(path, output, settings(), use_llm=False)
    assert post.call_count == 1 and put.call_count == 1


@respx.mock
async def test_bad_pdf_makes_no_remote_request(tmp_path):
    path = tmp_path / "not.pdf"
    path.write_bytes(b"bad")
    with pytest.raises(ParsingError):
        await parse_pdf(path, tmp_path / "job", settings(), use_llm=False)
    assert len(respx.calls) == 0


@pytest.mark.parametrize("enabled", [True, False])
@respx.mock
async def test_pipeline_injects_accounted_extractor_and_reuses_completed_metadata(
    tmp_path, enabled
):
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    output = tmp_path / "job"
    setup_routes(path)
    calls = []

    async def extractor(text):
        calls.append(text)
        return MetadataExtraction()

    await parse_pdf(path, output, settings(), use_llm=enabled, extractor=extractor)
    await parse_pdf(path, output, settings(), use_llm=enabled, extractor=extractor)
    assert len(calls) == int(enabled)
    if enabled:
        assert "Test paper" in calls[0]
