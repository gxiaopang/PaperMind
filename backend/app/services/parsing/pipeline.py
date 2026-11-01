import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from app.core.config import Settings
from app.services.parsing.archive import download_archive, extract_archive
from app.services.parsing.errors import ParsingError
from app.services.parsing.metadata import Extractor, MetadataService, model_extractor
from app.services.parsing.mineru_client import MinerUClient, RemoteResult, UploadTicket
from app.services.parsing.mineru_result import load_result
from app.services.parsing.pdf import inspect_pdf


def private_json(path: Path, data: Any) -> None:
    """Publish durable state without exposing a partially written manifest."""
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as target:
        os.fchmod(target.fileno(), 0o600)
        json.dump(data, target, ensure_ascii=False, indent=2)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


async def parse_pdf(
    pdf_path: Path,
    output: Path,
    settings: Settings,
    *,
    batch_id: str | None = None,
    use_llm: bool = True,
    extractor: Extractor | None = None,
    notify: Callable[[str], None] = lambda _: None,
    checkpoint: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Durable cloud workflow shared by the authenticated worker and operator CLI.

    The output directory must be private to this job. Callers serialize access
    (the CLI takes a filesystem lock). No external submission until validation.
    """
    pdf = await asyncio.to_thread(inspect_pdf, pdf_path, settings)
    await asyncio.to_thread(private_directory, output)
    manifest_path = output / "job.json"
    identity = {
        "sha256": pdf.sha256,
        "model": settings.mineru_model_version,
        "formula": settings.mineru_enable_formula,
        "table": settings.mineru_enable_table,
        "language": settings.mineru_language,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest: dict[str, Any] = {"identity": identity, "fingerprint": fingerprint, "stage": "new"}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") != fingerprint:
            raise ParsingError("输出目录对应其他 PDF 或解析配置，请使用新目录", code="job_mismatch")
        if batch_id and manifest.get("batch_id") not in {None, batch_id}:
            raise ParsingError("恢复批次与现有任务不匹配", code="job_mismatch")
    if manifest["stage"] == "complete":
        return json.loads((output / "report.json").read_text())
    if batch_id:
        manifest.update(batch_id=batch_id, stage="polling")
        private_json(manifest_path, manifest)
    if manifest["stage"] in {"submitting", "submission_uncertain"}:
        raise ParsingError(
            "上次提交结果未知，请核对远端后用 --batch-id 恢复", code="submission_uncertain"
        )

    async def save_checkpoint() -> None:
        if checkpoint:
            await checkpoint(manifest)

    await save_checkpoint()
    async with httpx.AsyncClient() as http:
        client = MinerUClient(settings, http)
        try:
            if not manifest.get("batch_id"):
                # Persist intention before the non-idempotent remote request.
                manifest["stage"] = "submitting"
                private_json(manifest_path, manifest)
                await save_checkpoint()
                notify("申请 MinerU 上传地址")
                try:
                    ticket = await client.request_upload(pdf_path.name, pdf.sha256)
                except ParsingError as error:
                    manifest["stage"] = (
                        "submission_uncertain"
                        if error.code == "submission_uncertain"
                        else "rejected"
                    )
                    private_json(manifest_path, manifest)
                    raise
                manifest.update(
                    batch_id=ticket.batch_id, upload_url=ticket.upload_url, stage="uploading"
                )
                private_json(manifest_path, manifest)
            await save_checkpoint()
            if manifest["stage"] == "uploading":
                # A crash after successful PUT should resume polling, not upload
                # again if the server already accepted the file.
                current = await client.result(manifest["batch_id"], pdf.sha256)
                if current.state == "waiting-file":
                    notify("上传 PDF 至 MinerU")
                    await client.upload(
                        UploadTicket(
                            batch_id=manifest["batch_id"],
                            data_id=pdf.sha256,
                            upload_url=manifest["upload_url"],
                        ),
                        pdf_path,
                    )
                manifest.update(stage="polling")
                manifest.pop("upload_url", None)
                private_json(manifest_path, manifest)

            await save_checkpoint()

            async def progress(result: RemoteResult) -> None:
                manifest.update(remote_state=result.state, progress=result.extract_progress)
                if result.state == "failed":
                    # Diagnostic vendor text stays in this private job file,
                    # never in console output or the public error message.
                    manifest["remote_error"] = result.err_msg[:2000]
                private_json(manifest_path, manifest)
                await save_checkpoint()
                notify(f"MinerU: {result.state}")

            raw = output / "raw"
            if not raw.exists():
                result = await client.poll(manifest["batch_id"], pdf.sha256, progress)
                archive = output / "result.zip"
                assert result.full_zip_url
                await save_checkpoint()
                await download_archive(http, result.full_zip_url, archive, settings)
                await asyncio.to_thread(extract_archive, archive, raw, settings)
                manifest["stage"] = "normalizing"
                private_json(manifest_path, manifest)
            await save_checkpoint()
            document = await asyncio.to_thread(load_result, raw, pdf)
            private_json(output / "document.json", document.model_dump())
            metadata_service = MetadataService(
                http, (extractor or model_extractor(settings)) if use_llm else None
            )
            metadata = await metadata_service.enrich(pdf, document)
            private_json(output / "metadata.json", metadata.model_dump())
            counts: dict[str, int] = {}
            for block in document.blocks:
                counts[block.kind] = counts.get(block.kind, 0) + 1
            report = {
                "pdf_sha256": pdf.sha256,
                "pages": document.page_count,
                "blocks": counts,
                "sections": len(document.sections),
                "assets": len(document.assets),
                "locations": sum(len(b.locations) for b in document.blocks),
                "boxes": sum(loc.bbox is not None for b in document.blocks for loc in b.locations),
                "warnings": document.warnings,
                "metadata_warnings": metadata.warnings,
            }
            private_json(output / "report.json", report)
            manifest.update(stage="complete", error=None)
            private_json(manifest_path, manifest)
            await save_checkpoint()
            return report
        except ParsingError as error:
            manifest["error"] = {
                "code": error.code,
                "message": str(error),
                "retryable": error.retryable,
            }
            private_json(manifest_path, manifest)
            raise
