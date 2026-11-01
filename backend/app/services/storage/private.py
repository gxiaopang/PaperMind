import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, Request

from app.core.config import Settings
from app.services.parsing.pipeline import private_directory


def paper_directory(settings: Settings, user_id: UUID, paper_id: UUID) -> Path:
    return settings.storage_root / str(user_id) / str(paper_id)


def resolve_private(settings: Settings, relative: str) -> Path:
    root = settings.storage_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("invalid private path")
    return path


async def receive_pdf(request: Request, settings: Settings) -> Path:
    directory = settings.storage_root / "incoming"
    private_directory(directory)
    path = directory / f"{uuid4()}.pdf"
    size = 0
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            async for chunk in request.stream():
                size += len(chunk)
                if size > settings.max_upload_mb * 1024**2:
                    raise HTTPException(413, "PDF 超过上传大小限制")
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink()
    )
