import hashlib
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

from app.core.config import Settings
from app.services.parsing.errors import ParsingError


class PDFPage(BaseModel):
    # Dimensions of the displayed crop box, including the PDF /Rotate value.
    width: float
    height: float
    rotation: int
    crop_box: tuple[float, float, float, float]
    media_box: tuple[float, float, float, float]


class PDFInfo(BaseModel):
    sha256: str
    size: int
    pages: list[PDFPage]
    title: str | None = None
    author: str | None = None
    first_page_text: str = ""


def inspect_pdf(path: Path, settings: Settings) -> PDFInfo:
    if not path.is_file() or path.stat().st_size > settings.max_upload_mb * 1024**2:
        raise ParsingError("PDF 不存在或超过上传大小限制", code="invalid_pdf")
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise ParsingError("文件不是 PDF", code="invalid_pdf")
    try:
        reader = PdfReader(path, strict=True)
        if reader.is_encrypted:
            raise ParsingError("不支持加密 PDF，请先解密", code="encrypted_pdf")
        if not 0 < len(reader.pages) <= settings.max_upload_pages:
            raise ParsingError("PDF 页数为空或超过限制", code="pdf_page_limit")
        pages = []
        for page in reader.pages:
            box = page.cropbox
            width, height = float(box.width), float(box.height)
            rotation = page.rotation % 360
            if width <= 0 or height <= 0 or rotation not in {0, 90, 180, 270}:
                raise ParsingError("PDF 页面尺寸或旋转无效", code="invalid_pdf")
            if rotation in {90, 270}:
                width, height = height, width
            pages.append(
                PDFPage(
                    width=width,
                    height=height,
                    rotation=rotation,
                    crop_box=tuple(float(v) for v in box),
                    media_box=tuple(float(v) for v in page.mediabox),
                )
            )
        metadata = reader.metadata
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        return PDFInfo(
            sha256=digest,
            size=path.stat().st_size,
            pages=pages,
            title=metadata.title if metadata else None,
            author=metadata.author if metadata else None,
            first_page_text=reader.pages[0].extract_text()[:30000],
        )
    except ParsingError:
        raise
    except Exception:
        raise ParsingError("PDF 损坏或无法读取", code="invalid_pdf") from None
