import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.services.parsing.archive import read_json, safe_member
from app.services.parsing.errors import ParsingError
from app.services.parsing.pdf import PDFInfo, PDFPage


class Location(BaseModel):
    page_idx: int = Field(ge=0)
    # Origin is the TOP LEFT of the displayed PDF crop box, after rotation.
    bbox: tuple[float, float, float, float] | None = None


class Section(BaseModel):
    id: str
    parent_id: str | None
    title: str
    level: int
    path: list[str]


class ParsedBlock(BaseModel):
    id: str
    kind: Literal["title", "text", "table", "formula", "image", "caption", "reference", "footnote"]
    text: str
    locations: list[Location]
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    asset_path: str | None = None
    indexable: bool = True


class ParsedDocument(BaseModel):
    schema_version: str = "papermind.parse.v1"
    page_count: int
    blocks: list[ParsedBlock]
    sections: list[Section]
    markdown_path: str | None
    content_list_path: str
    layout_path: str | None
    assets: list[str]
    warnings: list[str]


def _box(value: Any, width: float, height: float) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
        for v in value
    ):
        return None
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    return x0 / width, y0 / height, x1 / width, y1 / height


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(2, b * 0.01)


def convert_bbox(
    value: Any, page: PDFPage | None = None, layout_size: Any = None
) -> tuple[float, float, float, float] | None:
    """Cloud Content List V1 is normalized to 1000, NOT PDF points.

    Resolve orientation against recorded layout geometry. Never guess a frame
    for rotated/cropped pages if geometry is absent or inconsistent.
    """
    box = _box(value, 1000, 1000)
    if box is None or page is None:
        return box
    cx0, cy0, cx1, cy1 = page.crop_box
    mx0, my0, mx1, my1 = page.media_box
    cropped = page.crop_box != page.media_box
    if layout_size is None:
        return None if page.rotation or cropped else box
    if (
        not isinstance(layout_size, list)
        or len(layout_size) != 2
        or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in layout_size
        )
    ):
        return None
    lw, lh = layout_size
    # Cloud commonly returns the displayed page directly. No second rotation.
    if _close(lw, page.width) and _close(lh, page.height):
        return box
    cw, ch = cx1 - cx0, cy1 - cy0
    x0, y0, x1, y1 = box
    if _close(lw, cw) and _close(lh, ch):
        pass  # Crop frame before rotation.
    elif _close(lw, mx1 - mx0) and _close(lh, my1 - my0):
        # Media-frame top-left coordinates -> crop-frame top-left coordinates.
        box = _box(
            [
                (x0 * lw - (cx0 - mx0)),
                (y0 * lh - (my1 - cy1)),
                (x1 * lw - (cx0 - mx0)),
                (y1 * lh - (my1 - cy1)),
            ],
            cw,
            ch,
        )
        if box is None:
            return None
        x0, y0, x1, y1 = box
    else:
        return None
    if page.rotation == 90:
        return 1 - y1, x0, 1 - y0, x1
    if page.rotation == 180:
        return 1 - x1, 1 - y1, 1 - x0, 1 - y0
    if page.rotation == 270:
        return y0, 1 - x1, y1, 1 - x0
    return x0, y0, x1, y1


def _one(paths: list[Path], label: str, *, required: bool = False) -> Path | None:
    if len(paths) > 1 or (required and not paths):
        raise ParsingError(f"结果包中 {label} 缺失或不唯一", code="invalid_result")
    return paths[0] if paths else None


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    return []


def load_result(root: Path, pdf: PDFInfo | None = None) -> ParsedDocument:
    candidates = list(root.rglob("*_content_list.json")) + list(root.rglob("content_list.json"))
    content_path = _one(candidates, "content_list.json", required=True)
    assert content_path is not None
    content = read_json(content_path)
    if (
        not isinstance(content, list)
        or not content
        or any(not isinstance(r, dict) for r in content)
    ):
        raise ParsingError("content_list 结构无效或为空", code="invalid_result")
    layout_path = _one(list(root.rglob("layout.json")) + list(root.rglob("middle.json")), "layout")
    layout = read_json(layout_path) if layout_path else {}
    if not isinstance(layout, dict):
        raise ParsingError("layout 结构无效", code="invalid_result")
    sizes = {
        p.get("page_idx"): p.get("page_size")
        for p in layout.get("pdf_info", [])
        if isinstance(p, dict)
    }
    pages = [r.get("page_idx") for r in content]
    if any(isinstance(p, bool) or not isinstance(p, int) or p < 0 for p in pages):
        raise ParsingError("结果页码无效", code="invalid_result")
    count = len(pdf.pages) if pdf else max(pages) + 1
    if any(p >= count for p in pages):
        raise ParsingError("解析页码超过原 PDF 页数", code="invalid_result")
    blocks: list[ParsedBlock] = []
    sections: list[Section] = []
    stack: list[Section] = []
    warnings: list[str] = []
    assets: set[str] = set()
    excluded = {"header", "footer", "page_header", "page_footer", "page_number", "aside_text"}
    for index, row in enumerate(content):
        kind = row.get("type")
        if kind in excluded:
            continue
        page_idx = row["page_idx"]
        bbox = convert_bbox(
            row.get("bbox"), pdf.pages[page_idx] if pdf else None, sizes.get(page_idx)
        )
        if bbox is None:
            warnings.append(f"block-{index}: bbox 不可用，保留页级定位")
        location = Location(page_idx=page_idx, bbox=bbox)
        text = row.get("text", "")
        if not isinstance(text, str):
            text = ""
        block_kind: Any = "text"
        caption: list[str] = []
        if kind == "text" and row.get("text_level"):
            level = row["text_level"]
            if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 6:
                raise ParsingError("章节层级无效", code="invalid_result")
            while stack and stack[-1].level >= level:
                stack.pop()
            section = Section(
                id=f"section-{index}",
                parent_id=stack[-1].id if stack else None,
                title=text,
                level=level,
                path=[s.title for s in stack] + [text],
            )
            sections.append(section)
            stack.append(section)
            block_kind = "title"
        elif kind == "table":
            block_kind = "table"
            text = "\n".join(_strings(row.get("table_body")) + _strings(row.get("table_footnote")))
            caption = _strings(row.get("table_caption"))
        elif kind in {"image", "chart"}:
            block_kind = "image"
            text = "\n".join(_strings(row.get("content")))
            caption = _strings(row.get(f"{kind}_caption")) + _strings(row.get(f"{kind}_footnote"))
        elif kind in {"equation", "interline_equation"}:
            block_kind = "formula"
        elif kind in {"ref_text", "reference"}:
            block_kind = "reference"
        elif kind == "page_footnote":
            block_kind = "footnote"
        elif kind not in {"text", "list", "code"}:
            warnings.append(f"block-{index}: 未识别类型 {str(kind)[:40]}，保留文本")
        if kind == "list":
            text = "\n".join(_strings(row.get("list_items")))
        elif kind == "code":
            text = "\n".join(_strings(row.get("code_body")) or _strings(row.get("text")))
            caption = _strings(row.get("code_caption"))
        asset = row.get("img_path")
        if asset:
            if not isinstance(asset, str):
                raise ParsingError("图片路径格式无效", code="invalid_result")
            relative = safe_member(asset)
            target = content_path.parent / str(relative)
            if (
                target.is_file()
                and not target.is_symlink()
                and target.resolve().is_relative_to(root.resolve())
            ):
                asset = str(target.relative_to(root))
                assets.add(asset)
            else:
                warnings.append(f"block-{index}: 图片缺失")
                asset = None
        section_id = stack[-1].id if stack else None
        section_path = stack[-1].path if stack else []
        reference_section = any(
            s.strip().casefold() in {"references", "bibliography", "参考文献"} for s in section_path
        )
        blocks.append(
            ParsedBlock(
                id=f"block-{index}",
                kind=block_kind,
                text=text,
                locations=[location],
                section_id=section_id,
                section_path=section_path,
                asset_path=asset,
                indexable=not reference_section and block_kind != "reference",
            )
        )
        if caption:
            # A content-list bbox spans the parent object, not the caption alone.
            blocks.append(
                ParsedBlock(
                    id=f"block-{index}-caption",
                    kind="caption",
                    text="\n".join(caption),
                    locations=[Location(page_idx=page_idx)],
                    section_id=section_id,
                    section_path=section_path,
                    indexable=not reference_section,
                )
            )
    markdown = _one(list(root.rglob("full.md")), "full.md")
    return ParsedDocument(
        page_count=count,
        blocks=blocks,
        sections=sections,
        markdown_path=str(markdown.relative_to(root)) if markdown else None,
        content_list_path=str(content_path.relative_to(root)),
        layout_path=str(layout_path.relative_to(root)) if layout_path else None,
        assets=sorted(assets),
        warnings=warnings,
    )
