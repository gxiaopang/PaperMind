"""Structure-aware parent/child chunks with offline BGE-M3 token budgets."""

import hashlib
import re
from dataclasses import dataclass, field

from app.services.chunking.academic import (
    cited_references,
    context_prefix,
    heading,
    is_references,
    markdown_table,
    references,
    section_role,
    table_summary,
)
from app.services.chunking.crossrefs import attach_retrieval_index
from app.services.chunking.tokenizer import bounded, token_count
from app.services.parsing.mineru_result import ParsedBlock, ParsedDocument

CHUNKER_VERSION = "academic-bge-v5"
TARGET = 512
OVERLAP = 64
HARD_LIMIT = 1024

# These spans must never be cut by sentence or word fallback. Oversized atomic
# content remains available as an unindexed preview, with its original location.
PROTECTED = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|(?<!\\)\$(?:\\.|[^$\\])*?(?<!\\)\$"
    r"|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|<[^>\n]+>",
)
ABBREVIATIONS = {
    "fig",
    "figs",
    "eq",
    "eqs",
    "sec",
    "secs",
    "dr",
    "mr",
    "mrs",
    "prof",
    "al",
    "vs",
    "e.g",
    "i.e",
}


def _trim(value: str, start: int, end: int) -> tuple[int, int]:
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    return start, end


def _sentences(value: str, start: int, end: int) -> list[tuple[int, int]]:
    protected = [(m.start(), m.end()) for m in PROTECTED.finditer(value, start, end)]
    spans, cursor = [], start
    for match in re.finditer(r"[.!?。！？]+[\"'”’）)\]]*", value[start:end]):
        left, right = start + match.start(), start + match.end()
        if any(a <= left < b for a, b in protected):
            continue
        if value[left] == ".":
            previous = re.search(r"([\w.]+)$", value[start:left])
            if previous and previous.group(1).casefold() in ABBREVIATIONS:
                continue
            if right < end and not value[right].isspace():
                continue
        a, b = _trim(value, cursor, right)
        if a < b:
            spans.append((a, b))
        cursor = right
    a, b = _trim(value, cursor, end)
    if a < b:
        spans.append((a, b))
    return spans


def _words(value: str, start: int, end: int) -> list[tuple[int, int]]:
    """Word/CJK boundaries outside math, code and markup; never partial words."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in PROTECTED.finditer(value, start, end):
        spans.extend(
            (cursor + m.start(), cursor + m.end())
            for m in re.finditer(
                r"[\u3400-\u9fff]|[^\s\u3400-\u9fff]+", value[cursor : match.start()]
            )
        )
        spans.append((match.start(), match.end()))
        cursor = match.end()
    spans.extend(
        (cursor + m.start(), cursor + m.end())
        for m in re.finditer(r"[\u3400-\u9fff]|[^\s\u3400-\u9fff]+", value[cursor:end])
    )
    return spans


def _pack(value: str, units: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for start, end in units:
        if current is not None and token_count(value[current[0] : end]) > target:
            spans.append(current)
            current = None
        current = (current[0] if current else start, end)
    if current:
        spans.append(current)
    return spans


def split_spans(
    value: str, target: int, hard_limit: int, overlap: int = 0
) -> list[tuple[int, int]]:
    """Keep whole paragraphs/sentences past the soft target, under the hard cap."""
    paragraphs, cursor = [], 0
    protected = [(m.start(), m.end()) for m in PROTECTED.finditer(value)]
    for match in re.finditer(r"\n\s*\n", value):
        if any(a <= match.start() < b for a, b in protected):
            continue
        paragraphs.append(_trim(value, cursor, match.start()))
        cursor = match.end()
    paragraphs.append(_trim(value, cursor, len(value)))
    units = []
    oversized = []
    for start, end in paragraphs:
        if start == end:
            continue
        if token_count(value[start:end]) <= hard_limit:
            units.append((start, end))
            continue
        oversized.append((start, end))
        for a, b in _sentences(value, start, end):
            if token_count(value[a:b]) <= hard_limit:
                units.append((a, b))
            else:
                units.extend(_pack(value, _words(value, a, b), target))
    spans = _pack(value, units, target)
    result = []
    for i, (start, end) in enumerate(spans):
        # Whole-element boundaries need no overlap. Only split oversized
        # paragraphs get a complete-word tail, subject to the actual hard limit.
        if overlap and i and any(a < start < b for a, b in oversized):
            previous_start = spans[i - 1][0]
            boundary = start
            for a, _ in reversed(_words(value, previous_start, boundary)):
                if (
                    token_count(value[a:boundary]) > overlap
                    or token_count(value[a:end]) > hard_limit
                ):
                    break
                start = a
        result.append((start, end))
    return result


@dataclass
class ParentDraft:
    ordinal: int
    kind: str
    text: str
    section_id: str | None
    section_path: list[str]
    locations: list[dict]


@dataclass
class ChunkDraft:
    kind: str
    text: str
    embedding_input: str
    section_id: str | None
    section_path: list[str]
    locations: list[dict]
    indexable: bool = True
    unindexed_reason: str | None = None
    parent: ParentDraft | None = None
    source_spans: list[dict] | None = None

    chunk_metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embedding_input.encode()).hexdigest()

    @property
    def token_count(self) -> int:
        return token_count(self.embedding_input)


def chunk_document(
    document: ParsedDocument,
    title: str,
    abstract: str | None = None,
    authors: list[str] | None = None,
) -> list[ChunkDraft]:
    """L0 overview, L1 native sections, L2 citable semantic/modality units."""
    result: list[ChunkDraft] = []
    entries = references(document)
    blocks = [b for b in document.blocks if b.indexable and not is_references(b)]

    def locations(items: list[ParsedBlock]) -> list[dict]:
        return list(
            {
                (loc.page_idx, loc.bbox): loc.model_dump(mode="json")
                for item in items
                for loc in item.locations
            }.values()
        )

    parent_ordinal = 0

    def parent_for(
        items: list[ParsedBlock], kind: str, text: str | None = None, path: list[str] | None = None
    ) -> ParentDraft:
        nonlocal parent_ordinal
        first = items[0] if items else None
        parent = ParentDraft(
            parent_ordinal,
            kind,
            text if text is not None else "\n\n".join(b.text for b in items),
            first.section_id if first else None,
            path if path is not None else (first.section_path if first else []),
            locations(items),
        )
        parent_ordinal += 1
        return parent

    def emit(
        items: list[ParsedBlock],
        kind: str,
        parent: ParentDraft,
        *,
        text: str | None = None,
        retrieval_text: str | None = None,
        spans: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> None:
        value = text if text is not None else "\n\n".join(b.text for b in items)
        role = section_role(parent.section_path)
        if kind == "document_meta":
            role = "全文概览（摘要、贡献与结论）"
        elif kind == "figure":
            role = "图注与跨章节分析（按原文归属）"
        modality = "paragraph" if kind in {"text", "abstract"} else kind
        prefix = context_prefix(title, parent.section_path, role, modality)
        embedding_input = prefix + (retrieval_text if retrieval_text is not None else value)
        reason = None
        if token_count(embedding_input) > HARD_LIMIT:
            reason = (
                "formula_exceeds_input_limit"
                if kind == "formula"
                else "atomic_span_exceeds_input_limit"
            )
        result.append(
            ChunkDraft(
                kind,
                value,
                embedding_input,
                parent.section_id,
                parent.section_path,
                locations(items),
                reason is None,
                reason,
                parent,
                spans
                if spans is not None
                else [{"block_id": b.id, "start": 0, "end": len(b.text)} for b in items],
                {
                    "level": 0 if kind == "document_meta" else 2,
                    "section_role": role,
                    "context_prefix": prefix,
                    "citations": cited_references(value, entries),
                    **(
                        {
                            "source_provenance": [
                                {
                                    "block_id": item.id,
                                    "section_path": item.section_path,
                                    "section_role": section_role(item.section_path),
                                }
                                for item in items
                            ]
                        }
                        if kind in {"document_meta", "figure"}
                        else {}
                    ),
                    **(metadata or {}),
                },
            )
        )

    # The overview is an evidence aggregation, not a generated factual summary.
    abstract_blocks = [
        b
        for b in blocks
        if b.kind == "text" and any(heading(p) in {"abstract", "摘要"} for p in b.section_path)
    ]
    intro = [
        b
        for b in blocks
        if b.kind == "text"
        and b.section_path
        and any(heading(p) in {"introduction", "引言", "绪论"} for p in b.section_path)
    ]
    conclusion = [
        b
        for b in blocks
        if b.kind == "text"
        and any(
            heading(p) in {"conclusion", "conclusions", "concluding remarks", "结论", "总结"}
            for p in b.section_path
        )
    ]
    abstract_text = "\n\n".join(b.text for b in abstract_blocks) or abstract or ""
    overview_items = abstract_blocks + intro[-1:] + conclusion
    if abstract_text or intro or conclusion or authors:
        components = [
            ("Title", title),
            ("Authors", ", ".join(authors or [])),
            ("Abstract", abstract_text),
            ("Introduction (last paragraph)", intro[-1].text if intro else ""),
            ("Conclusion", "\n\n".join(b.text for b in conclusion)),
        ]
        value = "\n\n".join(f"{label}: {body}" for label, body in components if body)
        # Each component has its own allowance, so conclusions survive long abstracts.
        proxy = "\n".join(
            f"{label}: {bounded(body, cap)}"
            for (label, body), cap in zip(components, (80, 60, 220, 180, 220), strict=True)
            if body
        )
        parent = parent_for(overview_items, "document_meta", value, ["Document Overview"])
        parent.section_id = None
        emit(
            overview_items,
            "document_meta",
            parent,
            text=value,
            retrieval_text=proxy,
            metadata={"representation": "extractive_overview"},
        )

    # Reuse the metadata abstract without inventing a PDF page or bounding box.
    if abstract and not abstract_blocks:
        blocks.insert(
            0,
            ParsedBlock(
                id="metadata:abstract",
                kind="text",
                text=abstract,
                locations=[],
                section_path=["Abstract"],
            ),
        )

    groups: list[list[ParsedBlock]] = []
    for block in blocks:
        if block.kind == "title":
            continue
        if not groups or (groups[-1][0].section_id, groups[-1][0].section_path) != (
            block.section_id,
            block.section_path,
        ):
            groups.append([])
        groups[-1].append(block)

    figure_caption = re.compile(r"^\s*(?:figure|fig\.?|图)\s*(\d+[a-z]?)\b", re.I)
    for group in groups:
        usable = [b for b in group if b.kind != "image" and b.text.strip()]
        if not usable:
            continue
        parent = parent_for(usable, "section")
        units: list[list[ParsedBlock]] = []
        for b in usable:
            # Glue a formula to its immediate introduction and definitions; merge
            # chains transitively so neither neighboring formulas lose context.
            if b.kind == "formula" and units and units[-1][-1].kind in {"text", "formula"}:
                units[-1].append(b)
            elif b.kind == "text" and units and units[-1][-1].kind == "formula":
                units[-1].append(b)
            elif b.kind == "caption" and units and units[-1][0].kind == "table":
                units[-1].append(b)
            else:
                units.append([b])
        normal: list[ParsedBlock] = []

        def emit_normal(
            normal: list[ParsedBlock] = normal,
            parent: ParentDraft = parent,
        ) -> None:
            if not normal:
                return
            value = "\n\n".join(b.text for b in normal)
            offsets = []
            cursor = 0
            for item in normal:
                offsets.append((cursor, cursor + len(item.text), item))
                cursor += len(item.text) + 2
            prefix = context_prefix(title, parent.section_path, section_role(parent.section_path))
            budget = HARD_LIMIT - token_count(prefix) - 8
            for a, b in split_spans(value, min(TARGET, budget), budget, OVERLAP):
                touched = [(s, e, item) for s, e, item in offsets if s < b and e > a]
                kind = (
                    "abstract"
                    if any(heading(p) in {"abstract", "摘要"} for p in parent.section_path)
                    else "text"
                )
                emit(
                    [item for _, _, item in touched],
                    kind,
                    parent,
                    text=value[a:b],
                    spans=[
                        {"block_id": item.id, "start": max(a, s) - s, "end": min(b, e) - s}
                        for s, e, item in touched
                    ],
                )
            normal.clear()

        for unit in units:
            if all(b.kind == "text" for b in unit):
                normal.extend(unit)
                continue
            emit_normal()
            if any(b.kind == "formula" for b in unit):
                emit(unit, "formula", parent, metadata={"formula_context": len(unit) > 1})
            elif unit[0].kind == "table":
                table_caption = "\n".join(b.text for b in unit[1:])
                summary = table_summary(unit[0].text, table_caption)
                emit(
                    unit,
                    "table",
                    parent,
                    retrieval_text=summary,
                    metadata={
                        "retrieval_summary": summary,
                        "summary_method": "extractive",
                        "table_caption": table_caption,
                        "raw_table": unit[0].text,
                        "markdown_table": markdown_table(unit[0].text),
                    },
                )
            else:
                emit(unit, unit[0].kind, parent)
        emit_normal()

    # Virtual figure sources include only matched sentences and real source spans.
    for i, caption in enumerate(blocks):
        match = figure_caption.match(caption.text) if caption.kind == "caption" else None
        if not match:
            continue
        number = match.group(1)
        mention = re.compile(
            r"(?:\b(?:figure|figs?\.?)\s*|图\s*)" + re.escape(number) + r"(?![\w])", re.I
        )
        pieces, items, spans = (
            [caption.text],
            [caption],
            [{"block_id": caption.id, "start": 0, "end": len(caption.text)}],
        )
        for block in blocks:
            if block.kind != "text":
                continue
            for left, right in _sentences(block.text, 0, len(block.text)):
                if mention.search(block.text[left:right]):
                    pieces.append(block.text[left:right])
                    items.append(block)
                    spans.append({"block_id": block.id, "start": left, "end": right})
        image = blocks[i - 1] if i and blocks[i - 1].kind == "image" else None
        if image and image.text.strip():
            pieces.append(image.text)
            items.append(image)
            spans.append({"block_id": image.id, "start": 0, "end": len(image.text)})
        value = "\n\n".join(pieces)
        parent = parent_for(items, "figure", value)
        emit(
            items,
            "figure",
            parent,
            text=value,
            retrieval_text=bounded(value, 700),
            spans=spans,
            metadata={
                "figure_id": number,
                "visual_summary_available": bool(image and image.text.strip()),
                "source_sections": list(dict.fromkeys(" > ".join(b.section_path) for b in items)),
            },
        )
    attach_retrieval_index(document, result)
    return result
