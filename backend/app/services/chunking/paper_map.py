"""Lossless navigation metadata, separate from bounded retrieval representations."""

import re
from collections.abc import Mapping
from uuid import UUID

from app.services.chunking.academic import heading
from app.services.chunking.crossrefs import MENTION, labels
from app.services.parsing.mineru_result import ParsedDocument


def bind_section_ids(value: dict, mapping: Mapping[str, UUID | None]) -> dict:
    def remap(nodes: list[dict]) -> None:
        for node in nodes:
            node["section_id"] = str(mapping[node["section_id"]])
            remap(node["children"])

    remap(value["toc"])
    for element in value["elements"]:
        target = mapping.get(element["section_id"])
        element["section_id"] = str(target) if target else None
    return value


def section_std(path: list[str]) -> str:
    for pattern, name in (
        (r"limitation|局限|限制", "limitations"),
        (r"related work|prior work|相关工作", "related_work"),
        (r"ablation|消融", "ablation"),
    ):
        if any(re.search(pattern, heading(part)) for part in path):
            return name
    for part in reversed(path):
        value = heading(part)
        for pattern, name in (
            (r"limitation|局限|限制", "limitations"),
            (r"related work|prior work|相关工作", "related_work"),
            (r"ablation|消融", "ablation"),
            (r"abstract|摘要", "abstract"),
            (r"introduct|引言|绪论", "introduction"),
            (r"conclu|结论|总结", "conclusion"),
            (r"method|approach|model|framework|方法|模型|框架", "method"),
            (r"experiment|evaluat|result|实验|评估|结果", "experiments"),
            (r"discuss|讨论", "discussion"),
            (r"appendix|appendices|附录", "appendix"),
            (r"reference|bibliograph|参考文献", "references"),
        ):
            if re.search(pattern, value):
                return name
    return "other"


def build_paper_map(document: ParsedDocument, title: str, abstract: str | None) -> dict:
    abstract_text = (
        "\n\n".join(
            b.text
            for b in document.blocks
            if b.kind == "text" and section_std(b.section_path) == "abstract"
        )
        or abstract
        or ""
    )
    nodes: dict[str, dict] = {}
    toc: list[dict] = []
    # Depth is relative to actual parents; parser heading levels can start at 2.
    for section in document.sections:
        parent = nodes.get(section.parent_id or "")
        depth = parent["depth"] + 1 if parent else 1
        node = {
            "section_id": section.id,
            "title": section.title,
            "section_std": section_std(section.path),
            "path": section.path,
            "depth": depth,
            "children": [],
        }
        nodes[section.id] = node
        if depth <= 2:
            (parent["children"] if parent else toc).append(node)
    elements: list[dict] = []
    for block in document.blocks:
        if block.kind not in {"caption", "table"}:
            continue
        match = MENTION.match(block.text.lstrip())
        for label in labels(match.group(0) if match else ""):
            family, number = label.split(":", 1)
            if family not in {"figure", "table"}:
                continue
            elements.append(
                {
                    "label": f"{family.title()} {number}",
                    "key": label,
                    "caption": block.text,
                    "section_id": block.section_id,
                    "locations": [loc.model_dump(mode="json") for loc in block.locations],
                }
            )
    return {
        "schema_version": "papermind.map.v1",
        "title": title,
        "abstract": abstract_text,
        "toc": toc,
        "elements": elements,
    }
