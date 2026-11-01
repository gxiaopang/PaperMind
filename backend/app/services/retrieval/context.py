"""Bounded context assembly; every returned source remains one real child chunk."""

import re
from collections import Counter

from langchain_core.documents import Document


def requires_neighbor(doc: Document) -> bool:
    value = doc.page_content.strip()
    return (
        doc.metadata.get("kind") in {"formula", "table", "caption"}
        or bool(re.match(r"^(?:where\b|here\b|in this equation\b|其中|式中|上述公式)", value, re.I))
        or not value.endswith((".", "!", "?", "。", "！", "？"))
    )


def mark_completeness(documents: list[Document], expected: dict[str, int]) -> None:
    counts = Counter(d.metadata.get("parent_id") for d in documents)
    for doc in documents:
        parent = doc.metadata.get("parent_id")
        if parent:
            doc.metadata["parent_complete"] = counts[parent] == expected.get(parent, -1)


def assemble_context(
    hits: list[Document],
    siblings: dict[str, list[Document]],
    neighbors: dict[str, list[Document]],
    expected: dict[str, int],
    *,
    budget: int,
    parent_budget: int,
    max_sources: int,
) -> list[Document]:
    selected: dict[str, Document] = {}
    parent_bytes: Counter = Counter()

    def add(doc: Document, role: str, anchor: str) -> None:
        nonlocal budget
        key, parent = doc.metadata["chunk_id"], doc.metadata.get("parent_id")
        size = len(doc.page_content.encode())
        if key in selected or size > budget or len(selected) >= max_sources:
            return
        if role != "hit" and parent_bytes[parent] + size > parent_budget:
            return
        current = doc.model_copy(deep=True)
        current.metadata.update(retrieval_role=role, context_anchor=anchor)
        selected[key] = current
        parent_bytes[parent] += size
        budget -= size

    # Reserve space for ranked hits before any optional context. Expanding one
    # parent cannot evict evidence from another relevant paper.
    for rank, hit in enumerate(hits, 1):
        add(hit, "hit", hit.metadata["chunk_id"])
        if hit.metadata["chunk_id"] in selected:
            selected[hit.metadata["chunk_id"]].metadata["hit_rank"] = rank
    anchors = [h for h in hits if h.metadata["chunk_id"] in selected]
    for hit in anchors:
        key = hit.metadata["chunk_id"]
        for doc in siblings.get(key, []):
            add(doc, "parent", key)
    for hit in anchors:
        key = hit.metadata["chunk_id"]
        for doc in neighbors.get(key, []):
            add(doc, "neighbor", key)

    # Keep parents in hit-priority order and their member sources in document
    # order. Do not concatenate text or assign a parent's bbox to its children.
    group_rank: dict[str, int] = {}
    anchor_group = {}
    for rank, hit in enumerate(anchors):
        key = hit.metadata["chunk_id"]
        group = hit.metadata.get("parent_id") or key
        group_rank.setdefault(group, rank)
        anchor_group[key] = group
    result = sorted(
        selected.values(),
        key=lambda d: (
            group_rank[anchor_group[d.metadata["context_anchor"]]],
            d.metadata.get("ordinal", 0),
        ),
    )
    mark_completeness(result, expected)
    return result
