"""Rerank evidence bundles atomically, then restore individually citable sources."""

from collections import Counter

from langchain_core.documents import Document

from app.services.retrieval.context import mark_completeness


def fits_context(unit: Document, *, budget: int, max_sources: int) -> bool:
    """A candidate must be deliverable even when selected before all other groups."""
    rows = unit.metadata.get("evidence", [{"text": unit.page_content, "metadata": unit.metadata}])
    sources = {row["metadata"]["chunk_id"]: row["text"] for row in rows}
    return len(sources) <= max_sources and sum(len(v.encode()) for v in sources.values()) <= budget


def evidence_unit(
    anchor: Document,
    members: list[Document],
    references: list[Document],
    *,
    merged: bool,
    expected: dict[str, int],
    incomplete: list[str],
) -> Document:
    sources: dict[str, Document] = {}
    for doc in members + references:
        key = doc.metadata["chunk_id"]
        if key in sources:
            continue
        current = doc.model_copy(deep=True)
        current.metadata.update(
            retrieval_role="cross_ref" if doc in references else "parent" if merged else "neighbor",
            context_anchor=anchor.metadata["chunk_id"],
            merged_parent=merged,
        )
        if key == anchor.metadata["chunk_id"] or doc.metadata.get("retrieval_role") == "hit":
            current.metadata["retrieval_role"] = "hit"
        sources[key] = current
    required = list(dict.fromkeys(d.metadata["chunk_id"] for d in references))
    unresolved = sorted(
        {
            edge["label"]
            for doc in members
            for edge in doc.metadata.get("chunk_metadata", {}).get("cross_refs", [])
            if not edge.get("target_ordinals")
        }
    )
    unit = anchor.model_copy(deep=True)
    unit.page_content = "\n\n".join(doc.page_content for doc in sources.values())
    unit.metadata.update(
        embedding_input=anchor.metadata.get("chunk_metadata", {}).get("context_prefix", "")
        + unit.page_content,
        evidence=[{"text": doc.page_content, "metadata": doc.metadata} for doc in sources.values()],
        expected_parents=expected,
        required_cross_ref_ids=required,
        unresolved_cross_refs=unresolved,
        expansion_incomplete=incomplete,
    )
    return unit


def select_evidence(
    units: list[Document], *, budget: int, max_sources: int
) -> tuple[list[Document], dict[str, int]]:
    """A bundle fits whole or is omitted, so truncation cannot orphan a table."""
    selected: dict[str, Document] = {}
    expected: dict[str, int] = {}
    for rank, unit in enumerate(units, 1):
        rows = unit.metadata.get(
            "evidence", [{"text": unit.page_content, "metadata": unit.metadata}]
        )
        additions = [row for row in rows if row["metadata"]["chunk_id"] not in selected]
        size = sum(len(row["text"].encode()) for row in additions)
        if size > budget or len(selected) + len(additions) > max_sources:
            continue
        budget -= size
        expected.update(unit.metadata.get("expected_parents", {}))
        for row in rows:
            previous = selected.get(row["metadata"]["chunk_id"])
            if previous is not None and not (
                row["metadata"].get("retrieval_role") == "hit"
                and previous.metadata.get("retrieval_role") != "hit"
            ):
                continue
            # A source can first arrive as a reference/window, then as a real
            # seed of an admitted group. Preserve that independent hit and its
            # actual group rank, rather than hiding it behind the earlier role.
            doc = Document(page_content=row["text"], metadata=dict(row["metadata"]))
            for field in (
                "rerank_score",
                "retrieval_intent",
                "required_cross_ref_ids",
                "unresolved_cross_refs",
                "expansion_incomplete",
            ):
                if field in unit.metadata:
                    doc.metadata[field] = unit.metadata[field]
            doc.metadata["hit_rank"] = rank
            selected[doc.metadata["chunk_id"]] = doc
    result = list(selected.values())
    for doc in result:
        doc.metadata["parent_expected_count"] = expected.get(doc.metadata.get("parent_id", ""), -1)
    mark_completeness(result, expected)
    return result, expected


def revoke_orphans(
    documents: list[Document], expected: dict[str, int] | None = None
) -> list[Document]:
    result = documents
    while result:
        available = {doc.metadata["chunk_id"] for doc in result}
        current = [
            doc
            for doc in result
            if set(doc.metadata.get("required_cross_ref_ids", [])) <= available
            and (
                not doc.metadata.get("context_anchor")
                or doc.metadata["context_anchor"] in available
            )
        ]
        if len(current) == len(result):
            break
        result = current
    if expected is None:
        expected = {
            doc.metadata["parent_id"]: doc.metadata.get("parent_expected_count", -1)
            for doc in result
            if doc.metadata.get("parent_id")
        }
    mark_completeness(result, expected)
    return result


def hit_counts(hits: list[Document]) -> Counter:
    return Counter(doc.metadata["parent_id"] for doc in hits if doc.metadata.get("parent_id"))
