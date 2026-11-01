from langchain_core.documents import Document

from app.services.retrieval.context import assemble_context, mark_completeness, requires_neighbor


def source(key, ordinal, text, parent="p"):
    return Document(
        page_content=text,
        metadata={
            "chunk_id": key,
            "ordinal": ordinal,
            "parent_id": parent,
            "locations": [{"page_idx": ordinal, "bbox": None}],
        },
    )


def assemble(hits, siblings=None, neighbors=None, expected=None, **kwargs):
    return assemble_context(
        hits,
        siblings or {},
        neighbors or {},
        expected or {},
        budget=kwargs.get("budget", 10000),
        parent_budget=kwargs.get("parent_budget", 6000),
        max_sources=kwargs.get("max_sources", 24),
    )


def test_parent_members_are_returned_in_order_as_independently_citable_sources():
    a, b, c = [source(str(i), i, f"Original text {i}.") for i in range(3)]
    result = assemble([b], {"1": [b, a, c]}, expected={"p": 3})
    assert [d.metadata["chunk_id"] for d in result] == ["0", "1", "2"]
    assert [d.page_content for d in result] == [a.page_content, b.page_content, c.page_content]
    assert [d.metadata["retrieval_role"] for d in result] == ["parent", "hit", "parent"]
    assert all(d.metadata["parent_complete"] for d in result)
    assert [d.metadata["locations"][0]["page_idx"] for d in result] == [0, 1, 2]
    assert "retrieval_role" not in b.metadata


def test_expansion_preserves_ranked_hits_before_spending_budget_on_context():
    a, b = source("a", 0, "AAA", "first"), source("b", 10, "BBB", "second")
    extra = source("extra", 1, "XX", "first")
    result = assemble([a, b], {"a": [extra]}, budget=6)
    assert [d.metadata["chunk_id"] for d in result] == ["a", "b"]


def test_shared_parents_and_neighbor_windows_are_deduplicated():
    a, b, c = [source(str(i), i, f"Text {i}") for i in range(3)]
    result = assemble([b, a], {"1": [a, b, c], "0": [a, b, c]}, {"1": [c]}, {"p": 3})
    assert len(result) == 3 and len({d.metadata["chunk_id"] for d in result}) == 3
    ranked = sorted(
        (d for d in result if d.metadata["retrieval_role"] == "hit"),
        key=lambda d: d.metadata["hit_rank"],
    )
    assert [d.metadata["chunk_id"] for d in ranked] == ["1", "0"]
    assert (
        next(d for d in result if d.metadata["chunk_id"] == "0").metadata["retrieval_role"] == "hit"
    )


def test_global_parent_and_source_limits_do_not_slice_source_text():
    hit = source("a", 2, "中" * 10)
    extras = [source(str(i), i, "文" * 10) for i in (0, 1, 3, 4)]
    result = assemble(
        [hit], {"a": extras}, expected={"p": 5}, budget=95, parent_budget=65, max_sources=3
    )
    assert len(result) == 2 and sum(len(d.page_content.encode()) for d in result) <= 65
    assert all(len(d.page_content) == 10 and not d.metadata["parent_complete"] for d in result)
    assert len(assemble([hit], {"a": extras}, max_sources=1)) == 1


def test_other_parent_is_not_evicted_and_partial_status_is_recomputed_after_revocation():
    a, b = source("a", 0, "A"), source("b", 1, "B")
    result = assemble([a], {"a": [b]}, expected={"p": 2})
    assert result[0].metadata["parent_complete"]
    result.pop()
    mark_completeness(result, {"p": 2})
    assert not result[0].metadata["parent_complete"]


def test_legacy_chunks_keep_hit_order_and_zero_budget_returns_nothing():
    hits = [source("a", 10, "A", None), source("b", 0, "B", None)]
    assert [d.metadata["chunk_id"] for d in assemble(hits)] == ["a", "b"]
    assert assemble(hits, budget=0) == []


def test_definition_introductions_and_formula_hits_request_neighbors():
    assert requires_neighbor(source("a", 0, "Where x is the hidden dimension."))
    assert requires_neighbor(source("b", 0, "其中 x 为隐藏维度。"))
    formula = source("c", 0, "L = 0.")
    formula.metadata["kind"] = "formula"
    assert requires_neighbor(formula)
    assert not requires_neighbor(source("d", 0, "A self-contained scientific statement."))
