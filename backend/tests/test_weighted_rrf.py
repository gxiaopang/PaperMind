import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.core.config import Settings
from app.services.chat.store import execution_config
from app.services.retrieval.hybrid import fuse


def sources():
    a = Document(page_content="BM25 first", metadata={"chunk_id": "a"})
    z = Document(page_content="Vector first", metadata={"chunk_id": "z"})
    return {"vector": [z, a], "bm25": [a, z]}


def test_equal_weight_scaling_preserves_original_order_and_does_not_mutate_sources():
    channels = sources()
    original = fuse(channels, 60)
    scaled = fuse(channels, 60, {"vector": 0.5, "bm25": 0.5})
    assert [d.metadata["chunk_id"] for d in original] == ["a", "z"]
    assert [d.metadata["chunk_id"] for d in scaled] == ["a", "z"]
    for old, new in zip(original, scaled, strict=True):
        assert new.metadata["rrf_score"] == pytest.approx(old.metadata["rrf_score"] / 2)
    assert all("rrf_score" not in d.metadata for rows in channels.values() for d in rows)


@pytest.mark.parametrize("vector_weight", [0.7, 0.8, 0.9])
def test_vector_weight_changes_fused_order_using_channel_ranks(vector_weight):
    result = fuse(sources(), 60, {"vector": vector_weight, "bm25": 1 - vector_weight})
    assert [d.metadata["chunk_id"] for d in result] == ["z", "a"]
    assert result[0].metadata["rrf_score"] == pytest.approx(
        vector_weight / 61 + (1 - vector_weight) / 62
    )
    assert result[0].metadata["ranks"] == {"vector": 1, "bm25": 2}
    reversed_weights = fuse(sources(), 60, {"vector": 1 - vector_weight, "bm25": vector_weight})
    assert reversed_weights[0].metadata["chunk_id"] == "a"


def test_single_available_channel_keeps_its_ranking():
    channels = sources()
    result = fuse({"bm25": channels["bm25"]}, 60, {"vector": 0.9, "bm25": 0.1})
    assert [d.metadata["chunk_id"] for d in result] == ["a", "z"]
    assert result[0].metadata["rrf_score"] == pytest.approx(0.1 / 61)


@pytest.mark.parametrize("weight", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_weights_fail_before_retrieval(weight):
    with pytest.raises(ValueError, match="weights"):
        fuse(sources(), 60, {"vector": weight, "bm25": 0.5})
    for field in ("rrf_vector_weight", "rrf_bm25_weight"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, tracing_enabled=False, **{field: weight})


def test_weight_changes_are_bound_to_chat_execution_snapshot():
    settings = Settings(_env_file=None, tracing_enabled=False)
    settings.rrf_vector_weight = settings.rrf_bm25_weight = 1
    original = execution_config(settings)
    settings.rrf_vector_weight, settings.rrf_bm25_weight = 0.8, 0.2
    changed = execution_config(settings)
    assert changed["rrf_vector_weight"] == 0.8 and changed["rrf_bm25_weight"] == 0.2
    assert {k for k in changed if changed[k] != original[k]} == {
        "rrf_vector_weight",
        "rrf_bm25_weight",
    }
