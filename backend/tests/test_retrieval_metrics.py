import math

import pytest

from app.eval.retrieval import detailed_metrics, metrics


def row(ranks, *, relevant=2, sources=4, context_relevant=2, latency=100):
    return {
        "relevant_ranks": ranks,
        "relevant_count": relevant,
        "returned": [{}, {}, {}] if ranks else [],
        "context_sources": sources,
        "context_relevant_count": context_relevant,
        "core_bytes": 100 if ranks else 0,
        "context_bytes": 200,
        "latency_ms": latency,
    }


def test_multicutoff_binary_ranking_metrics_against_known_positions():
    result = detailed_metrics([row([1, 3])])
    assert result["precision_at_1"] == 1
    assert result["recall_at_1"] == 0.5
    assert result["full_evidence_at_1"] == 0
    assert result["precision_at_3"] == pytest.approx(2 / 3)
    assert result["recall_at_3"] == 1
    assert result["map_at_3"] == pytest.approx(5 / 6)
    assert result["ndcg_at_3"] == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert result["f1_at_3"] == pytest.approx(0.8)
    assert result["full_evidence_at_3"] == 1
    assert result["precision_at_8"] == 0.25
    assert result["mrr_at_8"] == 1
    assert {key: result[key] for key in metrics([row([1, 3])])} == metrics([row([1, 3])])


def test_macro_averaging_context_rescue_and_latency_percentiles():
    rows = [row([1, 3]), row([], relevant=1, context_relevant=1, sources=2, latency=300)]
    result = detailed_metrics(rows)
    assert result["hit_at_8"] == 0.5
    assert result["map_at_8"] == pytest.approx(5 / 12)
    assert result["empty_core_rate"] == 0.5
    assert result["context_hit_rate"] == 1
    assert result["context_full_evidence_rate"] == 1
    assert result["context_precision"] == 0.5
    assert result["context_recall"] == 1
    assert result["context_additional_hit_rate"] == 0.5
    assert result["context_sources_mean"] == 3
    assert result["latency_ms_mean"] == result["latency_ms_p50"] == 200
    assert result["latency_ms_p95"] == 290
    assert result["latency_ms_max"] == 300


def test_partial_context_and_ap_denominator_with_more_labels_than_cutoff():
    result = detailed_metrics([row([2], relevant=10, context_relevant=2)])
    assert result["hit_at_1"] == 0
    assert result["mrr_at_3"] == 0.5
    assert result["map_at_3"] == pytest.approx(0.5 / 3)
    assert result["map_at_8"] == 0.5 / 8
    assert result["recall_at_8"] == 0.1
    assert result["context_recall"] == 0.2
    assert result["context_full_evidence_rate"] == 0
    empty = detailed_metrics([row([], sources=0, context_relevant=0)])
    assert empty["context_precision"] == empty["context_hit_rate"] == 0


@pytest.mark.parametrize("ranks,relevant", [([1, 1], 2), ([9], 1), ([0], 1), ([1, 2], 1)])
def test_invalid_label_positions_are_rejected(ranks, relevant):
    with pytest.raises(ValueError, match="Invalid evidence ranks"):
        detailed_metrics([row(ranks, relevant=relevant)])


def test_empty_evaluation_is_rejected():
    with pytest.raises(ValueError, match="empty dataset"):
        detailed_metrics([])
