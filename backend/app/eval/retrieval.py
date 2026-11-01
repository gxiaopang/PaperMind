import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def load_dataset(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.with_suffix(".sha256").read_text().split()[0] != digest:
        raise ValueError("Frozen dataset checksum mismatch")
    dataset = json.loads(path.read_text())
    for entry in dataset.get("supplemental_review", {}).values():
        if hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Supplemental review checksum mismatch")
    corpus = {}
    for name, entry in dataset["corpus"].items():
        content = (ROOT / entry["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen corpus checksum mismatch")
        corpus[name] = json.loads(content)
        for row in corpus[name]["chunks"]:
            if hashlib.sha256(row["embedding_input"].encode()).hexdigest() != row["content_hash"]:
                raise ValueError("Chunk identity mismatch")
    validate_dataset(dataset, corpus)
    return dataset, corpus


def validate_dataset(dataset: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> None:
    """Validate a draft in memory too, before publishing review artifacts."""
    for paper in corpus.values():
        if paper.get("paper_map"):
            section_ids = {section["id"] for section in paper["sections"]}

            def check_nodes(nodes: list[dict], valid_ids: set[str]) -> None:
                for node in nodes:
                    if node["section_id"] not in valid_ids:
                        raise ValueError("Paper map references an unknown section")
                    check_nodes(node["children"], valid_ids)

            check_nodes(paper["paper_map"]["toc"], section_ids)
            if any(
                e.get("section_id") is not None and e["section_id"] not in section_ids
                for e in paper["paper_map"]["elements"]
            ):
                raise ValueError("Paper map element references an unknown section")
        for ordinal, row in enumerate(paper["chunks"]):
            if row["ordinal"] != ordinal:
                raise ValueError("Corpus ordinal mismatch")
            for edge in row.get("chunk_metadata", {}).get("cross_refs", []):
                if not isinstance(edge.get("label"), str) or any(
                    type(target) is not int or not 0 <= target < len(paper["chunks"])
                    for target in edge.get("target_ordinals", [])
                ):
                    raise ValueError("Invalid version-local cross-reference target")
    ids: set[str] = set()
    for case in dataset["cases"] + dataset.get("answerability_cases", []):
        if case["id"] in ids:
            raise ValueError("Duplicate evaluation case ID")
        ids.add(case["id"])
        if case["scope"] != "all" and case["scope"] not in corpus:
            raise ValueError("Unknown evaluation scope")
        if case.get("papers") and (
            len(set(case["papers"])) != len(case["papers"])
            or not set(case["papers"]) <= corpus.keys()
        ):
            raise ValueError("Invalid explicit paper scope")
        if not case["standard_answer"].strip():
            raise ValueError("Missing reference answer")
        if case in dataset.get("answerability_cases", []):
            if (
                case["relevant"]
                or case.get("expected_retrieval", {}).get("answerability") != "insufficient"
            ):
                raise ValueError("Invalid answerability label")
            continue
        if not case["relevant"]:
            raise ValueError("Missing evidence label")
        for reference in case["relevant"]:
            chunk = corpus[reference["paper"]]["chunks"][reference["ordinal"]]
            if (
                not chunk["indexable"] and dataset.get("evidence_policy") != "source_backed_context"
            ) or chunk["content_hash"] != reference["content_hash"]:
                raise ValueError("Evidence label differs from frozen corpus")
            if (
                case["scope"] != "all"
                and reference["paper"] != case["scope"]
                or case.get("papers")
                and reference["paper"] not in case["papers"]
            ):
                raise ValueError("Evidence outside evaluation scope")
            if sorted({loc["page_idx"] for loc in chunk["locations"]}) != reference["pages"]:
                raise ValueError("Evidence page mismatch")


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")
    return {
        "hit_at_8": sum(bool(r["relevant_ranks"]) for r in rows) / len(rows),
        "mrr_at_8": sum(1 / min(r["relevant_ranks"]) if r["relevant_ranks"] else 0 for r in rows)
        / len(rows),
        "precision_at_8": sum(len(r["relevant_ranks"]) / 8 for r in rows) / len(rows),
        "recall_at_8": sum(len(r["relevant_ranks"]) / r["relevant_count"] for r in rows)
        / len(rows),
    }


def detailed_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Macro binary-label metrics on delivered core hits; context is unordered."""
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")
    for row in rows:
        ranks = row["relevant_ranks"]
        if (
            row["relevant_count"] < 1
            or len(ranks) != len(set(ranks))
            or len(ranks) > row["relevant_count"]
            or any(type(rank) is not int or not 1 <= rank <= 8 for rank in ranks)
        ):
            raise ValueError("Invalid evidence ranks")

    def mean(values: Iterable[float]) -> float:
        return sum(values) / len(rows)

    result = {}
    for k in (1, 3, 5, 8):
        selected = [[rank for rank in r["relevant_ranks"] if rank <= k] for r in rows]
        result.update(
            {
                f"hit_at_{k}": mean(bool(s) for s in selected),
                f"mrr_at_{k}": mean(1 / min(s) if s else 0 for s in selected),
                f"precision_at_{k}": mean(len(s) / k for s in selected),
                f"recall_at_{k}": mean(
                    len(s) / r["relevant_count"] for s, r in zip(selected, rows, strict=True)
                ),
                f"ndcg_at_{k}": mean(
                    sum(1 / math.log2(rank + 1) for rank in s)
                    / sum(1 / math.log2(i + 1) for i in range(1, min(k, r["relevant_count"]) + 1))
                    for s, r in zip(selected, rows, strict=True)
                ),
                f"map_at_{k}": mean(
                    sum(i / rank for i, rank in enumerate(sorted(s), 1))
                    / min(k, r["relevant_count"])
                    for s, r in zip(selected, rows, strict=True)
                ),
                f"f1_at_{k}": mean(
                    2 * len(s) / (k + r["relevant_count"])
                    for s, r in zip(selected, rows, strict=True)
                ),
                f"full_evidence_at_{k}": mean(
                    len(s) == r["relevant_count"] for s, r in zip(selected, rows, strict=True)
                ),
            }
        )
    result["empty_core_rate"] = mean(not r["returned"] for r in rows)
    result["core_sources_mean"] = mean(len(r["returned"]) for r in rows)
    result["context_hit_rate"] = mean(r["context_relevant_count"] > 0 for r in rows)
    result["context_full_evidence_rate"] = mean(
        r["context_relevant_count"] == r["relevant_count"] for r in rows
    )
    result["context_precision"] = mean(
        r["context_relevant_count"] / r["context_sources"] if r["context_sources"] else 0
        for r in rows
    )
    result["context_recall"] = mean(r["context_relevant_count"] / r["relevant_count"] for r in rows)
    result["context_sources_mean"] = mean(r["context_sources"] for r in rows)
    result["context_additional_hit_rate"] = mean(
        not r["relevant_ranks"] and r["context_relevant_count"] > 0 for r in rows
    )
    result["core_bytes_mean"] = mean(r["core_bytes"] for r in rows)
    result["context_bytes_mean"] = mean(r["context_bytes"] for r in rows)
    result["latency_ms_mean"] = mean(r["latency_ms"] for r in rows)
    result["latency_ms_max"] = max(r["latency_ms"] for r in rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    for label, q in (("p50", 0.5), ("p95", 0.95)):
        index = q * (len(latencies) - 1)
        left, right = math.floor(index), math.ceil(index)
        result[f"latency_ms_{label}"] = latencies[left] + (index - left) * (
            latencies[right] - latencies[left]
        )
    return result
