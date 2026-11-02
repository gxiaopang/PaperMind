"""Offline recall ranking benchmark; filters/provenance have separate workflow tests."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.memory.history import estimate_tokens  # noqa: E402
from app.services.memory.store import select_memories  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "backend/tests/fixtures/memory/recall.json"
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/papermind-memory-eval.json"))
    args = parser.parse_args()
    dataset = json.loads(args.cases.read_text())
    rows = []
    hits = expected_count = false_count = selected_count = 0
    for case in dataset["cases"]:
        entries = [
            {"pinned": False, "status": "active", "updated_at": "2026-11-02", **item}
            for item in case["entries"]
        ]
        started = time.perf_counter()
        selected = select_memories(entries, case["question"], case["budget"])
        latency = round((time.perf_counter() - started) * 1000, 3)
        actual, expected = {v["id"] for v in selected}, set(case["expected"])
        tokens = sum(estimate_tokens(json.dumps(v, ensure_ascii=False)) + 8 for v in selected)
        hits += len(actual & expected)
        expected_count += len(expected)
        false_count += len(actual - expected)
        selected_count += len(actual)
        rows.append(
            {
                "id": case["id"],
                "passed": actual == expected
                and tokens <= case["budget"]
                and (not case.get("expected_first") or selected[0]["id"] == case["expected_first"]),
                "selected": sorted(actual),
                "expected": sorted(expected),
                "estimated_tokens": tokens,
                "budget": case["budget"],
                "latency_ms": latency,
            }
        )
    report = {
        "dataset": dataset["version"],
        "synthetic": True,
        "model_calls": 0,
        "scope": "ranking after authorization and evidence filtering; not a model accuracy evaluation",
        "passed": sum(v["passed"] for v in rows),
        "total": len(rows),
        "recall": hits / expected_count if expected_count else 1,
        "false_recall_rate": false_count / selected_count if selected_count else 0,
        "budget_violations": sum(v["estimated_tokens"] > v["budget"] for v in rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
