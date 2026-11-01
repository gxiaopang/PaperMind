"""Node-specific model inputs; authoritative source text is never truncated."""

import json

from app.graph.state import RAGState

CONTEXT_VERSION = "node-context-v1"
CONTROL_EVIDENCE_BYTES = 12000
SOURCE_PREVIEW_BYTES = 512


def serialize(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _prefix(value: str, budget: int) -> str:
    return value.encode()[:budget].decode("utf-8", errors="ignore")


def _sources(documents: list[dict]) -> list[dict]:
    result = []
    for number, doc in enumerate(documents, 1):
        meta = doc["metadata"]
        chunk = meta.get("chunk_metadata", {})
        result.append(
            {
                "number": number,
                "chunk_id": meta["chunk_id"],
                "paper_id": meta.get("paper_id"),
                "text": doc["text"],
                "title": meta.get("title", ""),
                "section": meta.get("section_path", []),
                "section_role": chunk.get("section_role"),
                "citations": chunk.get("citations", []),
                "source_provenance": chunk.get("source_provenance", []),
                "parent_id": meta.get("parent_id"),
                "parent_complete": meta.get("parent_complete"),
                "retrieval_role": meta.get("retrieval_role"),
                "unresolved_cross_refs": meta.get("unresolved_cross_refs", []),
            }
        )
    return result


def _inventory(documents: list[dict]) -> dict:
    """Bounded navigation previews, explicitly distinct from complete evidence."""
    entries: list[dict] = []
    for source in _sources(documents):
        entry = {
            key: source[key]
            for key in (
                "number",
                "chunk_id",
                "paper_id",
                "title",
                "section",
                "section_role",
                "parent_complete",
                "retrieval_role",
                "unresolved_cross_refs",
            )
        }
        entry["preview"] = _prefix(source["text"], SOURCE_PREVIEW_BYTES)
        candidate = {
            "preview_only": True,
            "sources": [*entries, entry],
            "omitted_sources": len(documents) - len(entries) - 1,
        }
        if len(serialize(candidate).encode()) <= CONTROL_EVIDENCE_BYTES:
            entries.append(entry)
    return {
        "preview_only": True,
        "sources": entries,
        "omitted_sources": len(documents) - len(entries),
    }


def _reports(state: RAGState, documents: list[dict]) -> list[dict]:
    available = {d["metadata"]["chunk_id"] for d in documents}
    reports = []
    for report in state.get("tool_reports", []):
        current = {
            key: report[key]
            for key in ("tool", "arguments", "status", "guidance", "required_bytes", "issues")
            if key in report
        }
        parents = []
        for parent in report.get("parents", []):
            if not set(parent["source_ids"]) <= available:
                continue
            parents.append(
                {
                    **{key: value for key, value in parent.items() if key != "text"},
                    "text_in_sources": True,
                }
            )
        if parents:
            current["parents"] = parents
        reports.append(current)
    return reports


def node_payload(
    node: str, state: RAGState, documents: list[dict] | None = None, *, max_search_probes: int
) -> dict:
    docs = documents or []
    result: dict = {"question": state["question"]}
    if node in {"route", "rewrite_query", "generate", "direct_answer"}:
        result.update(history=state.get("history", []), summary=state.get("summary", ""))
    if node == "rewrite_query":
        result.update(
            active_papers=state.get("active_paper_ids", []),
            papers=[
                {key: paper[key] for key in ("paper_id", "title") if key in paper}
                for paper in state.get("paper_maps", [])
            ],
        )
    if node in {
        "plan_paper_tools",
        "transform_query",
        "grade_documents",
        "generate",
        "verify_grounding",
    }:
        result["query"] = state.get("rewritten_query", state["question"])
        result["tool_reports"] = _reports(state, docs)
    if node in {"plan_paper_tools", "transform_query"}:
        result["evidence_inventory"] = _inventory(docs)
    if node == "plan_paper_tools":
        # Complete L0 is needed only for tool selection, not each model call.
        result.update(paper_maps=state.get("paper_maps", []), max_search_probes=max_search_probes)
    if node in {"grade_documents", "generate", "verify_grounding"}:
        result["sources"] = _sources(docs)
    if node in {"generate", "direct_answer"}:
        result["preferences"] = state.get("preferences") or {
            "answer_language": "zh",
            "detail_level": "standard",
        }
    if node == "generate" and state.get("verify_result"):
        result.update(answer=state.get("answer", ""), previous_validation=state["verify_result"])
    if node == "verify_grounding":
        result["answer"] = state.get("answer", "")
    return result
