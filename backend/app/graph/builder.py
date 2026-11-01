from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import GraphServices
from app.graph.state import RAGState


def build_graph(services: GraphServices, mode: str, checkpointer: Any) -> Any:
    graph = StateGraph(RAGState)
    common = [
        "load_memory",
        "route",
        "rewrite_query",
        "retrieve",
        "generate",
        "validate_citations",
        "direct_answer",
        "insufficient_answer",
        "safe_answer",
        "persist",
    ]
    for name in common + (
        ["grade_documents", "transform_query", "verify_grounding"] if mode == "agentic" else []
    ):
        graph.add_node(name, getattr(services, name))
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "route")
    graph.add_conditional_edges(
        "route", lambda s: "direct_answer" if s["route"] == "chitchat" else "rewrite_query"
    )
    graph.add_edge("rewrite_query", "retrieve")
    if mode == "agentic":
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            lambda s: (
                "generate"
                if s["sufficient"]
                else (
                    "transform_query"
                    if s["retrieval_retries"] < services.settings.rag_max_retrieval_retries
                    else "insufficient_answer"
                )
            ),
        )
        graph.add_edge("transform_query", "retrieve")
    else:
        graph.add_conditional_edges(
            "retrieve", lambda s: "generate" if s["documents"] else "insufficient_answer"
        )
    graph.add_edge("generate", "validate_citations")

    def after_validation(state: RAGState) -> str:
        if state["valid"]:
            return "verify_grounding" if mode == "agentic" else "persist"
        return (
            "generate"
            if state["generation_retries"] < services.settings.rag_max_generation_retries
            else "safe_answer"
        )

    graph.add_conditional_edges("validate_citations", after_validation)
    if mode == "agentic":
        graph.add_conditional_edges(
            "verify_grounding",
            lambda s: (
                "persist"
                if s["grounded"]
                else (
                    "generate"
                    if s["generation_retries"] < services.settings.rag_max_generation_retries
                    else "safe_answer"
                )
            ),
        )
    for name in ("direct_answer", "insufficient_answer", "safe_answer"):
        graph.add_edge(name, "persist")
    graph.add_edge("persist", END)
    return graph.compile(checkpointer=checkpointer, name=f"papermind_{mode}")
