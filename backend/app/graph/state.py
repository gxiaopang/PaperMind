from typing import TypedDict


class RAGState(TypedDict, total=False):
    user_id: str
    conversation_id: str
    run_id: str
    question: str
    scope: dict
    mode: str
    history: list[dict]
    summary: str
    active_paper_ids: list[str]
    route: str
    rewritten_query: str
    documents: list[dict]
    retrieval_retries: int
    generation_retries: int
    attempt: int
    answer: str
    citations: list[int]
    verify_result: dict
    sufficient: bool
    valid: bool
    grounded: bool
    result_status: str
    degraded: list[str]
    message_id: str
    preferences: dict
    paper_maps: list[dict]
    tool_reports: list[dict]


def initial_state(run: object) -> RAGState:
    from app.db.models import ChatRun

    assert isinstance(run, ChatRun)
    return RAGState(
        user_id=str(run.user_id),
        conversation_id=str(run.conversation_id),
        run_id=str(run.id),
        question=run.question,
        scope=run.scope,
        mode=run.mode,
        preferences=run.preferences or {"answer_language": "zh", "detail_level": "standard"},
        history=[],
        summary="",
        active_paper_ids=[],
        route="paper_qa",
        rewritten_query=run.question,
        documents=[],
        retrieval_retries=0,
        generation_retries=0,
        attempt=0,
        answer="",
        citations=[],
        verify_result={},
        sufficient=False,
        valid=False,
        grounded=False,
        result_status="answered",
        degraded=[],
        message_id="",
        paper_maps=[],
        tool_reports=[],
    )
