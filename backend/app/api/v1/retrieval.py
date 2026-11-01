from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import VerifiedUser
from app.services.retrieval import RetrievalScope, RetrievalUnavailable, create_retriever
from app.services.retrieval.store import RetrievalStore
from app.services.retrieval.tools import (
    ElementArguments,
    ElementCall,
    PaperTools,
    SectionArguments,
    SectionCall,
)

router = APIRouter(tags=["retrieval"])


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1024)
    collection_ids: list[UUID] | None = Field(default=None, max_length=100)
    paper_ids: list[UUID] | None = Field(default=None, max_length=100)
    queries: list[str] | None = Field(default=None, min_length=1, max_length=4)
    section_hint: str | None = None
    modality_hint: str | None = None


@router.post("/retrieval/search")
async def search(body: SearchRequest, request: Request, user: VerifiedUser) -> dict:
    state = request.app.state
    scope = RetrievalScope(
        user.id,
        tuple(body.collection_ids) if body.collection_ids is not None else None,
        tuple(body.paper_ids) if body.paper_ids is not None else None,
    )
    retriever = create_retriever(state.settings, state.session_factory, state.engine, scope)
    try:
        result = await retriever.search(
            body.query,
            probes=body.queries,
            section_hint=body.section_hint,
            modality_hint=body.modality_hint,
        )
    except RetrievalUnavailable:
        raise HTTPException(503, "检索服务暂时不可用，请稍后重试") from None
    return {
        "operation_id": result.operation_id,
        "status": "degraded" if result.degraded else "ok",
        "degraded": result.degraded,
        "channel_counts": result.channel_counts,
        "plan": result.plan,
        "documents": [
            {
                "text": doc.page_content,
                **{
                    k: v for k, v in doc.metadata.items() if k not in {"user_id", "embedding_input"}
                },
            }
            for doc in result.documents
        ],
    }


class ScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_ids: list[UUID] | None = Field(default=None, max_length=100)
    paper_ids: list[UUID] | None = Field(default=None, max_length=100)


class ReadSectionRequest(ScopeRequest, SectionArguments):
    pass


class InspectRequest(ScopeRequest, ElementArguments):
    pass


def request_scope(body: ScopeRequest, user_id: UUID) -> RetrievalScope:
    return RetrievalScope(
        user_id,
        tuple(body.collection_ids) if body.collection_ids is not None else None,
        tuple(body.paper_ids) if body.paper_ids is not None else None,
    )


@router.post("/retrieval/paper-maps")
async def paper_maps(body: ScopeRequest, request: Request, user: VerifiedUser) -> dict:
    state = request.app.state
    return {
        "paper_maps": await RetrievalStore(state.session_factory, state.settings).paper_maps(
            request_scope(body, user.id)
        )
    }


async def execute_tool(
    body: ScopeRequest, call: SectionCall | ElementCall, request: Request, user_id: UUID
) -> dict:
    state = request.app.state
    scope = request_scope(body, user_id)
    retriever = create_retriever(state.settings, state.session_factory, state.engine, scope)
    result, report = await PaperTools(
        retriever.store, scope, retriever, state.settings.paper_tool_max_bytes
    ).execute(call)
    return {
        **report,
        "documents": [
            {
                "text": doc.page_content,
                **{
                    k: v for k, v in doc.metadata.items() if k not in {"user_id", "embedding_input"}
                },
            }
            for doc in result.documents
        ],
    }


@router.post("/retrieval/read-section")
async def read_paper_section(
    body: ReadSectionRequest, request: Request, user: VerifiedUser
) -> dict:
    args = SectionArguments.model_validate(body.model_dump(exclude={"paper_ids", "collection_ids"}))
    return await execute_tool(
        body, SectionCall(name="read_paper_section", args=args), request, user.id
    )


@router.post("/retrieval/inspect-element")
async def inspect_element(body: InspectRequest, request: Request, user: VerifiedUser) -> dict:
    args = ElementArguments.model_validate(body.model_dump(exclude={"paper_ids", "collection_ids"}))
    return await execute_tool(
        body, ElementCall(name="inspect_element", args=args), request, user.id
    )
