import re
import time
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from app.db.models import ChatRun, Chunk, Citation, Conversation, Message, TaskOutbox, utcnow
from app.graph.context import node_payload
from app.graph.prompts import Grade, Grounding, Query, Route
from app.graph.state import RAGState
from app.graph.validation import SAFE_ANSWER, citation_issues, validate_citations
from app.services.chat.events import append, emit, mirror
from app.services.chat.store import assert_live, finish_task, owned_run, scope_for
from app.services.ingestion.quota import lock_user
from app.services.memory.history import history_tokens, load_memory
from app.services.memory.tools import curate_user_turn
from app.services.retrieval import create_retriever
from app.services.retrieval.store import RetrievalStore
from app.services.retrieval.tools import (
    PaperTools,
    ToolPlan,
    boundary_calls,
    element_followups,
    merge_tool_documents,
)


@dataclass
class GraphServices:
    settings: Any
    sessions: Any
    engine: Any
    redis: Any
    models: Any
    retriever_factory: Any = create_retriever

    async def run(self, state: RAGState) -> ChatRun:
        return await assert_live(self.sessions, UUID(state["user_id"]), UUID(state["run_id"]))

    async def event(self, state: RAGState, kind: str, data: dict) -> None:
        await emit(
            self, UUID(state["user_id"]), UUID(state["run_id"]), kind, data, state.get("attempt", 0)
        )

    async def step(self, state: RAGState, node: str) -> ChatRun:
        run = await self.run(state)
        await RetrievalStore(self.sessions, self.settings).validate_scope(
            scope_for(run.user_id, run.scope)
        )
        await self.event(state, "step", {"node": node})
        return run

    async def fresh(self, state: RAGState) -> list[dict]:
        run = await self.run(state)
        state.update(
            cast(
                RAGState,
                await load_memory(
                    self.sessions,
                    self.settings,
                    run.user_id,
                    run.conversation_id,
                    run.scope,
                    run.question,
                ),
            )
        )
        state["paper_maps"] = await RetrievalStore(self.sessions, self.settings).paper_maps(
            scope_for(run.user_id, run.scope)
        )
        documents = [
            Document(page_content=d["text"], metadata=d["metadata"])
            for d in state.get("documents", [])
        ]
        docs = await RetrievalStore(self.sessions, self.settings).refresh(
            scope_for(run.user_id, run.scope), documents
        )
        return [{"text": d.page_content, "metadata": d.metadata} for d in docs]

    def payload(self, node: str, state: RAGState, documents: list[dict] | None = None) -> dict:
        return node_payload(
            node, state, documents, max_search_probes=self.settings.retrieval_max_subqueries
        )

    async def load_memory(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "load_memory")
        memory = await load_memory(
            self.sessions, self.settings, run.user_id, run.conversation_id, run.scope, run.question
        )
        memory["paper_maps"] = await RetrievalStore(self.sessions, self.settings).paper_maps(
            scope_for(run.user_id, run.scope)
        )
        return memory

    async def route(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "route")
        result = await self.models.invoke(
            run, "route", self.payload("route", state), config, schema=Route
        )
        # A model cannot bypass retrieval by labelling an arbitrary research request
        # chitchat. Only a deliberately narrow greeting / product-help gate can.
        q = state["question"].strip().lower()
        greeting = bool(
            re.fullmatch(r"(你好|您好|嗨|谢谢|再见|hello|hi|thanks|thank you|bye)[！!。.?？\s]*", q)
        )
        product_help = bool(
            re.fullmatch(
                r"(?:papermind)?(?:怎么|如何)(?:上传(?:pdf|论文)?|登录|注册|选择论文|使用(?:papermind)?|创建知识库)[？?\s]*",
                q,
            )
        )
        route = (
            "chitchat" if result.route == "chitchat" and (greeting or product_help) else "paper_qa"
        )
        return {"route": route}

    async def rewrite_query(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "rewrite_query")
        result = await self.models.invoke(
            run, "rewrite_query", self.payload("rewrite_query", state), config, schema=Query
        )
        query = result.query.strip()
        if not query or len(query.encode()) > self.settings.retrieval_query_bytes:
            query = state["question"]
        await self.event(state, "rewrite", {"query": query})
        return {"rewritten_query": query}

    async def claim(self, state: RAGState, field: str, limit: int) -> int | None:
        async with self.sessions() as db:
            run = await owned_run(db, UUID(state["user_id"]), UUID(state["run_id"]), lock=True)
            value = getattr(run, field)
            if value >= limit:
                return None
            setattr(run, field, value + 1)
            await db.commit()
            return value + 1

    async def retrieve(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "retrieve")
        limit = 1 + (run.config["retrieval_retries"] if run.mode == "agentic" else 0)
        count = await self.claim(state, "retrieval_count", limit)
        if count is None:
            return {"documents": [], "retrieval_retries": max(0, limit - 1), "sufficient": False}
        scope = scope_for(run.user_id, run.scope)
        if scope.paper_ids == () or scope.collection_ids == ():
            return {"documents": [], "tool_reports": [], "retrieval_retries": count - 1}
        retriever = self.retriever_factory(
            self.settings,
            self.sessions,
            self.engine,
            scope_for(run.user_id, run.scope),
            run_id=run.id,
        )
        previous = await self.fresh(state)
        plan = await self.models.invoke(
            run,
            "plan_paper_tools",
            self.payload("plan_paper_tools", state, previous),
            config,
            schema=ToolPlan,
        )
        calls = boundary_calls(state["question"], plan.calls[:4])
        toolset = PaperTools(
            retriever.store,
            scope_for(run.user_id, run.scope),
            retriever,
            self.settings.paper_tool_max_bytes,
        )
        # Each search internally fans out Dense/Sparse probes. Independent tools
        # also execute together, retaining their existing quota/provider slots.
        import asyncio

        results = await asyncio.gather(*(toolset.execute(call) for call in calls))
        followups = element_followups([d for result, _ in results for d in result.documents], calls)
        if followups:
            results.extend(await asyncio.gather(*(toolset.execute(call) for call in followups)))
        documents, reports, degraded = merge_tool_documents(
            results, self.settings.paper_tool_max_bytes
        )
        if plan.rejected_calls:
            degraded.append("tool_arguments_rejected")
            reports.extend(
                {"tool": call["name"], "status": "invalid_arguments", "issues": call["issues"]}
                for call in plan.rejected_calls
            )
        docs: list[dict] = [{"text": d.page_content, "metadata": d.metadata} for d in documents]
        await self.event(
            state,
            "step",
            {
                "node": "paper_tools",
                "tools": [r["tool"] for r in reports],
                "statuses": [r["status"] for r in reports],
            },
        )
        await self.event(
            state,
            "sources",
            {
                "references": [
                    {"number": i, "chunk_id": d["metadata"]["chunk_id"]}
                    for i, d in enumerate(docs, 1)
                ],
                "retrieval_count": count,
                "degraded": degraded,
            },
        )
        return {
            "documents": docs,
            "retrieval_retries": count - 1,
            "degraded": sorted(set(state.get("degraded", []) + degraded)),
            "tool_reports": reports,
            "paper_maps": state.get("paper_maps", []),
        }

    async def grade_documents(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "grade_documents")
        docs = await self.fresh(state)
        if not docs:
            return {"sufficient": False, "documents": []}
        result = await self.models.invoke(
            run,
            "grade_documents",
            self.payload("grade_documents", state, docs),
            config,
            schema=Grade,
        )
        return {"sufficient": result.sufficient, "documents": docs}

    async def transform_query(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "transform_query")
        docs = await self.fresh(state)
        result = await self.models.invoke(
            run,
            "transform_query",
            self.payload("transform_query", state, docs),
            config,
            schema=Query,
        )
        query = result.query.strip()
        if not query or len(query.encode()) > self.settings.retrieval_query_bytes:
            query = state["question"]
        await self.event(
            state, "rewrite", {"query": query, "retrieval_retry": state["retrieval_retries"] + 1}
        )
        return {"rewritten_query": query, "documents": docs}

    async def generate(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "generate")
        docs = await self.fresh(state)
        if not docs:
            return {
                "answer": "",
                "documents": [],
                "valid": False,
                "generation_retries": run.config["generation_retries"],
            }
        count = await self.claim(state, "generation_count", 1 + run.config["generation_retries"])
        if count is None:
            return {
                "answer": "",
                "documents": docs,
                "valid": False,
                "generation_retries": run.config["generation_retries"],
            }
        current: RAGState = {**state, "attempt": count}
        # Always reset the client's draft at the start, including a resumed attempt.
        await self.event(current, "regenerate", {"draft": True})
        await self.event(
            current,
            "sources",
            {
                "references": [
                    {"number": i, "chunk_id": d["metadata"]["chunk_id"]}
                    for i, d in enumerate(docs, 1)
                ]
            },
        )
        buffer, last = "", time.monotonic()

        async def token(text: str) -> None:
            nonlocal buffer, last
            buffer += text
            if len(buffer) >= 80 or time.monotonic() - last >= 0.15:
                await self.event(current, "token", {"text": buffer, "draft": True})
                buffer, last = "", time.monotonic()

        answer = await self.models.invoke(
            run, "generate", self.payload("generate", state, docs), config, on_token=token
        )
        if buffer:
            await self.event(current, "token", {"text": buffer, "draft": True})
        return {
            "answer": answer,
            "documents": docs,
            "attempt": count,
            "generation_retries": count - 1,
        }

    async def validate_citations(self, state: RAGState, config: RunnableConfig) -> dict:
        await self.step(state, "validate_citations")
        fresh = await self.fresh(state)
        same = [d["metadata"]["chunk_id"] for d in fresh] == [
            d["metadata"]["chunk_id"] for d in state["documents"]
        ]
        valid, citations = validate_citations(state["answer"], len(fresh))
        result = {
            "citations_valid": valid and same,
            "citation_issues": citation_issues(state["answer"], len(fresh)),
            "sources_revoked": not same,
            "grounding": "not_run" if state["mode"] == "fast" else "pending",
        }
        await self.event(state, "verify", result)
        # Do not renumber documents between generation and validation.
        return {"valid": valid and same, "citations": citations, "verify_result": result}

    async def verify_grounding(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "verify_grounding")
        fresh = await self.fresh(state)
        same = [d["metadata"]["chunk_id"] for d in fresh] == [
            d["metadata"]["chunk_id"] for d in state["documents"]
        ]
        grounded = False
        if same and fresh:
            result = await self.models.invoke(
                run,
                "verify_grounding",
                self.payload("verify_grounding", state, fresh),
                config,
                schema=Grounding,
            )
            grounded = result.grounded
        verification = {**state["verify_result"], "grounding": "passed" if grounded else "failed"}
        await self.event(state, "verify", verification)
        return {"grounded": grounded, "verify_result": verification}

    async def direct_answer(self, state: RAGState, config: RunnableConfig) -> dict:
        run = await self.step(state, "direct_answer")
        answer = await self.models.invoke(
            run, "direct_answer", self.payload("direct_answer", state), config
        )
        # Product/greeting replies have no source identifiers.
        answer = re.sub(r"\[\d+\]", "", answer)
        return {
            "answer": answer,
            "citations": [],
            "result_status": "chitchat",
            "verify_result": {"citations_valid": True, "grounding": "not_applicable"},
        }

    async def insufficient_answer(self, state: RAGState, config: RunnableConfig) -> dict:
        await self.step(state, "insufficient_answer")
        return {
            "answer": SAFE_ANSWER,
            "citations": [],
            "result_status": "insufficient",
            "verify_result": {"citations_valid": True, "grounding": "insufficient"},
        }

    async def safe_answer(self, state: RAGState, config: RunnableConfig) -> dict:
        await self.step(state, "safe_answer")
        return {
            "answer": SAFE_ANSWER,
            "citations": [],
            "result_status": "safe_fallback",
            "verify_result": {
                **state.get("verify_result", {}),
                "final_citations_valid": True,
                "safe_fallback": True,
            },
        }

    async def persist(self, state: RAGState, config: RunnableConfig) -> dict:
        await self.step(state, "persist")
        from app.services.chat.results import usage_for

        async with self.sessions() as db:
            await lock_user(db, UUID(state["user_id"]))
            run = await owned_run(db, UUID(state["user_id"]), UUID(state["run_id"]), lock=True)
            existing = await db.scalar(
                select(Message).where(
                    Message.user_id == run.user_id,
                    Message.run_id == run.id,
                    Message.role == "assistant",
                )
            )
            if existing:
                return {"message_id": str(existing.id)}
            conversation = await db.scalar(
                select(Conversation)
                .where(Conversation.id == run.conversation_id, Conversation.user_id == run.user_id)
                .with_for_update()
            )
            assert conversation is not None
            answer, numbers, status = (
                state["answer"],
                state.get("citations", []),
                state["result_status"],
            )
            verification = state.get("verify_result", {})
            selected = [state["documents"][n - 1] for n in numbers]
            required = {
                key for d in selected for key in d["metadata"].get("required_cross_ref_ids", [])
            }
            checked = selected + [
                d
                for d in state["documents"]
                if d["metadata"]["chunk_id"] in required and d not in selected
            ]
            ids = [UUID(d["metadata"]["chunk_id"]) for d in checked]
            store = RetrievalStore(self.sessions, self.settings)
            current = (
                (
                    await db.execute(
                        store.candidates(scope_for(run.user_id, run.scope), searchable=False).where(
                            Chunk.id.in_(ids)
                        )
                    )
                ).all()
                if ids
                else []
            )
            available = {str(c.id): c for c, _ in current}
            if not required <= available.keys() or any(
                d["metadata"]["chunk_id"] not in available
                or available[d["metadata"]["chunk_id"]].text != d["text"]
                for d in checked
            ):
                answer, numbers, status = SAFE_ANSWER, [], "safe_fallback"
                verification = {
                    "final_citations_valid": True,
                    "sources_revoked": True,
                    "safe_fallback": True,
                }
            user_message = Message(
                user_id=run.user_id,
                conversation_id=run.conversation_id,
                run_id=run.id,
                seq=conversation.next_seq,
                role="user",
                content=run.question,
            )
            used_memories = [
                {
                    "id": str(m["id"]),
                    "kind": m.get("kind", "note"),
                    "level": m.get("level", "project"),
                    "content": str(m.get("content", ""))[:160],
                    "status": m.get("status", "active"),
                }
                for m in state.get("memories", [])
                if not str(m.get("id", "")).startswith("project:")
            ]
            message = Message(
                id=uuid4(),
                user_id=run.user_id,
                conversation_id=run.conversation_id,
                run_id=run.id,
                seq=conversation.next_seq + 1,
                role="assistant",
                content=answer,
                rewritten_query=state.get("rewritten_query"),
                route=state["route"],
                verify_result={
                    **verification,
                    "degraded": sorted(set(state.get("degraded", []))),
                    **({"used_memories": used_memories} if used_memories else {}),
                },
            )
            db.add_all([user_message, message])
            await db.flush()
            memory_events: list[dict] = []
            if state["route"] == "paper_qa":
                memory_events = await curate_user_turn(db, conversation, user_message, run.id)
                if memory_events:
                    message.verify_result = {
                        **message.verify_result,
                        "memory_events": memory_events,
                    }
            for n in numbers:
                chunk = available[state["documents"][n - 1]["metadata"]["chunk_id"]]
                db.add(
                    Citation(
                        user_id=run.user_id,
                        message_id=message.id,
                        number=n,
                        paper_id=chunk.paper_id,
                        version_id=chunk.version_id,
                        chunk_id=chunk.id,
                        source_text=chunk.text,
                        locations=chunk.locations,
                    )
                )
            conversation.next_seq += 2
            conversation.scope = run.scope
            cited_paper_ids = list(
                dict.fromkeys(
                    str(available[state["documents"][n - 1]["metadata"]["chunk_id"]].paper_id)
                    for n in numbers
                )
            )
            if cited_paper_ids:
                conversation.active_paper_ids = cited_paper_ids
            pending_history = list(
                await db.scalars(
                    select(Message)
                    .where(
                        Message.user_id == run.user_id,
                        Message.conversation_id == run.conversation_id,
                        Message.seq > conversation.summarized_seq,
                    )
                    .order_by(Message.seq)
                )
            )
            memory_pressure = (
                history_tokens([{"content": m.content} for m in pending_history])
                > self.settings.memory_recent_token_budget
            )
            if conversation.title == "新会话":
                conversation.title = run.question[:120]
            if (
                message.seq - conversation.summarized_seq
            ) // 2 >= self.settings.memory_summarize_after_turns or memory_pressure:
                # Every threshold-triggering run has its own idempotent summary task.
                db.add(TaskOutbox(user_id=run.user_id, kind="summarize", resource_id=run.id))
            run.status, run.result_status, run.finished_at = "completed", status, utcnow()
            await finish_task(db, run)
            usage = await usage_for(db, run.user_id, run.id)
            events = [
                append(
                    db,
                    run,
                    "answer",
                    {"message_id": str(message.id), "text": answer, "status": status},
                    state.get("attempt", 0),
                ),
                append(
                    db, run, "citations", {"message_id": str(message.id)}, state.get("attempt", 0)
                ),
                append(
                    db,
                    run,
                    "done",
                    {
                        "message_id": str(message.id),
                        "status": status,
                        "usage": usage,
                        **({"used_memories": used_memories} if used_memories else {}),
                        **({"memory_events": memory_events} if memory_events else {}),
                    },
                    state.get("attempt", 0),
                ),
            ]
            await db.commit()
        for event in events:
            await mirror(self.redis, self.settings, event)
        return {
            "message_id": str(message.id),
            "answer": answer,
            "citations": numbers,
            "result_status": status,
            "verify_result": verification,
        }
