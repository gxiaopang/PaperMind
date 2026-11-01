"""Three model-callable paper tools, bound to a server-owned retrieval scope."""

import re
from typing import Annotated, Any, Literal
from uuid import UUID

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.services.chunking.crossrefs import labels
from app.services.retrieval.types import RetrievalResult, RetrievalScope


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: UUID | None = None


class SearchArguments(Arguments):
    queries: list[str] = Field(min_length=1, max_length=4)
    section_hint: str | None = None
    modality_hint: Literal["text", "table", "formula", "figure"] | None = None


class SectionArguments(Arguments):
    section_std: str = Field(min_length=1, max_length=200)
    section_id: UUID | None = None
    mode: Literal["full", "summary"] = "full"


class ElementArguments(Arguments):
    label: str = Field(min_length=1, max_length=100)


class SearchCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["search_paper_chunks"]
    args: SearchArguments


class SectionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["read_paper_section"]
    args: SectionArguments


class ElementCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["inspect_element"]
    args: ElementArguments


PaperCall = Annotated[SearchCall | SectionCall | ElementCall, Field(discriminator="name")]


class ToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calls: list[PaperCall] = Field(min_length=1, max_length=6)
    rejected_calls: list[dict] = Field(default_factory=list, max_length=6)


def validate_tool_calls(calls: list[dict]) -> dict:
    """Reject malformed calls individually; never repair or widen their arguments."""
    if len(calls) > 6:
        return {"calls": calls}  # Keep the existing whole-plan bound.
    accepted, rejected = [], []
    adapter: TypeAdapter[SearchCall | SectionCall | ElementCall] = TypeAdapter(PaperCall)  # type: ignore[arg-type]
    for call in calls:
        try:
            accepted.append(adapter.validate_python(call).model_dump(mode="json"))
        except ValidationError as exc:
            rejected.append(
                {
                    "name": call.get("name", "unknown"),
                    "issues": [
                        {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
                        for e in exc.errors(include_input=False, include_url=False)
                    ],
                }
            )
    # An empty accepted list still fails ToolPlan's minimum-size validation and
    # follows the existing bounded retry; it never becomes an empty success.
    return {"calls": accepted, "rejected_calls": rejected}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema.model_json_schema(),
        },
    }
    for name, description, schema in (
        (
            "search_paper_chunks",
            "主力混合检索：queries 为可并行的多探针；Dense+Sparse、动态融合、章节/模态软加权、父子回溯与精排。隐性缺点、计算开销、失败案例应分探针查询。",
            SearchArguments,
        ),
        (
            "read_paper_section",
            "按 section_std（method、limitations、experiments、introduction、related_work、ablation、discussion、conclusion、appendix 等）或地图中的 section_id 直接读取章节摘要或完整 L1；跳过向量检索。提出 IDEA 前读取 limitations 和 method。",
            SectionArguments,
        ),
        (
            "inspect_element",
            "按 Table 2、Figure 3、Equation 4 等精确标签读取完整元素和正文引用分析；不会模糊匹配其他编号。",
            ElementArguments,
        ),
    )
]


def boundary_calls(question: str, calls: list[PaperCall]) -> list[PaperCall]:
    """Require known method/limitation boundaries before proposing research ideas."""
    result = list(calls)
    if re.search(
        r"\bideas?\b|research proposal|novel direction|创新方向|改进方案|研究思路|新思路|提.*创意",
        question,
        re.I,
    ):
        for standard in ("limitations", "method"):
            if not any(
                isinstance(c, SectionCall)
                and c.args.section_std == standard
                and c.args.mode == "full"
                and c.args.section_id is None
                and c.args.paper_id is None
                for c in result
            ):
                result.append(
                    SectionCall(
                        name="read_paper_section", args=SectionArguments(section_std=standard)
                    )
                )
    return result


def element_followups(
    documents: list[Document], calls: list[PaperCall], limit: int = 4
) -> list[ElementCall]:
    """Follow explicit table/figure/equation references once before answering."""
    available = {
        (d.metadata["paper_id"], label)
        for d in documents
        for label in d.metadata.get("chunk_metadata", {}).get("defined_labels", [])
    }
    requested = {
        (str(c.args.paper_id), label)
        for c in calls
        if isinstance(c, ElementCall)
        for label in labels(c.args.label)
    }
    followups = []
    for doc in documents:
        paper = doc.metadata["paper_id"]
        for edge in doc.metadata.get("chunk_metadata", {}).get("cross_refs", []):
            label = edge["label"]
            if (
                label.startswith(("table:", "figure:", "equation:"))
                and (paper, label) not in available | requested
            ):
                family, number = label.split(":", 1)
                followups.append(
                    ElementCall(
                        name="inspect_element",
                        args=ElementArguments(label=f"{family} {number}", paper_id=UUID(paper)),
                    )
                )
                requested.add((paper, label))
                if len(followups) >= limit:
                    return followups
    return followups


class PaperTools:
    def __init__(self, store: Any, scope: RetrievalScope, retriever: Any, max_bytes: int = 256000):
        self.store, self.scope, self.retriever, self.max_bytes = store, scope, retriever, max_bytes

    async def execute(self, call: PaperCall) -> tuple[RetrievalResult, dict]:
        scope = self.scope
        if call.args.paper_id is not None:
            if scope.paper_ids is not None and call.args.paper_id not in scope.paper_ids:
                from fastapi import HTTPException

                raise HTTPException(404, "工具请求的论文不在授权范围内")
            scope = RetrievalScope(scope.user_id, scope.collection_ids, (call.args.paper_id,))
        await self.store.validate_scope(scope)
        data: dict = {"tool": call.name, "arguments": call.args.model_dump(mode="json")}
        if isinstance(call, SearchCall):
            # A per-call copy prevents concurrent tool requests from mutating scope.
            retriever = self.retriever.model_copy(update={"scope": scope})
            result = await retriever.search(
                call.args.queries[0],
                probes=call.args.queries,
                section_hint=call.args.section_hint,
                modality_hint=call.args.modality_hint,
            )
            data.update(plan=result.plan, status="ok" if result.documents else "not_found")
            return result, data
        if isinstance(call, SectionCall):
            documents, parents = await self.store.read_section(
                scope,
                call.args.section_std,
                section_id=call.args.section_id,
                mode=call.args.mode,
            )
            required = {key for p in parents for key in p["source_ids"]}
            if not required <= {d.metadata["chunk_id"] for d in documents}:
                documents, parents = [], []
            data["parents"] = parents
        else:
            keys = labels(call.args.label)
            if len(keys) != 1 or not keys[0].startswith(("table:", "figure:", "equation:")):
                data.update(status="invalid_label")
                return RetrievalResult([]), data
            documents = await self.store.inspect(scope, keys[0])
        size = sum(len(d.page_content.encode()) for d in documents)
        size += sum(len(p["text"].encode()) for p in data.get("parents", []))
        if size > self.max_bytes:
            data.pop("parents", None)
            data.update(
                status="budget_exceeded",
                required_bytes=size,
                guidance="请按地图选择更小的 section_id，或读取 summary；元素与表格不会截断。",
            )
            return RetrievalResult([]), data
        data["status"] = "ok" if documents else "not_found"
        return RetrievalResult(documents), data


def merge_tool_documents(
    results: list[tuple[RetrievalResult, dict]], max_bytes: int
) -> tuple[list[Document], list[dict], list[str]]:
    selected: dict[str, Document] = {}
    reports, degraded = [], set()
    for result, report in results:
        additions = [d for d in result.documents if d.metadata["chunk_id"] not in selected]
        size = sum(len(d.page_content.encode()) for d in additions)
        size += sum(len(p["text"].encode()) for p in report.get("parents", []))
        if size > max_bytes:
            reports.append({"tool": report["tool"], "status": "combined_budget_exceeded"})
            continue
        max_bytes -= size
        selected.update((d.metadata["chunk_id"], d) for d in additions)
        reports.append(report)
        degraded.update(result.degraded)
    return list(selected.values()), reports, sorted(degraded)
