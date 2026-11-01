from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db.models import Chunk, ChunkParent, Paper, PaperVersion, Section
from app.services.chunking.academic import markdown_table, table_summary
from app.services.chunking.chunker import chunk_document
from app.services.chunking.paper_map import build_paper_map, section_std
from app.services.parsing.mineru_result import Section as ParsedSection
from app.services.retrieval.planning import fusion_policy, route_query
from app.services.retrieval.tools import (
    ElementArguments,
    ElementCall,
    PaperTools,
    SearchArguments,
    SearchCall,
    SectionArguments,
    SectionCall,
    ToolPlan,
    boundary_calls,
    element_followups,
)
from app.services.retrieval.types import RetrievalScope
from tests.test_chunking import block, document
from tests.test_parent_retrieval_store import parent_store as parent_store
from tests.test_retrieval_pipeline import offline_retriever


def test_mixed_tool_plan_rejects_bad_arguments_without_mutating_valid_call():
    from app.services.retrieval.tools import ToolPlan, validate_tool_calls

    paper = str(uuid4())
    result = ToolPlan.model_validate(
        validate_tool_calls(
            [
                {
                    "name": "read_paper_section",
                    "args": {"paper_id": paper, "section_std": "method", "mode": "full"},
                },
                {"name": "inspect_element", "args": {"section_id": str(uuid4()), "mode": "full"}},
            ]
        )
    )
    assert len(result.calls) == 1 and str(result.calls[0].args.paper_id) == paper
    assert result.calls[0].name == "read_paper_section"
    assert result.rejected_calls[0]["name"] == "inspect_element"
    assert {i["type"] for i in result.rejected_calls[0]["issues"]} == {"extra_forbidden", "missing"}
    with pytest.raises(ValidationError):
        ToolPlan.model_validate(
            validate_tool_calls(
                [
                    {"name": "inspect_element", "args": {"user_id": "foreign", "mode": "full"}},
                ]
            )
        )


def test_map_keeps_full_abstract_two_level_tree_and_every_caption():
    abstract = "Full abstract evidence. " * 2000
    doc = document(
        [block(abstract, path=["Abstract"])]
        + [block(f"Table {i}: GSM8K full caption.", "caption", page=1) for i in range(1, 41)]
        + [block("Figure 2: architecture.", "caption", page=2)]
    )
    doc.sections = [
        ParsedSection(id="m", parent_id=None, title="3 Method", level=2, path=["3 Method"]),
        ParsedSection(
            id="s", parent_id="m", title="3.2 Loss", level=3, path=["3 Method", "3.2 Loss"]
        ),
        ParsedSection(
            id="d", parent_id="s", title="Detail", level=4, path=["3 Method", "3.2 Loss", "Detail"]
        ),
    ]
    value = build_paper_map(doc, "Paper", "Short external abstract")
    assert value["abstract"] == abstract
    assert value["toc"][0]["children"][0]["title"] == "3.2 Loss"
    assert value["toc"][0]["children"][0]["children"] == []
    assert len(value["elements"]) == 41
    assert value["elements"][-1]["label"] == "Figure 2"
    assert section_std(["Discussion", "Limitations"]) == "limitations"
    assert section_std(["Experiments", "Ablation study"]) == "ablation"
    assert section_std(["Related Work", "Model"]) == "related_work"


def navigation(c):
    ids = [uuid4() for _ in range(3)]
    with Session(c.engine) as db:
        for index, (title, path, standard) in enumerate(
            [
                ("3 Method", ["3 Method"], "method"),
                ("3.2 Loss", ["3 Method", "3.2 Loss"], "method"),
                ("Limitations", ["Limitations"], "limitations"),
            ]
        ):
            db.add(
                Section(
                    id=ids[index],
                    user_id=c.owner,
                    version_id=c.versions[0],
                    parent_id=ids[0] if index == 1 else None,
                    title=title,
                    path=path,
                    level=2 if index == 1 else 1,
                    section_std=standard,
                )
            )
        for i, chunk_id in enumerate(c.chunks[0]):
            chunk = db.get(Chunk, chunk_id)
            chunk.section_id = ids[0 if i < 3 else i - 2]
            if i == 1:
                chunk.indexable, chunk.embedding = False, None
        for i, parent_id in enumerate(c.parents[:3]):
            parent = db.get(ChunkParent, parent_id)
            parent.section_id = ids[i]
            indices = list(range(3)) if i == 0 else [i + 2]
            parent.text = "\n\n".join(f"Original source {j}." for j in indices)
        db.commit()
    return ids


async def test_full_l1_read_includes_unindexed_formula_and_descendants_without_providers(
    parent_store,
):
    c = parent_store
    ids = navigation(c)
    docs, parents = await c.store.read_section(c.scope, "method")
    assert [d.metadata["ordinal"] for d in docs] == [0, 1, 2, 3]
    assert [p["text"] for p in parents] == [
        "Original source 0.\n\nOriginal source 1.\n\nOriginal source 2.",
        "Original source 3.",
    ]
    assert all(p["complete"] for p in parents)
    docs, _ = await c.store.read_section(c.scope, "method", section_id=ids[0])
    assert len(docs) == 4
    docs, summaries = await c.store.read_section(c.scope, "method", mode="summary")
    assert [d.metadata["ordinal"] for d in docs] == [0, 2, 3]
    assert all(p["summary_method"] == "extractive" and not p["complete"] for p in summaries)
    assert await c.store.read_section(RetrievalScope(c.owner, paper_ids=()), "method") == ([], [])


async def test_exact_element_inspection_fetches_complete_unindexed_target_and_analysis(
    parent_store,
):
    c = parent_store
    table = "| Model | Score |\n| --- | --- |\n" + "| Ours | 92.3 |\n" * 2000
    with Session(c.engine) as db:
        source = db.get(Chunk, c.chunks[0][0])
        source.text = "Table 2 demonstrates the accuracy increase."
        source.chunk_metadata = {"cross_refs": [{"label": "table:2", "target_ordinals": [4]}]}
        target = db.get(Chunk, c.chunks[0][4])
        target.text, target.kind, target.indexable, target.embedding = table, "table", False, None
        target.chunk_metadata = {"defined_labels": ["table:2"]}
        db.get(Chunk, c.chunks[0][3]).chunk_metadata = {"defined_labels": ["table:20"]}
        db.commit()
    tools = PaperTools(c.store, c.scope, None)
    result, report = await tools.execute(
        ElementCall(name="inspect_element", args=ElementArguments(label="表 2"))
    )
    assert report["status"] == "ok"
    assert [d.metadata["ordinal"] for d in result.documents] == [4, 0]
    assert result.documents[0].page_content == table
    result, report = await tools.execute(
        ElementCall(name="inspect_element", args=ElementArguments(label="Table 200"))
    )
    assert not result.documents and report["status"] == "not_found"
    tools.max_bytes = 100
    result, report = await tools.execute(
        ElementCall(name="inspect_element", args=ElementArguments(label="Table 2"))
    )
    assert not result.documents and report["status"] == "budget_exceeded"


async def test_map_scope_and_version_switch_revoke_navigation(parent_store):
    c = parent_store
    navigation(c)
    with Session(c.engine) as db:
        version = db.get(PaperVersion, c.versions[0])
        version.paper_map = {
            "title": "Paper",
            "abstract": "Full abstract",
            "toc": [],
            "elements": [],
        }
        db.commit()
    maps = await c.store.paper_maps(c.scope)
    assert len(maps) == 1 and maps[0]["abstract"] == "Full abstract"
    assert maps[0]["paper_id"] == str(c.papers[0])
    with pytest.raises(HTTPException):
        await c.store.paper_maps(RetrievalScope(c.owner, paper_ids=(c.papers[1],)))
    with Session(c.engine) as db:
        db.get(Paper, c.papers[0]).active_version_id = None
        db.commit()
    assert await c.store.paper_maps(c.scope) == []
    assert await c.store.read_section(c.scope, "method") == ([], [])


async def test_native_tool_plan_cannot_set_tenant_and_paper_scope_only_narrows(parent_store):
    c = parent_store
    with pytest.raises(ValidationError):
        ToolPlan.model_validate(
            {
                "calls": [
                    {
                        "name": "read_paper_section",
                        "args": {"section_std": "method", "user_id": str(c.foreign)},
                    }
                ]
            }
        )
    tools = PaperTools(c.store, RetrievalScope(c.owner, paper_ids=(c.papers[0],)), None)
    with pytest.raises(HTTPException) as error:
        await tools.execute(
            SectionCall(
                name="read_paper_section",
                args=SectionArguments(section_std="method", paper_id=c.papers[1]),
            )
        )
    assert error.value.status_code == 404


async def test_explicit_probes_share_parallel_recall_and_adapt_fusion(parent_store, monkeypatch):
    c = parent_store
    retriever, seen = offline_retriever(c, monkeypatch)
    result, report = await PaperTools(c.store, c.scope, retriever).execute(
        SearchCall(
            name="search_paper_chunks",
            args=SearchArguments(
                queries=["failure cases", "compute overhead", "hidden assumptions"],
                section_hint="method",
            ),
        )
    )
    assert len(seen["queries"]) == 3 and result.documents
    assert report["plan"]["rrf_k"] == 90
    metric = fusion_policy(route_query("GSM8K accuracy"), 1, 1, 60)
    semantic = fusion_policy(route_query("hidden weakness"), 1, 1, 60)
    assert metric["alpha"] < semantic["alpha"] and metric["k"] < semantic["k"]


def test_idea_guards_require_full_boundaries_and_tables_keep_all_markdown_rows():
    search = SearchCall(
        name="search_paper_chunks", args=SearchArguments(queries=["possible improvement"])
    )
    calls = boundary_calls("请提出一个新 IDEA", [search])
    assert [c.args.section_std for c in calls if isinstance(c, SectionCall)] == [
        "limitations",
        "method",
    ]
    assert all(c.args.mode == "full" for c in calls if isinstance(c, SectionCall))
    raw = (
        "<table><tr><th>Method</th><th>Score</th></tr>"
        + "".join(f"<tr><td>Model {i}</td><td>{i}</td></tr>" for i in range(2000))
        + "</table>"
    )
    draft = chunk_document(document([block(raw, "table")]), "Paper")[0]
    assert draft.text == raw and draft.chunk_metadata["raw_table"] == raw
    assert "| Model 1999 | 1999 |" in draft.chunk_metadata["markdown_table"]
    assert markdown_table("| x | y |") == "| x | y |"
    summary_text = table_summary(
        "| Method | Score |\n| --- | --- |\n| A | 1.2 |\n| Middle | 99.3 |\n| Z | 3.4 |"
    )
    assert "最大为 99.3（Middle）" in summary_text and "最小为 1.2（A）" in summary_text
    summary = SectionCall(
        name="read_paper_section", args=SectionArguments(section_std="method", mode="summary")
    )
    guarded = boundary_calls("new research idea", [summary])
    assert any(
        isinstance(c, SectionCall) and c.args.section_std == "method" and c.args.mode == "full"
        for c in guarded
    )


async def test_model_adapter_binds_three_native_tools_and_validates_arguments(monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage

    from app.core.config import Settings
    from app.services.chat.llm import ChatModels

    calls = []

    class Accounting:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def slot(self, user_id):
            yield

        async def call(self, user_id, operation_id, node, amount, callback, usage, **kwargs):
            result = await callback()
            calls.append((node, usage()))
            return result

    class Client:
        bound = []

        def bind_tools(self, tools, **kwargs):
            self.bound = [tool["function"]["name"] for tool in tools]
            assert kwargs["tool_choice"] == "required"
            return self

        async def ainvoke(self, messages, **kwargs):
            assert "response_format" not in kwargs
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inspect_element",
                        "args": {"label": "Table 2"},
                        "id": "tool-1",
                    }
                ],
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            )

    monkeypatch.setattr("app.services.chat.llm.RetrievalAccounting", Accounting)
    models = ChatModels.__new__(ChatModels)
    models.settings = Settings(_env_file=None, tracing_enabled=False)
    models.sessions = models.engine = None
    models.fast = Client()
    plan = await models.invoke(
        SimpleNamespace(id=uuid4(), user_id=uuid4()), "plan_paper_tools", {}, {}, schema=ToolPlan
    )
    assert set(models.fast.bound) == {
        "search_paper_chunks",
        "read_paper_section",
        "inspect_element",
    }
    assert plan.calls[0].name == "inspect_element" and plan.calls[0].args.label == "Table 2"
    assert calls == [("plan_paper_tools", 13)]


def test_reading_prose_proactively_follows_elements_once():
    from langchain_core.documents import Document

    paper = str(uuid4())
    docs = [
        Document(
            page_content="See Table 2 and Eq. 4.",
            metadata={
                "paper_id": paper,
                "chunk_metadata": {
                    "cross_refs": [
                        {"label": "table:2"},
                        {"label": "equation:4"},
                        {"label": "section:3"},
                    ]
                },
            },
        )
    ]
    calls = element_followups(docs, [])
    assert [c.args.label for c in calls] == ["table 2", "equation 4"]
    assert not element_followups(docs, calls)


async def test_section_revocation_cannot_leave_stale_full_parent(parent_store, monkeypatch):
    c = parent_store
    navigation(c)
    refresh = c.store.refresh

    async def revoked(scope, docs):
        with Session(c.engine) as db:
            db.delete(db.get(Chunk, c.chunks[0][1]))
            db.commit()
        return await refresh(scope, docs)

    monkeypatch.setattr(c.store, "refresh", revoked)
    result, report = await PaperTools(c.store, c.scope, None).execute(
        SectionCall(name="read_paper_section", args=SectionArguments(section_std="method"))
    )
    assert not result.documents and report["parents"] == []
