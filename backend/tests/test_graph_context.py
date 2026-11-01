import copy

import pytest

from app.graph.context import CONTROL_EVIDENCE_BYTES, node_payload, serialize
from app.graph.prompts import prompt
from app.graph.state import RAGState


@pytest.fixture
def context():
    text = "完整方法原文。" * 8000 + "不可丢失的结尾条件。"
    table = "<table><tr><td>完整实验数据 28.4</td></tr></table>"
    documents = [
        {
            "text": value,
            "metadata": {
                "chunk_id": str(i),
                "paper_id": "paper-1",
                "title": "LoRA",
                "section_path": ["Methods"],
                "parent_id": "parent-1",
                "parent_complete": True,
                "retrieval_role": "section",
                "unresolved_cross_refs": ["table:2"],
                "chunk_metadata": {
                    "section_role": "method",
                    "citations": [{"ref_id": "ref-1"}],
                    "source_provenance": [{"section_role": "method", "block_id": str(i)}],
                },
            },
        }
        for i, value in enumerate([text, table], 1)
    ]
    state = RAGState(
        question="它还有什么限制？",
        rewritten_query="LoRA 方法的限制是什么？",
        history=[{"role": "user", "content": "我们在讨论 LoRA。"}],
        summary="讨论低秩适配。",
        active_paper_ids=["paper-1"],
        paper_maps=[
            {
                "paper_id": "paper-1",
                "title": "LoRA",
                "abstract": "完整摘要" * 2000,
                "toc": [{"section_id": "section-1", "title": "Methods"}],
                "elements": [{"label": "Table 2", "caption": "完整图表索引"}],
            }
        ],
        tool_reports=[
            {
                "tool": "read_paper_section",
                "arguments": {"section_std": "method", "mode": "full"},
                "status": "ok",
                "parents": [
                    {
                        "parent_id": "parent-1",
                        "text": text + "\n\n" + table,
                        "complete": True,
                        "source_ids": ["1", "2"],
                    }
                ],
                "plan": {"unused": "verbose search diagnostics" * 100},
            },
            {
                "tool": "read_paper_section",
                "status": "not_found",
                "arguments": {"section_std": "limitations"},
            },
        ],
        answer="这是上一版有问题的答案[1]。",
        verify_result={"citation_issues": {"uncited_units": ["需要修正的事实句"]}},
        preferences={"answer_language": "zh", "detail_level": "concise"},
    )
    return state, documents


def payload(node, state, documents):
    return node_payload(node, state, documents, max_search_probes=3)


@pytest.mark.parametrize("node", ["route", "rewrite_query", "direct_answer"])
def test_conversation_nodes_do_not_receive_paper_bodies_or_navigation(context, node):
    state, documents = context
    data = payload(node, state, documents)
    encoded = serialize(data)
    assert "完整摘要" not in encoded and "完整方法原文" not in encoded
    assert "paper_maps" not in data and "sources" not in data and "tool_reports" not in data
    assert data["history"] == state["history"] and data["summary"] == state["summary"]
    if node == "rewrite_query":
        assert data["papers"] == [{"paper_id": "paper-1", "title": "LoRA"}]
        assert data["active_papers"] == ["paper-1"]


@pytest.mark.parametrize("node", ["plan_paper_tools", "transform_query"])
def test_control_nodes_receive_bounded_explicit_previews_not_full_evidence(context, node):
    state, documents = context
    data = payload(node, state, documents)
    assert "sources" not in data and "answer" not in data and "history" not in data
    assert data["query"] == state["rewritten_query"]
    inventory = data["evidence_inventory"]
    assert inventory["preview_only"] and inventory["omitted_sources"] == 0
    assert len(serialize(inventory).encode()) <= CONTROL_EVIDENCE_BYTES
    assert len(inventory["sources"][0]["preview"].encode()) <= 512
    assert "不可丢失的结尾条件" not in serialize(data)
    assert inventory["sources"][0]["unresolved_cross_refs"] == ["table:2"]
    assert data["tool_reports"][1]["status"] == "not_found"
    assert "plan" not in data["tool_reports"][0]
    if node == "plan_paper_tools":
        assert data["paper_maps"] == state["paper_maps"]
    else:
        assert "paper_maps" not in data
    assert "预览" in prompt(node).format_messages(payload=serialize(data))[0].content


def test_inventory_bounds_multibyte_sources_and_reports_omissions(context):
    state, documents = context
    many = []
    for i in range(200):
        doc = copy.deepcopy(documents[0])
        doc["metadata"]["chunk_id"] = str(i)
        many.append(doc)
    inventory = payload("transform_query", state, many)["evidence_inventory"]
    assert len(serialize(inventory).encode()) <= CONTROL_EVIDENCE_BYTES
    assert len(inventory["sources"]) + inventory["omitted_sources"] == len(many)
    assert inventory["omitted_sources"] > 0
    # A large metadata record must not prevent later small records being offered.
    many[0]["metadata"]["unresolved_cross_refs"] = ["large" * 10000]
    inventory = payload("transform_query", state, many)["evidence_inventory"]
    assert inventory["sources"][0]["chunk_id"] == "1"
    assert inventory["sources"][0]["number"] == 2


@pytest.mark.parametrize("node", ["generate", "grade_documents", "verify_grounding"])
def test_evidence_nodes_keep_complete_ordered_sources_without_duplicate_parent_text(context, node):
    state, documents = context
    before = copy.deepcopy((state, documents))
    data = payload(node, state, documents)
    assert [s["text"] for s in data["sources"]] == [d["text"] for d in documents]
    assert [s["number"] for s in data["sources"]] == [1, 2]
    assert "paper_maps" not in data and "evidence_inventory" not in data
    parent = data["tool_reports"][0]["parents"][0]
    assert "text" not in parent and parent["text_in_sources"]
    assert parent["source_ids"] == ["1", "2"] and parent["complete"]
    assert data["sources"][0]["section_role"] == "method"
    assert (
        data["sources"][0]["source_provenance"]
        == documents[0]["metadata"]["chunk_metadata"]["source_provenance"]
    )
    assert serialize(data).count("不可丢失的结尾条件") == 1
    assert (state, documents) == before
    if node == "generate":
        assert data["preferences"] == state["preferences"]
        assert data["history"] == state["history"]
        assert data["previous_validation"] == state["verify_result"]
        assert data["answer"] == state["answer"]
    else:
        assert "history" not in data and "summary" not in data and "preferences" not in data
        assert "previous_validation" not in data
        if node == "verify_grounding":
            assert data["answer"] == state["answer"]
        else:
            assert "answer" not in data


def test_unavailable_source_cannot_be_marked_as_complete_parent(context):
    state, documents = context
    data = payload("generate", state, documents[:1])
    assert "parents" not in data["tool_reports"][0]
    state["verify_result"] = {}
    data = payload("generate", state, documents)
    assert "answer" not in data and "previous_validation" not in data
