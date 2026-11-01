from app.services.chunking.academic import cited_references, references, section_role
from app.services.chunking.chunker import HARD_LIMIT, chunk_document
from tests.test_chunking import block, document


def test_overview_preserves_authors_abstract_contribution_and_conclusion():
    doc = document(
        [
            block("Abstract evidence. " * 500, path=["Abstract"]),
            block("Research background.", page=1, path=["1 Introduction"]),
            block("Our contribution is sparse routing.", page=2, path=["1 Introduction"]),
            block("Conclusion: latency improves.", page=4, path=["6 Conclusion"]),
        ]
    )
    overview = chunk_document(doc, "Scientific paper", authors=["Alice", "Bob"])[0]
    assert overview.kind == "document_meta" and overview.chunk_metadata["level"] == 0
    assert "Alice, Bob" in overview.text
    assert "Our contribution" in overview.text and "Research background" not in overview.text
    assert "Conclusion: latency improves" in overview.embedding_input
    assert overview.token_count <= HARD_LIMIT
    assert [loc["page_idx"] for loc in overview.locations] == [0, 2, 4]
    assert len(overview.source_spans) == 3


def test_metadata_only_abstract_does_not_invent_pdf_location():
    drafts = chunk_document(document([block("Actual body", page=3)]), "Paper", "External abstract")
    assert drafts[0].kind == "document_meta" and drafts[0].locations == []
    assert drafts[0].source_spans == []
    abstract = next(d for d in drafts if d.kind == "abstract")
    assert abstract.locations == []


def test_native_section_parent_survives_length_and_special_modalities():
    doc = document(
        [
            block("First paragraph. " * 160, page=0),
            block("Second paragraph. " * 160, page=1),
            block("<table><tr><th>Method</th></tr><tr><td>Ours</td></tr></table>", "table", page=2),
            block("Table 1: comparison", "caption", page=2),
            block("Final section paragraph. " * 160, page=3),
            block("Other section", page=4, section="s2", path=["Results"]),
        ]
    )
    drafts = chunk_document(doc, "Paper")
    first = drafts[:-1]
    assert len({id(d.parent) for d in first}) == 1
    assert first[0].parent.kind == "section"
    assert "Final section paragraph" in first[0].parent.text
    table = next(d for d in drafts if d.kind == "table")
    assert "Table 1: comparison" in table.embedding_input
    assert "<table>" in table.text and len(table.source_spans) == 2
    assert drafts[-1].parent is not first[0].parent


def test_contextual_prefix_distinguishes_identical_review_and_ablation_text():
    doc = document(
        [
            block("Sparse attention uses a local window.", path=["2 Related Work", "2.1 Model"]),
            block(
                "Sparse attention uses a local window.",
                path=["4 Experiments", "4.2 Ablation", "Results"],
            ),
            block("Sparse attention uses a local window.", path=["3 Methods"]),
        ]
    )
    drafts = chunk_document(doc, "Paper")
    assert len({d.content_hash for d in drafts}) == 3
    assert [d.chunk_metadata["section_role"] for d in drafts] == [
        "他人工作综述",
        "消融实验变体",
        "本文方法",
    ]
    assert (
        "[Paper: Paper | Section: 2 Related Work > 2.1 Model | Type: paragraph | Role: 他人工作综述]"
        in drafts[0].embedding_input
    )
    assert section_role(["Methods", "Ablation study"]) == "消融实验变体"


def test_numbered_references_are_isolated_and_numeric_citations_resolved():
    doc = document(
        [
            block("We compare with prior work [12, 13–14].", page=0),
            block(
                "[12] Alice and Bob. Sparse Routing. Conference, 2024.\n"
                "[13] Carol. Dense Routing. Conference, 2023.\n"
                "[14] Dan. Hybrid Routing. Conference, 2022.",
                page=4,
                path=["## 7. References"],
            ),
        ]
    )
    entries = references(doc)
    assert len(entries) == 3
    assert entries[0]["ref_id"] == "[12]" and entries[0]["year"] == 2024
    assert entries[0]["title"] == "Sparse Routing"
    drafts = chunk_document(doc, "Paper")
    assert len(drafts) == 1 and "Conference" not in drafts[0].text
    assert drafts[0].chunk_metadata["citations"] == entries
    assert cited_references("[999] [12-999999999]", entries) == []


def test_formula_chain_keeps_introduction_and_definitions_with_original_spans():
    source = [
        block("Define the objective.", page=0),
        block("$$ L = x^2 $$", "formula", page=1),
        block("Where x is the error. Then we define", page=2),
        block("$$ R = L + y $$", "formula", page=3),
        block("where y is regularization.", page=4),
    ]
    drafts = chunk_document(document(source), "Paper")
    assert len(drafts) == 1 and drafts[0].kind == "formula"
    assert drafts[0].text == "\n\n".join(b.text for b in source)
    assert [loc["page_idx"] for loc in drafts[0].locations] == [0, 1, 2, 3, 4]
    assert len(drafts[0].source_spans) == 5


def test_figure_links_caption_analysis_and_available_visual_description():
    source = [
        block("", "image", page=0),
        block("Figure 1: routing architecture.", "caption", page=0),
        block(
            "Unrelated sentence. Fig. 1 shows sparse routing. Figure 10 shows another model.",
            page=1,
        ),
        block("Figure 1 improves clarity. Other sentence.", page=2, path=["Results"]),
    ]
    source[0].text = "Two branches merge into one output."
    figure = next(d for d in chunk_document(document(source), "Paper") if d.kind == "figure")
    assert "Fig. 1 shows sparse routing." in figure.text
    assert "Figure 1 improves clarity." in figure.text
    assert "Two branches merge" in figure.text
    assert "Figure 10" not in figure.text and "Unrelated sentence" not in figure.text
    assert figure.chunk_metadata["visual_summary_available"]
    assert "跨章节分析" in figure.chunk_metadata["section_role"]
    assert any(p["section_path"] == ["Results"] for p in figure.chunk_metadata["source_provenance"])
    for span in figure.source_spans:
        original = next(b for b in source if b.id == span["block_id"])
        assert original.text[span["start"] : span["end"]] in figure.text
    assert {loc["page_idx"] for loc in figure.locations} == {0, 1, 2}


def test_caption_only_figure_does_not_invent_visual_summary():
    drafts = chunk_document(document([block("Figure 2: accuracy.", "caption")]), "Paper")
    figure = next(d for d in drafts if d.kind == "figure")
    assert figure.chunk_metadata["visual_summary_available"] is False
    assert figure.text == "Figure 2: accuracy."
