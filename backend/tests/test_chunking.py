from pathlib import Path

from app.eval.retrieval import load_dataset
from app.services.chunking.chunker import HARD_LIMIT, chunk_document, split_spans, token_count
from app.services.parsing.mineru_result import Location, ParsedBlock, ParsedDocument, load_result


def document(blocks):
    return ParsedDocument(
        page_count=5,
        blocks=blocks,
        sections=[],
        markdown_path=None,
        content_list_path="content_list.json",
        layout_path=None,
        assets=[],
        warnings=[],
    )


def block(text, kind="text", page=0, section="s", path=None, indexable=True):
    return ParsedBlock(
        id=f"b-{page}",
        kind=kind,
        text=text,
        locations=[Location(page_idx=page, bbox=(0.1, 0.2, 0.8, 0.4))],
        section_id=section,
        section_path=path or ["Methods"],
        indexable=indexable,
    )


def test_mixed_language_prefix_and_hard_limit():
    drafts = chunk_document(
        document([block("中文科研公式 $x=1$ with English words. " * 500)]), "标题" * 200
    )
    assert len(drafts) > 10
    assert all(d.token_count <= HARD_LIMIT for d in drafts)
    assert all(d.token_count == token_count(d.embedding_input) for d in drafts)
    assert all(d.locations[0]["bbox"] == [0.1, 0.2, 0.8, 0.4] for d in drafts)


def test_sections_do_not_mix_and_cross_page_locations_survive():
    drafts = chunk_document(
        document(
            [
                block("first paragraph", page=0),
                block("second paragraph", page=1),
                block("new section", page=2, section="s2", path=["Results"]),
            ]
        ),
        "A paper",
    )
    assert len(drafts) == 2
    assert [loc["page_idx"] for loc in drafts[0].locations] == [0, 1]
    assert "new section" not in drafts[0].text


def test_overlap_and_complete_text_coverage():
    text = " ".join(f"word{i}" for i in range(500))
    drafts = chunk_document(document([block(text)]), "Paper")
    for word in text.split():
        assert any(word in d.text for d in drafts)
    assert any(drafts[0].text[-30:] in d.text for d in drafts[1:])


def test_large_tables_remain_whole_with_bounded_search_summary():
    header = "<tr><th>Method</th><th>Score</th></tr>"
    table = (
        "<table>"
        + header
        + "".join(f"<tr><td>Method {i}</td><td>0.95</td></tr>" for i in range(200))
        + "</table>"
    )
    drafts = chunk_document(document([block(table, "table")]), "Paper")
    assert len(drafts) == 1
    assert drafts[0].text == table
    assert drafts[0].kind == "table" and drafts[0].token_count <= HARD_LIMIT
    assert drafts[0].indexable
    assert "Method 199" in drafts[0].embedding_input
    assert drafts[0].source_spans == [{"block_id": "b-0", "start": 0, "end": len(table)}]


def test_long_formula_kept_as_unindexed_preview():
    formula = "$$" + "x + " * 2000 + "x$$"
    drafts = chunk_document(document([block(formula, "formula")]), "Paper")
    assert drafts[0].text == formula
    assert drafts[0].indexable is False
    assert drafts[0].unindexed_reason == "formula_exceeds_input_limit"


def test_abstract_has_overview_references_excluded_and_caption_preserved():
    drafts = chunk_document(
        document(
            [
                block("body"),
                block("abstract", path=["Abstract"]),
                block("ref", kind="reference", indexable=False),
                block("caption", kind="caption"),
                block("x=1", kind="formula"),
            ]
        ),
        "Paper",
    )
    assert [d.kind for d in drafts] == ["document_meta", "text", "abstract", "caption", "formula"]
    assert not any("ref" == d.text for d in drafts)


def test_hash_includes_full_prefix_and_content():
    doc = document([block("shared text")])
    assert (
        chunk_document(doc, "Paper A")[0].content_hash
        != chunk_document(doc, "Paper B")[0].content_hash
    )


def test_recorded_cloud_samples_produce_bounded_chunks():
    root = Path(__file__).parent / "fixtures/mineru"
    for name in ("attention", "lora", "geometry"):
        drafts = chunk_document(load_result(root / name), "Recorded paper")
        assert drafts and all(d.token_count <= HARD_LIMIT for d in drafts if d.indexable)


def test_encoder_paragraph_is_kept_complete_including_previously_split_sentence():
    _, corpus = load_dataset(Path(__file__).parents[2] / "eval/datasets/m3-gold-v7.json")
    original = next(
        row["text"] for row in corpus["attention"]["chunks"] if row["text"].startswith("Encoder:")
    ).split("\n\nDecoder:", 1)[0]
    drafts = chunk_document(document([block(original)]), "Attention Is All You Need")
    assert len(drafts) == 1
    assert drafts[0].text == original
    assert "To facilitate these residual connections" in drafts[0].text
    assert drafts[0].parent.text == original


def test_real_bge_tokens_replace_byte_lengths_and_prefix_budget_is_enforced():
    text = "A complete scientific sentence with ordinary English words."
    assert 2 < token_count(text) < len(text.encode())
    drafts = chunk_document(document([block("实验结果与公式推导。" * 1000)]), "Paper " * 1000)
    assert all(d.token_count <= HARD_LIMIT for d in drafts if d.indexable)


def test_complete_paragraphs_share_parent_without_unnecessary_overlap_or_bbox_spill():
    first, second = "alpha " * 350 + ".", "beta " * 350 + "."
    drafts = chunk_document(document([block(first, page=0), block(second, page=1)]), "Paper")
    assert len(drafts) == 2 and drafts[0].parent is drafts[1].parent
    assert [d.text for d in drafts] == [first, second]
    assert [[p["page_idx"] for p in d.locations] for d in drafts] == [[0], [1]]
    assert [p["page_idx"] for p in drafts[0].parent.locations] == [0, 1]
    assert drafts[0].source_spans == [{"block_id": "b-0", "start": 0, "end": len(first)}]
    assert drafts[1].source_spans == [{"block_id": "b-1", "start": 0, "end": len(second)}]


def test_formula_and_definitions_form_one_citable_child():
    drafts = chunk_document(
        document(
            [
                block("The loss is defined as follows."),
                block("$$ L = \\sum_i x_i^2 $$", kind="formula"),
                block("Here x denotes an input value."),
            ]
        ),
        "Paper",
    )
    assert [d.kind for d in drafts] == ["formula"]
    assert "The loss is defined" in drafts[0].text
    assert "Here x denotes" in drafts[0].text
    assert len(drafts[0].source_spans) == 3
    assert len({id(d.parent) for d in drafts}) == 1
    assert "Here x denotes" in drafts[0].parent.text


def test_null_section_ids_still_respect_distinct_section_paths():
    drafts = chunk_document(
        document(
            [
                block("Method description.", section=None, path=["Methods"]),
                block("Result description.", section=None, path=["Results"]),
            ]
        ),
        "Paper",
    )
    assert len(drafts) == 2 and drafts[0].parent is not drafts[1].parent


def test_long_paragraph_splits_at_complete_sentences_without_cutting_inline_formula():
    sentences = [
        f"Sentence {i} explains $x_{{{i}}} = a + b$ using complete words." for i in range(30)
    ]
    original = " ".join(sentences)
    spans = split_spans(original, 60, 100)
    assert len(spans) > 1
    for start, end in spans:
        piece = original[start:end]
        assert piece.endswith(".") and piece.count("$") % 2 == 0
        assert token_count(piece) <= 100
    assert all(any(sentence in original[a:b] for a, b in spans) for sentence in sentences)


def test_oversized_inline_formula_is_not_cut_or_silently_dropped():
    formula = "$" + "x + " * 1500 + "z$"
    drafts = chunk_document(
        document([block("Before formula. " + formula + " After formula.")]), "Paper"
    )
    assert any(formula in d.text and not d.indexable for d in drafts)
    assert all(d.text.count("$") % 2 == 0 for d in drafts)
    assert all(d.token_count <= HARD_LIMIT for d in drafts if d.indexable)


def test_no_word_is_sliced_when_forced_to_split_a_long_sentence():
    words = [f"scientificword{i}" for i in range(200)]
    text = " ".join(words)
    pieces = [text[a:b] for a, b in split_spans(text, 45, 65, overlap=10)]
    assert set(" ".join(pieces).split()) == set(words)
