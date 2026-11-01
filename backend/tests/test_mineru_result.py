import json
from pathlib import Path

import pytest

from app.services.parsing.errors import ParsingError
from app.services.parsing.mineru_result import convert_bbox, load_result
from app.services.parsing.pdf import PDFInfo, PDFPage

FIXTURES = Path(__file__).parent / "fixtures/mineru"


@pytest.mark.parametrize("name,pages", [("attention", 15), ("lora", 26)])
def test_recorded_cloud_results_keep_structure_and_filter_noise(name, pages):
    root = FIXTURES / name
    pdf = PDFInfo.model_validate_json((root / "pdf_info.json").read_text())
    result = load_result(root, pdf)
    assert result.page_count == pages
    kinds = {b.kind for b in result.blocks}
    assert {"title", "table", "formula", "image", "caption", "reference"} <= kinds
    assert any("<table>" in b.text for b in result.blocks if b.kind == "table")
    assert all(b.indexable is False for b in result.blocks if b.kind == "reference")
    assert all(not b.indexable or b.kind != "reference" for b in result.blocks)
    assert all(loc.page_idx < pages for b in result.blocks for loc in b.locations)
    boxes = [loc.bbox for b in result.blocks for loc in b.locations if loc.bbox]
    assert boxes and all(0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1 for box in boxes)
    # Fixture intentionally omits binary assets; retain captions and page locators.
    assert all(b.asset_path is None for b in result.blocks)
    assert all(
        loc.bbox is None for b in result.blocks if b.kind == "caption" for loc in b.locations
    )
    assert not any("31st Conference" in b.text for b in result.blocks)


def test_recorded_rotated_and_cropped_pdf_matches_display_geometry():
    root = FIXTURES / "geometry"
    pdf = PDFInfo.model_validate_json((root / "pdf_info.json").read_text())
    assert pdf.pages[0].rotation == 90
    assert pdf.pages[1].crop_box != pdf.pages[1].media_box
    layout = json.loads((root / "layout.json").read_text())
    assert [p["page_size"] for p in layout["pdf_info"]] == [[792, 612], [480, 630]]
    result = load_result(root, pdf)
    rows = json.loads((root / "content_list.json").read_text())
    for block in result.blocks:
        if block.kind == "caption":
            continue
        index = int(block.id.split("-")[1])
        expected = tuple(v / 1000 for v in rows[index]["bbox"])
        assert block.locations[0].bbox == expected  # No accidental second rotation.
    assert any(b.kind == "text" and b.text for b in result.blocks)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [-1, 0, 1, 1],
        [0, 0, 1001, 1000],
        [1, 1, 1, 2],
        [0, 0, float("nan"), 1],
        [0, 0, float("inf"), 1],
        [False, 0, 1, 1],
        [0, 1, 2],
        "0,0,1,1",
    ],
)
def test_invalid_bbox_is_page_only(value):
    assert convert_bbox(value) is None


@pytest.mark.parametrize(
    "rotation,expected",
    [(90, (0.6, 0.1, 0.8, 0.3)), (180, (0.7, 0.6, 0.9, 0.8)), (270, (0.2, 0.7, 0.4, 0.9))],
)
def test_unrotated_crop_frame_converts_to_rotated_viewport(rotation, expected):
    p = PDFPage(
        width=800 if rotation != 180 else 600,
        height=600 if rotation != 180 else 800,
        rotation=rotation,
        crop_box=(0, 0, 600, 800),
        media_box=(0, 0, 600, 800),
    )
    # Rotation 180 has identical dimensions; the frame cannot be inferred from
    # width/height alone. Display-oriented cloud output is the supported contract.
    if rotation == 180:
        assert convert_bbox([100, 200, 300, 400], p, [600, 800]) == (0.1, 0.2, 0.3, 0.4)
    else:
        assert convert_bbox([100, 200, 300, 400], p, [600, 800]) == pytest.approx(expected)


def test_crop_offset_and_ambiguous_geometry():
    p = PDFPage(
        width=400, height=600, rotation=0, crop_box=(100, 100, 500, 700), media_box=(0, 0, 600, 800)
    )
    assert convert_bbox([250, 250, 500, 500], p, [600, 800]) == pytest.approx(
        (0.125, 1 / 6, 0.5, 0.5)
    )
    assert convert_bbox([250, 250, 500, 500], p, None) is None
    assert convert_bbox([250, 250, 500, 500], p, [222, 333]) is None
    assert convert_bbox([0, 0, 500, 500], p, [600, 800]) is None


def write_content(root, rows):
    (root / "content_list.json").write_text(json.dumps(rows))


def test_section_paths_reference_exclusion_missing_bbox_and_cross_page(tmp_path):
    rows = [
        {"type": "text", "text": "Methods", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Details", "text_level": 2, "page_idx": 0},
        {
            "type": "text",
            "text": "continues on next page",
            "page_idx": 1,
            "bbox": [0, 0, 1000, 1000],
        },
        {"type": "text", "text": "References", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": "reference body", "page_idx": 1},
    ]
    write_content(tmp_path, rows)
    doc = load_result(tmp_path)
    assert doc.sections[1].parent_id == doc.sections[0].id
    assert doc.blocks[2].section_path == ["Methods", "Details"]
    assert doc.blocks[2].locations[0].page_idx == 1
    assert doc.blocks[0].locations[0].bbox is None
    assert not doc.blocks[-1].indexable
    assert doc.warnings


@pytest.mark.parametrize(
    "rows",
    [
        [],
        {},
        [{"type": "text", "text": "x", "page_idx": -1}],
        [{"type": "text", "text": "x", "page_idx": True}],
        [{"type": "image", "page_idx": 0, "img_path": "../../escape"}],
    ],
)
def test_malformed_content_rejected(tmp_path, rows):
    write_content(tmp_path, rows)
    with pytest.raises(ParsingError):
        load_result(tmp_path)
