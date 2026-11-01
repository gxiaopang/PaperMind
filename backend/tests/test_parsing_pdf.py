import pytest
from pypdf import PdfWriter

from app.services.parsing.errors import ParsingError
from app.services.parsing.pdf import inspect_pdf
from tests.test_mineru_client import settings


def make_pdf(path, *, encrypted=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    page.cropbox.lower_left = (100, 100)
    page.cropbox.upper_right = (500, 700)
    page.rotate(90)
    writer.add_metadata({"/Title": "A paper", "/Author": "An Author"})
    if encrypted:
        writer.encrypt("secret")
    writer.write(path)


def test_pdf_validation_metadata_crop_and_rotation(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    info = inspect_pdf(path, settings())
    assert info.title == "A paper" and info.author == "An Author"
    assert info.pages[0].width == 600 and info.pages[0].height == 400
    assert info.pages[0].rotation == 90 and info.pages[0].crop_box == (100, 100, 500, 700)
    assert len(info.sha256) == 64 and info.first_page_text == ""


@pytest.mark.parametrize("mode", ["magic", "corrupt", "encrypted", "pages", "size", "empty"])
def test_invalid_pdf_fails_before_network(tmp_path, mode):
    path = tmp_path / "bad.pdf"
    config = settings()
    if mode == "magic":
        path.write_bytes(b"not a PDF")
    elif mode == "corrupt":
        path.write_bytes(b"%PDF-1.7\ntruncated")
    elif mode == "encrypted":
        make_pdf(path, encrypted=True)
    elif mode == "size":
        path.write_bytes(b"%PDF-" + b"0" * 1024**2)
        config = settings(max_upload_mb=1)
    elif mode == "empty":
        PdfWriter().write(path)
    else:
        writer = PdfWriter()
        writer.add_blank_page(width=10, height=10)
        writer.add_blank_page(width=10, height=10)
        writer.write(path)
        config = settings(max_upload_pages=1)
    with pytest.raises(ParsingError):
        inspect_pdf(path, config)
