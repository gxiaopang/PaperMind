import json

import httpx
import pytest
import respx

from app.services.parsing.metadata import (
    MetadataExtraction,
    MetadataExtractionSkipped,
    MetadataService,
    PaperMetadata,
    apply_evidence,
    local_metadata,
)
from app.services.parsing.mineru_result import load_result
from app.services.parsing.pdf import PDFInfo
from tests.test_mineru_result import FIXTURES


def document():
    return load_result(FIXTURES / "attention")


def pdf(text="", **kwargs):
    return PDFInfo(sha256="abc", size=1, pages=[], first_page_text=text, **kwargs)


def test_local_metadata_reads_properties_and_identifiers_without_guessing():
    result = local_metadata(
        pdf("arXiv:2106.09685v2 doi:10.1234/test-1.", title="A Paper", author="Alice; Bob"),
        document(),
    )
    assert result.title == "A Paper" and result.authors == ["Alice", "Bob"]
    assert result.arxiv_id == "2106.09685v2" and result.doi == "10.1234/test-1"
    assert result.year is None and result.venue is None
    assert result.sources["title"] == "pdf:properties"


def test_llm_requires_verbatim_evidence_and_preserves_known_fields():
    text = "A Paper\nAlice Example\nPublished in 2024 at Example Conference."
    extraction = MetadataExtraction.model_validate(
        {
            "candidates": [
                {"field": "title", "value": "Wrong title", "evidence": "Wrong title"},
                {"field": "authors", "value": "Alice Example", "evidence": "Alice Example"},
                {"field": "authors", "value": "Imagined Author", "evidence": "Alice Example"},
                {
                    "field": "year",
                    "value": "2024",
                    "evidence": "Published in 2024 at Example Conference.",
                },
                {"field": "doi", "value": "10.1234/fake", "evidence": "10.1234/fake"},
                {
                    "field": "venue",
                    "value": "Example Conference",
                    "evidence": "Published in 2024 at Example Conference.",
                },
            ]
        }
    )
    result = PaperMetadata(title="A Paper", sources={"title": "pdf"})
    apply_evidence(result, extraction, text)
    assert result.title == "A Paper" and result.authors == ["Alice Example"] and result.year == 2024
    assert result.doi is None and result.venue == "Example Conference"
    assert result.sources["year"] == "llm:evidence" and result.warnings


@respx.mock
async def test_crossref_exact_identifier_enrichment():
    route = respx.get("https://api.crossref.org/works/10.1234%2Fexample").respond(
        json={
            "message": {
                "DOI": "10.1234/example",
                "title": ["Canonical title"],
                "author": [{"given": "Alice", "family": "Example"}],
                "container-title": ["Example Journal"],
                "published": {"date-parts": [[2020, 1]]},
                "abstract": "<jats:p>Abstract</jats:p>",
            }
        }
    )
    async with httpx.AsyncClient() as http:
        result = await MetadataService(http).enrich(pdf("doi:10.1234/example"), document())
    assert (
        result.authors == ["Alice Example"]
        and result.year == 2020
        and result.venue == "Example Journal"
    )
    assert result.abstract == "Abstract" and result.sources["authors"] == "crossref"
    assert route.call_count == 1


@respx.mock
async def test_arxiv_atom_fields_and_identifier_match():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:x="http://arxiv.org/schemas/atom">
    <entry><id>http://arxiv.org/abs/1706.03762v7</id><title>Attention Is All You Need</title>
    <published>2017-06-12T00:00:00Z</published><summary>Abstract here</summary>
    <author><name>Alice Example</name></author><x:doi>10.1234/example</x:doi></entry></feed>"""
    respx.get("https://export.arxiv.org/api/query", params={"id_list": "1706.03762v7"}).respond(
        text=xml
    )
    async with httpx.AsyncClient() as http:
        result = await MetadataService(http).enrich(pdf("arXiv:1706.03762v7"), document())
    assert (
        result.year == 2017
        and result.authors == ["Alice Example"]
        and result.doi == "10.1234/example"
    )
    assert result.sources["year"] == "arxiv" and not result.warnings


@pytest.mark.parametrize(
    "body",
    [
        "<broken>",
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/9999.99999v1</id></entry></feed>',
        '<!DOCTYPE feed [<!ENTITY a "bad">]><feed/>',
    ],
)
@respx.mock
async def test_invalid_arxiv_response_keeps_local_fields(body):
    respx.get("https://export.arxiv.org/api/query").respond(text=body)
    async with httpx.AsyncClient() as http:
        result = await MetadataService(http).enrich(
            pdf("arXiv:1706.03762", title="Local Title"), document()
        )
    assert result.title == "Local Title" and result.authors == [] and result.warnings


@respx.mock
async def test_provider_failure_and_model_failure_leave_unknown_fields_empty():
    respx.get("https://api.crossref.org/works/10.1234%2Fexample").respond(404)

    async def extractor(_):
        raise ValueError("private error")

    async with httpx.AsyncClient() as http:
        result = await MetadataService(http, extractor).enrich(
            pdf("10.1234/example", title="Local Title"), document()
        )
    assert result.title == "Local Title" and result.year is None and result.venue is None
    assert len(result.warnings) == 2 and "private" not in json.dumps(result.warnings)


async def test_complete_metadata_does_not_call_the_optional_model(monkeypatch):
    complete = PaperMetadata(
        title="A Paper", authors=["Alice"], year=2024, venue="Conference", abstract="Abstract"
    )
    monkeypatch.setattr("app.services.parsing.metadata.local_metadata", lambda *_: complete)

    async def unexpected(_):
        pytest.fail("Complete metadata must not invoke a paid model")

    async with httpx.AsyncClient() as http:
        assert await MetadataService(http, unexpected).enrich(pdf(), document()) == complete


async def test_skipped_model_preserves_local_fields_and_safe_reason():
    async def skipped(_):
        raise MetadataExtractionSkipped("今日模型 token 额度不足，已跳过快速模型元数据提取")

    async with httpx.AsyncClient() as http:
        result = await MetadataService(http, skipped).enrich(pdf(title="Local Title"), document())
    assert result.title == "Local Title" and result.year is None
    assert len(result.warnings) == 1 and "额度不足" in result.warnings[0]
