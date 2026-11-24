"""Unit tests for the PDF parser using REAL PyMuPDF.

The whole module is skipped automatically if PyMuPDF is not installed.
"""
import pytest

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF versions use the name "fitz"
    fitz = pytest.importorskip("fitz")

from ingestion.errors import (  # noqa: E402
    EmptyDocumentError,
    InvalidDocumentError,
    OCRRequiredError,
)
from ingestion.pdf_parser import PdfParser  # noqa: E402


def build_pdf(pages):
    """Create a PDF in memory. An empty string makes a blank page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_valid_pdf_extraction():
    result = PdfParser().parse_bytes(build_pdf(["Rahul: Hello there"]))
    assert "Rahul: Hello there" in result.text
    assert result.page_count == 1
    assert result.warnings == []


def test_multi_page_pdf_keeps_page_order():
    result = PdfParser().parse_bytes(
        build_pdf(["Page one text", "Page two text", "Page three text"])
    )
    text = result.text
    assert result.page_count == 3
    assert text.index("Page one") < text.index("Page two") < text.index("Page three")


def test_multi_line_text_extracted():
    result = PdfParser().parse_bytes(build_pdf(["Rahul: Hello\nPriya: Hi"]))
    assert "Rahul: Hello" in result.text
    assert "Priya: Hi" in result.text


def test_some_blank_pages_add_warning():
    result = PdfParser().parse_bytes(build_pdf(["First page", "", "Third page"]))
    assert "First page" in result.text and "Third page" in result.text
    assert any("2" in w for w in result.warnings)


def test_pdf_without_text_requires_ocr():
    with pytest.raises(OCRRequiredError):
        PdfParser().parse_bytes(build_pdf(["", ""]))


def test_ocr_error_is_also_an_empty_document_error():
    with pytest.raises(EmptyDocumentError):
        PdfParser().parse_bytes(build_pdf([""]))


def test_zero_byte_pdf_rejected():
    with pytest.raises(EmptyDocumentError):
        PdfParser().parse_bytes(b"")


def test_invalid_pdf_rejected():
    with pytest.raises(InvalidDocumentError):
        PdfParser().parse_bytes(b"this is not a pdf at all")
