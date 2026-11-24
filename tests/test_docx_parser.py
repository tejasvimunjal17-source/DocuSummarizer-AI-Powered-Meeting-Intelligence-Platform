"""Unit tests for the DOCX parser (uses real python-docx)."""
import pytest

docx = pytest.importorskip("docx")

from ingestion.docx_parser import DocxParser  # noqa: E402
from ingestion.errors import EmptyDocumentError, InvalidDocumentError  # noqa: E402


def build_docx(path, paragraphs):
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(str(path))
    return path


def test_valid_docx_extraction(tmp_path):
    path = build_docx(
        tmp_path / "a.docx",
        ["Rahul: We should launch.", "Priya: I will prepare."],
    )
    result = DocxParser().parse_file(path)
    assert result.text == "Rahul: We should launch.\nPriya: I will prepare."
    assert result.warnings == []


def test_multiple_paragraphs_keep_order(tmp_path):
    paragraphs = [f"Paragraph number {i}" for i in range(10)]
    path = build_docx(tmp_path / "b.docx", paragraphs)
    assert DocxParser().parse_file(path).text.split("\n") == paragraphs


def test_blank_paragraph_between_turns_is_kept(tmp_path):
    path = build_docx(tmp_path / "c.docx", ["Rahul: Hi", "", "Priya: Hello"])
    assert DocxParser().parse_file(path).text == "Rahul: Hi\n\nPriya: Hello"


def test_unicode_content(tmp_path):
    text = "Priya: Caf\u00e9 \u2014 \u0928\u092e\u0938\u094d\u0924\u0947 \u4f60\u597d"
    path = build_docx(tmp_path / "d.docx", [text])
    assert DocxParser().parse_file(path).text == text


def test_line_break_inside_paragraph_is_kept(tmp_path):
    path = build_docx(tmp_path / "e.docx", ["line one\nline two"])
    assert DocxParser().parse_file(path).text == "line one\nline two"


def test_empty_docx_rejected(tmp_path):
    path = build_docx(tmp_path / "empty.docx", [])
    with pytest.raises(EmptyDocumentError):
        DocxParser().parse_file(path)


def test_zero_byte_docx_rejected():
    with pytest.raises(EmptyDocumentError):
        DocxParser().parse_bytes(b"")


def test_invalid_docx_rejected():
    with pytest.raises(InvalidDocumentError):
        DocxParser().parse_bytes(b"this is definitely not a docx file")


def test_table_text_extracted_in_order_with_warning(tmp_path):
    document = docx.Document()
    document.add_paragraph("Before")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    document.add_paragraph("After")
    path = tmp_path / "table.docx"
    document.save(str(path))

    result = DocxParser().parse_file(path)
    assert result.text.split("\n") == ["Before", "A\tB", "After"]
    assert any("table" in w.lower() for w in result.warnings)
