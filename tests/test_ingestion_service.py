"""Integration tests: file -> parser -> normalizer -> speaker turns."""
import json

import pytest

from ingestion import (
    EmptyDocumentError,
    FileFormat,
    IngestionStatus,
    InvalidDocumentError,
    OCRRequiredError,
    UnsupportedFileFormatError,
    detect_file_format,
    ingest_bytes,
    ingest_file,
)
from ingestion import pdf_parser

TRANSCRIPT = (
    "[10:30] Rahul: We need to finalize the budget.\r\n\r\n"
    "[10:32] Priya: I agree  with that decision.\r\n"
)


def test_txt_through_full_service(tmp_path):
    path = tmp_path / "meeting.txt"
    path.write_bytes(TRANSCRIPT.encode("utf-8"))
    result = ingest_file(path)

    assert result.filename == "meeting.txt"
    assert result.file_format == FileFormat.TXT
    assert "\r" in result.raw_text and "\r" not in result.normalized_text
    assert "agree with that" in result.normalized_text  # double space fixed
    assert result.speakers == ["Rahul", "Priya"]
    assert result.speaker_count == 2
    assert result.has_speaker_labels is True
    assert [t.timestamp for t in result.speaker_turns] == ["10:30", "10:32"]
    assert result.status == IngestionStatus.SUCCESS
    assert result.warnings == []
    assert result.word_count == len(result.normalized_text.split())


def test_docx_through_full_service(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Rahul: We should launch next month.")
    document.add_paragraph("Priya: I will prepare the presentation.")
    path = tmp_path / "meeting.docx"
    document.save(str(path))

    result = ingest_file(path)
    assert result.file_format == FileFormat.DOCX
    assert result.speakers == ["Rahul", "Priya"]
    assert result.page_count is None


def test_pdf_through_full_service_real_pymupdf(tmp_path):
    try:
        import pymupdf as fitz
    except ImportError:
        fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Rahul: We should launch.\nPriya: I will prepare.")
    path = tmp_path / "meeting.pdf"
    path.write_bytes(doc.tobytes())
    doc.close()

    result = ingest_file(path)
    assert result.file_format == FileFormat.PDF
    assert result.page_count == 1
    assert result.speakers == ["Rahul", "Priya"]


def test_pdf_through_full_service_with_fake_pymupdf(monkeypatch):
    """MOCKED: checks service wiring only, not real PDF extraction."""

    class Page:
        def get_text(self, mode):
            return "Rahul: Hello\nPriya: Hi\n"

    class Doc:
        needs_pass = False

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return Page()

        def close(self):
            pass

    class Fitz:
        def open(self, stream=None, filetype=None):
            return Doc()

    monkeypatch.setattr(pdf_parser, "_load_fitz", lambda: Fitz())
    result = ingest_bytes("m.pdf", b"x")
    assert result.file_format == FileFormat.PDF
    assert result.speakers == ["Rahul", "Priya"]


def test_scanned_pdf_reports_ocr_required_with_fake_pymupdf(monkeypatch):
    """MOCKED: a PDF whose pages have no text must not succeed silently."""

    class Page:
        def get_text(self, mode):
            return ""

    class Doc:
        needs_pass = False

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return Page()

        def close(self):
            pass

    class Fitz:
        def open(self, stream=None, filetype=None):
            return Doc()

    monkeypatch.setattr(pdf_parser, "_load_fitz", lambda: Fitz())
    with pytest.raises(OCRRequiredError):
        ingest_bytes("scan.pdf", b"x")


@pytest.mark.parametrize("name", ["data.csv", "old.doc", "image.png", "noextension"])
def test_unsupported_format_rejected(name):
    with pytest.raises(UnsupportedFileFormatError, match="Supported formats"):
        detect_file_format(name)
    with pytest.raises(UnsupportedFileFormatError):
        ingest_bytes(name, b"data")


def test_extension_is_case_insensitive():
    assert detect_file_format("MEETING.TXT") == FileFormat.TXT
    assert detect_file_format("a.DocX") == FileFormat.DOCX
    assert detect_file_format("a.Pdf") == FileFormat.PDF


def test_invalid_docx_file_rejected(tmp_path):
    pytest.importorskip("docx")
    path = tmp_path / "fake.docx"
    path.write_text("I am plain text pretending to be a docx")
    with pytest.raises(InvalidDocumentError):
        ingest_file(path)


def test_empty_txt_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        ingest_file(path)


def test_text_that_is_only_invisible_characters_rejected():
    with pytest.raises(EmptyDocumentError):
        ingest_bytes("x.txt", "\u200b\u200b \u00a0\n".encode("utf-8"))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(InvalidDocumentError, match="not found"):
        ingest_file(tmp_path / "nope.txt")


def test_no_speaker_labels_gives_warning():
    result = ingest_bytes("notes.txt", b"Just some plain notes.\n\nMore notes.")
    assert result.speaker_count == 0
    assert result.has_speaker_labels is False
    assert result.status == IngestionStatus.SUCCESS_WITH_WARNINGS
    assert any("No speaker labels" in w for w in result.warnings)
    assert [t.speaker for t in result.speaker_turns] == [None, None]


def test_unlabeled_block_warning():
    result = ingest_bytes("m.txt", b"Intro text\n\nRahul: Hi\nPriya: Hello")
    assert result.speakers == ["Rahul", "Priya"]
    assert any("unknown speaker" in w for w in result.warnings)


def test_ingest_bytes_strips_directories_from_filename():
    result = ingest_bytes("folder/sub/m.txt", b"Rahul: Hi\nPriya: Hello")
    assert result.filename == "m.txt"


def test_long_transcript_warning():
    data = ("word " * 50_001).encode("utf-8")
    result = ingest_bytes("long.txt", data)
    assert result.word_count == 50_001
    assert any("50000" in w for w in result.warnings)


def test_result_is_json_serializable():
    result = ingest_bytes("m.txt", b"[10:30] Rahul: Hi\n[10:31] Priya: Hello")
    data = result.model_dump(mode="json")
    text = json.dumps(data)
    assert data["file_format"] == "txt"
    assert data["status"] == "success"
    assert json.loads(text)["speaker_turns"][0]["speaker"] == "Rahul"
