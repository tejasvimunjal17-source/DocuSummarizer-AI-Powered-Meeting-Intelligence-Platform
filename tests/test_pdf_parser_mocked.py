"""PDF parser logic tests with a FAKE PyMuPDF (mocked, no real PDF involved).

These check our own logic (page order, warnings, errors). They do NOT prove
that real PyMuPDF extraction works; see test_pdf_parser.py for that.
"""
import pytest

from ingestion import pdf_parser
from ingestion.errors import (
    EmptyDocumentError,
    InvalidDocumentError,
    OCRRequiredError,
)


class FakePage:
    def __init__(self, text, fail=False):
        self._text = text
        self._fail = fail

    def get_text(self, mode):
        if self._fail:
            raise RuntimeError("boom")
        return self._text


class FakeDoc:
    def __init__(self, pages, needs_pass=False):
        self._pages = pages
        self.needs_pass = needs_pass
        self.closed = False

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        self.closed = True


class FakeFitz:
    def __init__(self, doc=None, error=None):
        self.doc = doc
        self.error = error

    def open(self, stream=None, filetype=None):
        if self.error:
            raise self.error
        return self.doc


def use_fake(monkeypatch, doc=None, error=None):
    monkeypatch.setattr(pdf_parser, "_load_fitz", lambda: FakeFitz(doc, error))


def test_pages_joined_in_order(monkeypatch):
    doc = FakeDoc([FakePage("one\n"), FakePage("two\n"), FakePage("three\n")])
    use_fake(monkeypatch, doc)
    result = pdf_parser.PdfParser().parse_bytes(b"x")
    assert result.text == "one\ntwo\nthree"
    assert result.page_count == 3
    assert doc.closed


def test_all_blank_pages_require_ocr(monkeypatch):
    use_fake(monkeypatch, FakeDoc([FakePage(" \n"), FakePage("")]))
    with pytest.raises(OCRRequiredError):
        pdf_parser.PdfParser().parse_bytes(b"x")


def test_some_blank_pages_warn_with_page_numbers(monkeypatch):
    use_fake(monkeypatch, FakeDoc([FakePage("a"), FakePage(""), FakePage("c")]))
    result = pdf_parser.PdfParser().parse_bytes(b"x")
    assert result.text == "a\nc"
    assert "2" in result.warnings[0]


def test_zero_pages_is_empty_document(monkeypatch):
    use_fake(monkeypatch, FakeDoc([]))
    with pytest.raises(EmptyDocumentError):
        pdf_parser.PdfParser().parse_bytes(b"x")


def test_password_protected_pdf_rejected(monkeypatch):
    use_fake(monkeypatch, FakeDoc([FakePage("secret")], needs_pass=True))
    with pytest.raises(InvalidDocumentError, match="password"):
        pdf_parser.PdfParser().parse_bytes(b"x")


def test_open_failure_becomes_invalid_document(monkeypatch):
    use_fake(monkeypatch, error=RuntimeError("bad file"))
    with pytest.raises(InvalidDocumentError):
        pdf_parser.PdfParser().parse_bytes(b"x")


def test_page_read_failure_becomes_invalid_document(monkeypatch):
    doc = FakeDoc([FakePage("a"), FakePage("b", fail=True)])
    use_fake(monkeypatch, doc)
    with pytest.raises(InvalidDocumentError):
        pdf_parser.PdfParser().parse_bytes(b"x")
    assert doc.closed
