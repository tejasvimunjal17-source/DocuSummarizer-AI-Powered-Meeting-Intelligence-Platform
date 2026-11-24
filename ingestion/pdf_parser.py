"""PDF transcript parser (PyMuPDF).

Pages are read in order. A PDF with pages but no text layer (a scan) raises
OCRRequiredError: OCR is not part of the mandatory scope.
"""
from __future__ import annotations

from ingestion.base_parser import BaseParser, ParsedDocument
from ingestion.errors import (
    EmptyDocumentError,
    IngestionError,
    InvalidDocumentError,
    OCRRequiredError,
)


def _load_fitz():
    """Import PyMuPDF (new name 'pymupdf', old name 'fitz')."""
    try:
        import pymupdf as fitz
        return fitz
    except ImportError:
        pass
    try:
        import fitz
        return fitz
    except ImportError as exc:
        raise IngestionError(
            "PyMuPDF is not installed. Run: pip install -r requirements.txt"
        ) from exc


class PdfParser(BaseParser):
    def parse_bytes(self, data: bytes) -> ParsedDocument:
        if not data:
            raise EmptyDocumentError("The PDF file is empty (0 bytes).")

        fitz = _load_fitz()
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise InvalidDocumentError(
                "Could not open the file as a PDF. It may be corrupted or may "
                "not be a real PDF."
            ) from exc

        try:
            if getattr(doc, "needs_pass", False):
                raise InvalidDocumentError(
                    "The PDF is password-protected. Remove the password and try again."
                )
            page_count = len(doc)
            if page_count == 0:
                raise EmptyDocumentError("The PDF has no pages.")

            page_texts = []
            for page_number in range(page_count):
                page_texts.append(doc[page_number].get_text("text") or "")
        except IngestionError:
            raise
        except Exception as exc:
            raise InvalidDocumentError(f"Failed to read the PDF: {exc}") from exc
        finally:
            try:
                doc.close()
            except Exception:
                pass

        empty_pages = [i + 1 for i, t in enumerate(page_texts) if not t.strip()]
        if len(empty_pages) == page_count:
            raise OCRRequiredError(
                "The PDF has no extractable text. It is probably a scanned "
                "document and needs OCR, which is not supported yet."
            )

        warnings = []
        if empty_pages:
            pages = ", ".join(str(n) for n in empty_pages)
            warnings.append(
                f"Page(s) {pages} contain no extractable text (possibly scanned "
                "images) and were skipped."
            )

        text = "\n".join(t.rstrip("\n") for t in page_texts if t.strip())
        return ParsedDocument(text=text, page_count=page_count, warnings=warnings)
