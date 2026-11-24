"""Exceptions raised by the ingestion layer.

Every ingestion failure derives from IngestionError so callers (and the
Day 7 Streamlit UI) can catch one type and show a friendly message.
"""


class IngestionError(Exception):
    """Base class for all ingestion problems."""


class UnsupportedFileFormatError(IngestionError):
    """The file extension is not .txt, .docx or .pdf."""


class InvalidDocumentError(IngestionError):
    """The file is corrupted, unreadable, password-protected or not text."""


class EmptyDocumentError(IngestionError):
    """The file was read successfully but contains no text."""


class OCRRequiredError(EmptyDocumentError):
    """A PDF has pages but no extractable text (probably a scan).

    OCR is NOT part of the mandatory project scope, so such files are rejected
    instead of being silently returned as empty.
    """
