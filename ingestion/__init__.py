"""Document ingestion layer (Day 2): TXT/DOCX/PDF -> clean text + speaker turns."""
from ingestion.errors import (
    EmptyDocumentError,
    IngestionError,
    InvalidDocumentError,
    OCRRequiredError,
    UnsupportedFileFormatError,
)
from ingestion.ingestion_service import (
    SUPPORTED_EXTENSIONS,
    detect_file_format,
    ingest_bytes,
    ingest_file,
)
from ingestion.models import FileFormat, IngestionResult, IngestionStatus, SpeakerTurn
from ingestion.speaker_detector import detect_speaker_turns, get_unique_speakers
from ingestion.text_normalizer import normalize_text

__all__ = [
    "EmptyDocumentError",
    "FileFormat",
    "IngestionError",
    "IngestionResult",
    "IngestionStatus",
    "InvalidDocumentError",
    "OCRRequiredError",
    "SUPPORTED_EXTENSIONS",
    "SpeakerTurn",
    "UnsupportedFileFormatError",
    "detect_file_format",
    "detect_speaker_turns",
    "get_unique_speakers",
    "ingest_bytes",
    "ingest_file",
    "normalize_text",
]
