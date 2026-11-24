"""Central ingestion service: file -> parser -> normalizer -> speaker turns.

Failures raise subclasses of IngestionError (see ingestion/errors.py).
Day 3's chunking engine should consume IngestionResult.speaker_turns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

from ingestion.base_parser import BaseParser, ParsedDocument
from ingestion.docx_parser import DocxParser
from ingestion.errors import EmptyDocumentError, UnsupportedFileFormatError
from ingestion.models import FileFormat, IngestionResult, IngestionStatus
from ingestion.pdf_parser import PdfParser
from ingestion.speaker_detector import detect_speaker_turns, get_unique_speakers
from ingestion.text_normalizer import normalize_text
from ingestion.txt_parser import TxtParser

SUPPORTED_EXTENSIONS: Dict[str, FileFormat] = {
    ".txt": FileFormat.TXT,
    ".docx": FileFormat.DOCX,
    ".pdf": FileFormat.PDF,
}

MAX_RECOMMENDED_WORDS = 50_000  # official project limit

_PARSERS: Dict[FileFormat, BaseParser] = {
    FileFormat.TXT: TxtParser(),
    FileFormat.DOCX: DocxParser(),
    FileFormat.PDF: PdfParser(),
}


def detect_file_format(filename: str) -> FileFormat:
    """Detect the format from the file extension (case-insensitive)."""
    extension = Path(filename).suffix.lower()
    file_format = SUPPORTED_EXTENSIONS.get(extension)
    if file_format is None:
        shown = extension or "(no extension)"
        raise UnsupportedFileFormatError(
            f"Unsupported file format {shown}. Supported formats: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
    return file_format


def ingest_file(path: Union[str, Path]) -> IngestionResult:
    """Ingest a file from disk."""
    file_path = Path(path)
    file_format = detect_file_format(file_path.name)
    parsed = _PARSERS[file_format].parse_file(file_path)
    return _build_result(file_path.name, file_format, parsed)


def ingest_bytes(filename: str, data: bytes) -> IngestionResult:
    """Ingest uploaded bytes (e.g. from the Streamlit uploader on Day 7)."""
    name = Path(filename).name
    file_format = detect_file_format(name)
    parsed = _PARSERS[file_format].parse_bytes(data)
    return _build_result(name, file_format, parsed)


def _build_result(
    filename: str, file_format: FileFormat, parsed: ParsedDocument
) -> IngestionResult:
    normalized = normalize_text(parsed.text)
    if not normalized:
        raise EmptyDocumentError("The document contains no readable text.")

    turns = detect_speaker_turns(normalized)
    speakers = get_unique_speakers(turns)
    word_count = len(normalized.split())

    warnings = list(parsed.warnings)
    if not speakers:
        warnings.append(
            "No speaker labels were detected. The text was kept, but every "
            "statement has an unknown speaker."
        )
    else:
        unknown = sum(1 for t in turns if t.speaker is None)
        if unknown:
            warnings.append(
                f"{unknown} block(s) of text have an unknown speaker "
                "(no label before them)."
            )
    if word_count > MAX_RECOMMENDED_WORDS:
        warnings.append(
            f"The transcript has {word_count} words, above the "
            f"{MAX_RECOMMENDED_WORDS}-word project limit."
        )

    return IngestionResult(
        filename=filename,
        file_format=file_format,
        raw_text=parsed.text,
        normalized_text=normalized,
        speaker_turns=turns,
        speakers=speakers,
        speaker_count=len(speakers),
        has_speaker_labels=bool(speakers),
        page_count=parsed.page_count,
        word_count=word_count,
        status=(
            IngestionStatus.SUCCESS_WITH_WARNINGS
            if warnings
            else IngestionStatus.SUCCESS
        ),
        warnings=warnings,
    )
