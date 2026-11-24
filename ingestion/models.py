"""Pydantic models describing the output of the ingestion layer.

These are separate from schemas/extraction_models.py (Day 1), which describes
what the LLM extracts. These describe the cleaned transcript that Day 3's
chunking engine will consume.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class FileFormat(str, Enum):
    """Supported input formats."""

    TXT = "txt"
    DOCX = "docx"
    PDF = "pdf"


class IngestionStatus(str, Enum):
    """Outcome of a successful ingestion. Failures raise IngestionError."""

    SUCCESS = "success"
    SUCCESS_WITH_WARNINGS = "success_with_warnings"


class SpeakerTurn(BaseModel):
    """One continuous statement by one speaker.

    speaker is None when the speaker is unknown. We never guess a name.
    """

    index: int = Field(ge=0, description="Position of the turn in the transcript.")
    speaker: Optional[str] = Field(
        default=None, description="Speaker name as written, or None if unknown."
    )
    timestamp: Optional[str] = Field(
        default=None, description="Timestamp as written (e.g. '10:30'), if any."
    )
    text: str = Field(description="The statement, lines joined with newlines.")


class IngestionResult(BaseModel):
    """Everything the ingestion layer produces for one file."""

    filename: str
    file_format: FileFormat
    raw_text: str = Field(description="Text exactly as extracted (before cleanup).")
    normalized_text: str = Field(description="Cleaned text used for later steps.")
    speaker_turns: List[SpeakerTurn] = Field(default_factory=list)
    speakers: List[str] = Field(
        default_factory=list, description="Unique speaker names, in order of appearance."
    )
    speaker_count: int = Field(default=0, ge=0)
    has_speaker_labels: bool = False
    page_count: Optional[int] = Field(
        default=None, description="Number of pages (PDF only)."
    )
    word_count: int = Field(default=0, ge=0)
    status: IngestionStatus = IngestionStatus.SUCCESS
    warnings: List[str] = Field(default_factory=list)
