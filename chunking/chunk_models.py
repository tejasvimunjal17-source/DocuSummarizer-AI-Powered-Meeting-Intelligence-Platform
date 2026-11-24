"""Pydantic models for the chunking engine (Day 3).

IMPORTANT: chunk_size and chunk_overlap are measured in CHARACTERS of the
rendered chunk text (labels such as "[10:00 AM] Alice: " included). They are
NOT token counts. Token counting arrives with the token budget manager (Day 6).
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from chunking.text_utils import label_prefix

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50


class ChunkingConfig(BaseModel):
    """Chunking settings (characters, not tokens)."""

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        ge=MIN_CHUNK_SIZE,
        description="Maximum characters per chunk (rendered text, labels included).",
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        ge=0,
        description="Maximum characters repeated from the end of the previous chunk "
        "(whole speaker turns/fragments only).",
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        if self.chunk_overlap > self.chunk_size // 2:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) is more than half of "
                f"chunk_size ({self.chunk_size}); this would duplicate too much text."
            )
        return self

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        load_dotenv_file: bool = True,
    ) -> "ChunkingConfig":
        """Read CHUNK_SIZE and CHUNK_OVERLAP from the environment (or .env)."""
        if env is None:
            if load_dotenv_file:
                load_dotenv()
            env = os.environ

        def read(name: str, default: int) -> int:
            raw = env.get(name, "").strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a whole number, got '{raw}'.") from exc

        return cls(
            chunk_size=read("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
            chunk_overlap=read("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        )


class ChunkSegment(BaseModel):
    """A statement (or fragment of a long statement) inside a chunk."""

    turn_index: int = Field(ge=0, description="SpeakerTurn.index from ingestion.")
    speaker: Optional[str] = Field(
        default=None, description="Speaker as written; None when unknown."
    )
    timestamp: Optional[str] = None
    text: str
    fragment_index: int = Field(
        default=0, ge=0, description="0 for a whole turn or the first fragment."
    )
    fragment_count: int = Field(
        default=1, ge=1, description="How many fragments the turn was divided into."
    )
    is_overlap: bool = Field(
        default=False, description="True if repeated from the previous chunk."
    )
    ends_mid_sentence: bool = Field(
        default=False,
        description="True if this fragment ends inside a sentence (a single "
        "sentence was longer than the size limit).",
    )

    @property
    def is_continuation(self) -> bool:
        """True if this continues a turn that began in an earlier fragment."""
        return self.fragment_index > 0

    def rendered_text(self) -> str:
        """Text as written into the chunk: label + statement."""
        return label_prefix(self.speaker, self.timestamp) + self.text


class TextChunk(BaseModel):
    """One chunk of the transcript with metadata."""

    chunk_id: str = Field(description="Predictable id, e.g. 'chunk-0003'.")
    chunk_index: int = Field(ge=0)
    text: str
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0, description="Whitespace-separated words.")
    speakers: List[str] = Field(
        default_factory=list, description="Unique known speakers, in order."
    )
    speaker_turn_ids: List[int] = Field(
        default_factory=list, description="Unique SpeakerTurn.index values, in order."
    )
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    is_continuation: bool = Field(
        default=False,
        description="True if this chunk's first NEW content continues a turn "
        "that started in an earlier chunk.",
    )
    overlap_character_count: int = Field(
        default=0, ge=0, description="Characters at the start repeated from the previous chunk."
    )
    source_filename: Optional[str] = None
    segments: List[ChunkSegment] = Field(default_factory=list)


class ChunkingResult(BaseModel):
    """All chunks plus processing statistics."""

    config: ChunkingConfig
    source_filename: Optional[str] = None
    chunks: List[TextChunk] = Field(default_factory=list)
    total_chunks: int = Field(default=0, ge=0)
    source_turn_count: int = Field(default=0, ge=0)
    split_turn_count: int = Field(
        default=0, ge=0, description="Turns divided into more than one fragment."
    )
    source_character_count: int = Field(default=0, ge=0)
    source_word_count: int = Field(default=0, ge=0)
    total_chunk_characters: int = Field(
        default=0, ge=0, description="Sum of chunk lengths (includes overlap and labels)."
    )
    oversized_chunk_count: int = Field(default=0, ge=0)
    warnings: List[str] = Field(default_factory=list)
    processing_seconds: float = Field(default=0.0, ge=0)
