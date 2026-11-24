"""Speaker-aware chunking engine (Day 3)."""
from chunking.chunk_models import (
    ChunkSegment,
    ChunkingConfig,
    ChunkingResult,
    TextChunk,
)
from chunking.chunk_preview import (
    ChunkPreview,
    build_chunk_previews,
    format_chunk_preview,
    print_chunk_preview,
)
from chunking.chunk_validator import (
    ChunkValidationResult,
    IssueSeverity,
    ValidationIssue,
    validate_chunks,
)
from chunking.chunking_service import (
    chunk_file,
    chunk_ingestion_result,
    chunk_text,
    chunk_turns,
)
from chunking.errors import ChunkingError
from chunking.speaker_aware_splitter import SpeakerAwareSplitter

__all__ = [
    "ChunkPreview",
    "ChunkSegment",
    "ChunkValidationResult",
    "ChunkingConfig",
    "ChunkingError",
    "ChunkingResult",
    "IssueSeverity",
    "SpeakerAwareSplitter",
    "TextChunk",
    "ValidationIssue",
    "build_chunk_previews",
    "chunk_file",
    "chunk_ingestion_result",
    "chunk_text",
    "chunk_turns",
    "format_chunk_preview",
    "print_chunk_preview",
    "validate_chunks",
]
