"""Chunk preview: inspect chunks without printing a whole transcript."""
from __future__ import annotations

from typing import List, Sequence

from pydantic import BaseModel, Field

from chunking.chunk_models import ChunkingResult, TextChunk

DEFAULT_MAX_CHUNKS = 5
DEFAULT_PREVIEW_CHARS = 120


class ChunkPreview(BaseModel):
    chunk_index: int
    chunk_id: str
    character_count: int
    word_count: int
    speakers: List[str] = Field(default_factory=list)
    is_continuation: bool = False
    overlap_character_count: int = 0
    preview: str = Field(description="Start of the chunk (whitespace collapsed).")


def _shorten(text: str, max_chars: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[:max_chars].rstrip() + "\u2026"


def build_chunk_previews(
    chunks: Sequence[TextChunk],
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> List[ChunkPreview]:
    """Previews for the first max_chunks chunks (display only; chunks are unchanged)."""
    if max_chunks < 0 or preview_chars < 1:
        raise ValueError("max_chunks must be >= 0 and preview_chars must be >= 1.")
    return [
        ChunkPreview(
            chunk_index=c.chunk_index,
            chunk_id=c.chunk_id,
            character_count=c.character_count,
            word_count=c.word_count,
            speakers=list(c.speakers),
            is_continuation=c.is_continuation,
            overlap_character_count=c.overlap_character_count,
            preview=_shorten(c.text, preview_chars),
        )
        for c in chunks[:max_chunks]
    ]


def format_chunk_preview(
    result: ChunkingResult,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> str:
    """Human-readable preview text."""
    lines = [
        f"Total chunks: {result.total_chunks} | Source turns: {result.source_turn_count} "
        f"| Turns divided: {result.split_turn_count} "
        f"| Chunk size: {result.config.chunk_size} characters "
        f"| Overlap: {result.config.chunk_overlap} characters"
    ]
    if not result.chunks:
        lines.append("No chunks (the transcript is empty).")
        return "\n".join(lines)

    previews = build_chunk_previews(result.chunks, max_chunks, preview_chars)
    lines.append(f"Showing {len(previews)} of {result.total_chunks} chunks")
    for p in previews:
        speakers = ", ".join(p.speakers) if p.speakers else "unknown"
        lines.append(
            f"[{p.chunk_index}] {p.character_count} chars, {p.word_count} words | "
            f"speakers: {speakers} | continuation: {'yes' if p.is_continuation else 'no'} "
            f"| overlap: {p.overlap_character_count} chars"
        )
        lines.append(f"    {p.preview}")
    return "\n".join(lines)


def print_chunk_preview(
    result: ChunkingResult,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> None:
    """Print the preview (never the whole transcript)."""
    print(format_chunk_preview(result, max_chunks, preview_chars))
