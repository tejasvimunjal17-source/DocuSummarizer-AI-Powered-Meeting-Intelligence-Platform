"""Chunking service: connects the Day 2 ingestion output to the chunking engine.

    file -> ingest_file -> IngestionResult.speaker_turns -> SpeakerAwareSplitter -> ChunkingResult

No LLM or network calls are made here.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence, Union

from chunking.chunk_models import ChunkingConfig, ChunkingResult
from chunking.speaker_aware_splitter import SpeakerAwareSplitter
from ingestion import (
    IngestionResult,
    SpeakerTurn,
    detect_speaker_turns,
    ingest_file,
    normalize_text,
)


def chunk_turns(
    turns: Sequence[SpeakerTurn],
    config: Optional[ChunkingConfig] = None,
    source_filename: Optional[str] = None,
) -> ChunkingResult:
    """Chunk already-detected speaker turns."""
    config = config or ChunkingConfig()
    started = time.perf_counter()
    output = SpeakerAwareSplitter(config).split(turns, source_filename)
    chunks = output.chunks
    non_empty = [t for t in turns if t.text.strip()]
    return ChunkingResult(
        config=config,
        source_filename=source_filename,
        chunks=chunks,
        total_chunks=len(chunks),
        source_turn_count=len(non_empty),
        split_turn_count=output.split_turn_count,
        source_character_count=sum(len(t.text) for t in non_empty),
        source_word_count=sum(len(t.text.split()) for t in non_empty),
        total_chunk_characters=sum(c.character_count for c in chunks),
        oversized_chunk_count=sum(1 for c in chunks if c.character_count > config.chunk_size),
        warnings=output.warnings,
        processing_seconds=time.perf_counter() - started,
    )


def chunk_text(
    text: str,
    config: Optional[ChunkingConfig] = None,
    source_filename: Optional[str] = None,
) -> ChunkingResult:
    """Normalize text, detect speaker turns, then chunk. Empty text gives no chunks."""
    normalized = normalize_text(text)
    turns = detect_speaker_turns(normalized) if normalized else []
    return chunk_turns(turns, config, source_filename)


def chunk_ingestion_result(
    result: IngestionResult, config: Optional[ChunkingConfig] = None
) -> ChunkingResult:
    """Chunk the output of the Day 2 ingestion service."""
    return chunk_turns(result.speaker_turns, config, result.filename)


def chunk_file(
    path: Union[str, Path], config: Optional[ChunkingConfig] = None
) -> ChunkingResult:
    """Convenience: ingest a TXT/DOCX/PDF file and chunk it."""
    return chunk_ingestion_result(ingest_file(path), config)
