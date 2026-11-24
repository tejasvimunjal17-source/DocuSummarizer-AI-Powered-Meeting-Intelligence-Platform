"""Validation of generated chunks. Returns a structured result; never raises
for ordinary problems and never "fixes" chunks silently."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from chunking.chunk_models import ChunkingConfig, TextChunk
from chunking.packing import Unit, rendered_length
from chunking.text_utils import SEGMENT_SEPARATOR, compact, count_words
from ingestion.models import SpeakerTurn


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    severity: IssueSeverity
    code: str = Field(description="Short machine-readable code, e.g. 'empty_chunk'.")
    message: str
    chunk_index: Optional[int] = None


class ChunkValidationResult(BaseModel):
    is_valid: bool = Field(description="True when there are no ERROR issues.")
    chunk_count: int = 0
    issues: List[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    def codes(self) -> List[str]:
        return [i.code for i in self.issues]


def _unit_of(segment) -> Unit:
    return Unit(
        segment.turn_index,
        segment.speaker,
        segment.timestamp,
        segment.text,
        segment.fragment_index,
        segment.fragment_count,
    )


def validate_chunks(
    chunks: Sequence[TextChunk],
    config: ChunkingConfig,
    source_turns: Optional[Sequence[SpeakerTurn]] = None,
) -> ChunkValidationResult:
    """Check chunks against the rules in the Day 3 brief.

    If source_turns is given, also checks that no transcript content was lost,
    duplicated or re-attributed to another speaker.
    """
    issues: List[ValidationIssue] = []

    def add(severity, code, message, index=None):
        issues.append(ValidationIssue(severity=severity, code=code, message=message, chunk_index=index))

    def error(code, message, index=None):
        add(IssueSeverity.ERROR, code, message, index)

    # --- settings
    if config.chunk_overlap >= config.chunk_size:
        error(
            "invalid_overlap",
            f"chunk_overlap ({config.chunk_overlap}) must be smaller than "
            f"chunk_size ({config.chunk_size}).",
        )
    elif config.chunk_overlap > config.chunk_size // 2:
        error(
            "excessive_overlap",
            f"chunk_overlap ({config.chunk_overlap}) is more than half of "
            f"chunk_size ({config.chunk_size}).",
        )

    previous_new: List = []
    last_position = None  # (turn_index, fragment_index) of last NEW segment
    new_by_turn: Dict[int, List] = {}

    for position, chunk in enumerate(chunks):
        idx = position

        # --- index
        if chunk.chunk_index != position:
            error(
                "invalid_index",
                f"Chunk at position {position} has chunk_index {chunk.chunk_index}.",
                idx,
            )

        # --- empty
        if not chunk.text.strip():
            error("empty_chunk", "Chunk text is empty.", idx)
            continue
        if not chunk.segments:
            error("no_segments", "Chunk has no segments.", idx)
            continue

        segments = chunk.segments
        new_segments = [s for s in segments if not s.is_overlap]
        overlap_segments = [s for s in segments if s.is_overlap]

        # --- text consistency
        expected_text = SEGMENT_SEPARATOR.join(s.rendered_text() for s in segments)
        if chunk.text != expected_text:
            error("text_mismatch", "Chunk text does not match its segments.", idx)

        # --- counts
        if chunk.character_count != len(chunk.text):
            error(
                "wrong_character_count",
                f"character_count is {chunk.character_count} but the text has "
                f"{len(chunk.text)} characters.",
                idx,
            )
        if chunk.word_count != count_words(chunk.text):
            error(
                "wrong_word_count",
                f"word_count is {chunk.word_count} but the text has "
                f"{count_words(chunk.text)} words.",
                idx,
            )

        # --- speaker metadata
        expected_speakers: List[str] = []
        for s in segments:
            if s.speaker is not None and s.speaker not in expected_speakers:
                expected_speakers.append(s.speaker)
        if expected_speakers and not chunk.speakers:
            error("missing_speaker_metadata", "Chunk has speakers but its speaker list is empty.", idx)
        elif chunk.speakers != expected_speakers:
            error(
                "invalid_speaker_metadata",
                f"speakers {chunk.speakers} do not match the segments {expected_speakers}.",
                idx,
            )
        for name in chunk.speakers:
            if name not in chunk.text:
                error("speaker_not_in_text", f"Speaker '{name}' is not present in the chunk text.", idx)

        expected_turn_ids: List[int] = []
        for s in segments:
            if s.turn_index not in expected_turn_ids:
                expected_turn_ids.append(s.turn_index)
        if chunk.speaker_turn_ids != expected_turn_ids:
            error("invalid_turn_ids", "speaker_turn_ids do not match the segments.", idx)

        # --- timestamps
        stamps = [s.timestamp for s in segments if s.timestamp]
        if chunk.start_timestamp != (stamps[0] if stamps else None) or chunk.end_timestamp != (
            stamps[-1] if stamps else None
        ):
            error("invalid_timestamps", "Start/end timestamps do not match the segments.", idx)

        # --- size
        if len(chunk.text) > config.chunk_size:
            if len(segments) == 1:
                add(
                    IssueSeverity.WARNING,
                    "oversized_unsplittable",
                    f"Chunk has {len(chunk.text)} characters (limit {config.chunk_size}) "
                    "because a single piece of text could not be divided safely. "
                    "The text was kept in full.",
                    idx,
                )
            else:
                error(
                    "chunk_too_large",
                    f"Chunk has {len(chunk.text)} characters, above the limit of "
                    f"{config.chunk_size}.",
                    idx,
                )

        # --- overlap structure
        if not new_segments:
            error("no_new_content", "Chunk contains only overlap text.", idx)
            continue
        first_new = segments.index(new_segments[0])
        if any(not s.is_overlap for s in segments[:first_new]) or any(
            s.is_overlap for s in segments[first_new:]
        ):
            error("overlap_not_at_start", "Overlap segments must come first.", idx)
        if position == 0 and overlap_segments:
            error("unexpected_overlap", "The first chunk cannot have overlap.", idx)
        if overlap_segments:
            overlap_units = [_unit_of(s) for s in overlap_segments]
            if rendered_length(overlap_units) > config.chunk_overlap:
                error(
                    "excessive_overlap_content",
                    f"Overlap is {rendered_length(overlap_units)} characters, above "
                    f"chunk_overlap ({config.chunk_overlap}).",
                    idx,
                )
            tail = previous_new[-len(overlap_segments):] if previous_new else []
            if [(s.turn_index, s.fragment_index) for s in overlap_segments] != [
                (s.turn_index, s.fragment_index) for s in tail
            ]:
                error(
                    "invalid_overlap_segments",
                    "Overlap does not repeat the end of the previous chunk.",
                    idx,
                )
        if chunk.overlap_character_count != rendered_length([_unit_of(s) for s in overlap_segments]):
            error("wrong_overlap_count", "overlap_character_count is inaccurate.", idx)

        # --- continuation flag
        if chunk.is_continuation != (new_segments[0].fragment_index > 0):
            error("invalid_continuation_flag", "is_continuation does not match the segments.", idx)

        # --- chronology
        for s in new_segments:
            position_key = (s.turn_index, s.fragment_index)
            if last_position is not None:
                prev_turn, prev_frag = last_position
                same_turn_next = s.turn_index == prev_turn and s.fragment_index == prev_frag + 1
                new_turn = s.turn_index > prev_turn and s.fragment_index == 0
                if not (same_turn_next or new_turn):
                    error(
                        "out_of_order",
                        f"Segment (turn {s.turn_index}, fragment {s.fragment_index}) "
                        f"follows (turn {prev_turn}, fragment {prev_frag}).",
                        idx,
                    )
            last_position = position_key
            new_by_turn.setdefault(s.turn_index, []).append(s)
        previous_new = new_segments

    # --- content preservation
    if source_turns is not None:
        source_by_index = {t.index: t for t in source_turns if t.text.strip()}
        for turn_index, turn in source_by_index.items():
            found = new_by_turn.get(turn_index)
            if not found:
                error("content_missing", f"Speaker turn {turn_index} is missing from the chunks.")
                continue
            if compact("".join(s.text for s in found)) != compact(turn.text):
                error(
                    "content_mismatch",
                    f"Text of speaker turn {turn_index} was lost, changed or duplicated.",
                )
            if any(s.speaker != turn.speaker or s.timestamp != turn.timestamp for s in found):
                error(
                    "attribution_mismatch",
                    f"Speaker or timestamp of turn {turn_index} changed during chunking.",
                )
        for turn_index in new_by_turn:
            if turn_index not in source_by_index:
                error("unknown_turn", f"Chunks contain turn {turn_index}, which is not in the source.")

    return ChunkValidationResult(
        is_valid=not any(i.severity == IssueSeverity.ERROR for i in issues),
        chunk_count=len(chunks),
        issues=issues,
    )
