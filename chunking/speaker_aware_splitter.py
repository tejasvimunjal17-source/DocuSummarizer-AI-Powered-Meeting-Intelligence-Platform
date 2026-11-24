"""Speaker-aware chunking built on LangChain's RecursiveCharacterTextSplitter.

How the work is divided
-----------------------
* Whole-turn packing and turn-level overlap are done by packing.py, because the
  LangChain splitter does not know about speakers.
* A turn that is too long for one chunk is divided by the REAL
  RecursiveCharacterTextSplitter. Its separators are tried in this order:
      blank line -> line break -> sentence end -> clause end (, ; :) -> space
  so a long turn is cut at sentence boundaries whenever possible.
* There is deliberately NO empty-string separator, so a word is never cut in half.

Every fragment of a divided turn keeps the speaker name and timestamp (the label
is repeated at the start of each fragment), and is flagged as a continuation.

Documented limitations
----------------------
* A single sentence longer than the limit is cut at a clause or word boundary;
  the fragment is flagged ends_mid_sentence=True and a warning is recorded.
* A single "word" longer than the limit (for example a long URL) cannot be cut;
  its chunk exceeds chunk_size and is reported, never truncated.
* The sentence separator also fires after abbreviations such as "Dr." (only
  while dividing turns that are already too long).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from chunking.chunk_models import ChunkingConfig, ChunkSegment, TextChunk
from chunking.errors import ChunkingError
from chunking.packing import PackedChunk, Unit, pack_units, rendered_length
from chunking.text_utils import (
    SEGMENT_SEPARATOR,
    compact,
    count_words,
    ends_with_sentence_boundary,
    label_prefix,
)
from ingestion.models import SpeakerTurn

# Regex separators for RecursiveCharacterTextSplitter (is_separator_regex=True).
SEPARATORS = [
    r"\n\n",
    r"\n",
    r"(?<=[.!?\u2026\u3002\uff01\uff1f\u0964])\s+",
    r"(?<=[,;:])\s+",
    r"\s+",
]

# Smallest fragment size used when a long label leaves almost no room.
MIN_FRAGMENT_CHARS = 20


def _splitter_class():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:  # older LangChain versions
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ChunkingError(
                "LangChain text splitters are not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
    return RecursiveCharacterTextSplitter


def create_turn_splitter(max_characters: int):
    """Create the real RecursiveCharacterTextSplitter used for long turns."""
    return _splitter_class()(
        chunk_size=max_characters,
        chunk_overlap=0,  # overlap is handled once, at chunk level (packing.py)
        length_function=len,  # characters, not tokens
        separators=SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
        strip_whitespace=True,
    )


@dataclass
class SplitOutput:
    chunks: List[TextChunk] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    split_turn_count: int = 0


class SpeakerAwareSplitter:
    """Turns a list of SpeakerTurn objects into TextChunk objects."""

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
        self._splitters: Dict[int, object] = {}

    def split(
        self,
        turns: Sequence[SpeakerTurn],
        source_filename: Optional[str] = None,
    ) -> SplitOutput:
        warnings: List[str] = []
        units, split_turn_count = self._build_units(turns, warnings)
        packed = pack_units(
            units, self.config.chunk_size, self.config.chunk_overlap
        )
        chunks = [
            self._build_chunk(i, p, source_filename) for i, p in enumerate(packed)
        ]
        for chunk in chunks:
            if chunk.character_count > self.config.chunk_size:
                warnings.append(
                    f"Chunk {chunk.chunk_index} has {chunk.character_count} characters, "
                    f"above the limit of {self.config.chunk_size}: it contains a piece "
                    "of text that cannot be divided safely. Nothing was truncated."
                )
        return SplitOutput(chunks, warnings, split_turn_count)

    # ------------------------------------------------------------------ units
    def _splitter_for(self, limit: int):
        if limit not in self._splitters:
            self._splitters[limit] = create_turn_splitter(limit)
        return self._splitters[limit]

    def _build_units(self, turns: Sequence[SpeakerTurn], warnings: List[str]):
        units: List[Unit] = []
        split_turn_count = 0
        size = self.config.chunk_size
        for turn in turns:
            text = turn.text.strip()
            if not text:
                continue
            prefix_length = len(label_prefix(turn.speaker, turn.timestamp))
            if prefix_length + len(text) <= size:
                units.append(Unit(turn.index, turn.speaker, turn.timestamp, text))
                continue

            limit = max(size - prefix_length, MIN_FRAGMENT_CHARS)
            fragments = self._splitter_for(limit).split_text(text)
            if not fragments or compact("".join(fragments)) != compact(text):
                raise ChunkingError(
                    f"Splitting speaker turn {turn.index} would lose text; aborting."
                )
            count = len(fragments)
            who = turn.speaker or "unknown speaker"
            if count > 1:
                split_turn_count += 1
            mid_sentence_turn = False
            oversize_piece = False
            for i, fragment in enumerate(fragments):
                is_last = i == count - 1
                mid = (not is_last) and not ends_with_sentence_boundary(fragment)
                mid_sentence_turn = mid_sentence_turn or mid
                oversize_piece = oversize_piece or len(fragment) > limit
                units.append(
                    Unit(
                        turn_index=turn.index,
                        speaker=turn.speaker,
                        timestamp=turn.timestamp,
                        text=fragment,
                        fragment_index=i,
                        fragment_count=count,
                        ends_mid_sentence=mid,
                    )
                )
            if mid_sentence_turn:
                warnings.append(
                    f"Turn {turn.index} ({who}): a sentence longer than the size limit "
                    "was divided at a clause or word boundary."
                )
            if oversize_piece:
                warnings.append(
                    f"Turn {turn.index} ({who}): contains text with no spaces that is "
                    "longer than the size limit and could not be divided."
                )
        return units, split_turn_count

    # ----------------------------------------------------------------- chunks
    def _build_chunk(
        self, index: int, packed: PackedChunk, source_filename: Optional[str]
    ) -> TextChunk:
        segments: List[ChunkSegment] = []
        for is_overlap, group in ((True, packed.overlap_units), (False, packed.new_units)):
            for unit in group:
                segments.append(
                    ChunkSegment(
                        turn_index=unit.turn_index,
                        speaker=unit.speaker,
                        timestamp=unit.timestamp,
                        text=unit.text,
                        fragment_index=unit.fragment_index,
                        fragment_count=unit.fragment_count,
                        is_overlap=is_overlap,
                        ends_mid_sentence=unit.ends_mid_sentence,
                    )
                )

        text = SEGMENT_SEPARATOR.join(s.rendered_text() for s in segments)
        speakers: List[str] = []
        turn_ids: List[int] = []
        for seg in segments:
            if seg.speaker is not None and seg.speaker not in speakers:
                speakers.append(seg.speaker)
            if seg.turn_index not in turn_ids:
                turn_ids.append(seg.turn_index)
        timestamps = [s.timestamp for s in segments if s.timestamp]

        return TextChunk(
            chunk_id=f"chunk-{index:04d}",
            chunk_index=index,
            text=text,
            character_count=len(text),
            word_count=count_words(text),
            speakers=speakers,
            speaker_turn_ids=turn_ids,
            start_timestamp=timestamps[0] if timestamps else None,
            end_timestamp=timestamps[-1] if timestamps else None,
            is_continuation=packed.new_units[0].fragment_index > 0,
            overlap_character_count=rendered_length(packed.overlap_units),
            source_filename=source_filename,
            segments=segments,
        )
