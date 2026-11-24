"""Packing of speaker-turn units into chunks (pure Python, no third-party imports).

A "unit" is either a whole speaker turn or one fragment of a long turn that
was divided by RecursiveCharacterTextSplitter (see speaker_aware_splitter.py).

Rules:
* Units are never reordered and never divided here.
* A chunk is closed when adding the next unit would exceed chunk_size, so
  boundaries always fall between units (between turns where possible).
* Overlap: the next chunk starts with a copy of the trailing whole units of the
  previous chunk whose total rendered length is <= chunk_overlap. The copies
  are kept separate (overlap_units) so content is never double-counted.
* If a unit alone is bigger than chunk_size, it gets its own chunk (it is
  reported as oversized later; it is never cut).
Sizes are measured in CHARACTERS of the rendered chunk text, not tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from chunking.text_utils import SEGMENT_SEPARATOR, label_prefix


@dataclass(frozen=True)
class Unit:
    turn_index: int
    speaker: Optional[str]
    timestamp: Optional[str]
    text: str
    fragment_index: int = 0
    fragment_count: int = 1
    ends_mid_sentence: bool = False

    @property
    def rendered(self) -> str:
        return label_prefix(self.speaker, self.timestamp) + self.text


@dataclass
class PackedChunk:
    overlap_units: List[Unit] = field(default_factory=list)
    new_units: List[Unit] = field(default_factory=list)


def rendered_length(units: Sequence[Unit]) -> int:
    """Length in characters of the units rendered and joined by the separator."""
    if not units:
        return 0
    return sum(len(u.rendered) for u in units) + len(SEGMENT_SEPARATOR) * (len(units) - 1)


def select_overlap(units: Sequence[Unit], chunk_overlap: int) -> List[Unit]:
    """Longest suffix of whole units whose rendered length is <= chunk_overlap."""
    selected: List[Unit] = []
    for unit in reversed(units):
        candidate = [unit] + selected
        if rendered_length(candidate) > chunk_overlap:
            break
        selected = candidate
    return selected


def pack_units(
    units: Sequence[Unit], chunk_size: int, chunk_overlap: int
) -> List[PackedChunk]:
    """Group units into chunks (see module docstring for the rules)."""
    chunks: List[PackedChunk] = []
    carry: List[Unit] = []
    current: List[Unit] = []
    for unit in units:
        if current and rendered_length(carry + current + [unit]) > chunk_size:
            chunks.append(PackedChunk(list(carry), list(current)))
            carry = select_overlap(current, chunk_overlap)
            if carry and rendered_length(carry + [unit]) > chunk_size:
                carry = []  # overlap must never push a unit over the limit
            current = []
        current.append(unit)
    if current:
        chunks.append(PackedChunk(list(carry), list(current)))
    return chunks
