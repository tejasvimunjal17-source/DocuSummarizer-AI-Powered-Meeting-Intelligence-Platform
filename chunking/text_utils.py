"""Small pure helpers shared by the chunking modules (standard library only)."""
from __future__ import annotations

import re
from typing import Optional

# Between two rendered segments inside one chunk.
SEGMENT_SEPARATOR = "\n"

_SENTENCE_END_RE = re.compile(r'[.!?\u2026\u3002\uff01\uff1f\u0964]["\'\u201d\u2019)\]]*$')
_WHITESPACE_RE = re.compile(r"\s+")


def ends_with_sentence_boundary(text: str) -> bool:
    """True if text ends with sentence punctuation (. ! ? ... and CJK/Hindi marks)."""
    return bool(_SENTENCE_END_RE.search(text.rstrip()))


def compact(text: str) -> str:
    """Remove ALL whitespace. Used to prove that no characters were lost."""
    return _WHITESPACE_RE.sub("", text)


def count_words(text: str) -> int:
    """Word count = number of whitespace-separated tokens."""
    return len(text.split())


def label_prefix(speaker: Optional[str], timestamp: Optional[str]) -> str:
    """The label written before a statement inside a chunk.

    Examples: "[10:00 AM] Alice: ", "Alice: ", "[10:00 AM] ", "" (unknown, no time).
    Unknown speakers never get an invented name.
    """
    prefix = ""
    if timestamp:
        prefix += f"[{timestamp}] "
    if speaker:
        prefix += f"{speaker}: "
    return prefix
