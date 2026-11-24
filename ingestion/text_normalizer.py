"""Text normalization: cleaner text, same meaning.

What it does:
* Unicode NFC form (e.g. "e" + accent becomes "é"); all characters are kept.
* Line endings (\\r\\n, \\r, \\u2028, \\u2029, form feed) become \\n.
* Non-breaking and other Unicode spaces become normal spaces.
* Zero-width space, word joiner, BOM and soft hyphen are removed.
* Control characters (except newline) are removed.
* Runs of spaces/tabs inside a line become one space; line edges are trimmed.
* 3+ newlines become 2 (one blank line); the whole text is trimmed.

What it never does: remove punctuation, remove speaker labels/timestamps,
or merge separate lines/speakers into one paragraph.
"""
from __future__ import annotations

import re
import unicodedata

_LINE_BREAK_RE = re.compile("\r\n|\r|\u2028|\u2029|\x0b|\x0c|\x85")
_ODD_SPACES_RE = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_INVISIBLE_RE = re.compile("[\u200b\u2060\ufeff\u00ad]")
_CONTROL_RE = re.compile("[\x00-\x08\x0e-\x1f\x7f]")
_INLINE_SPACE_RE = re.compile(r"[ \t]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Return a cleaned copy of text (the input is never modified)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _LINE_BREAK_RE.sub("\n", text)
    text = _ODD_SPACES_RE.sub(" ", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    lines = [_INLINE_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()
