"""Speaker-turn detection for meeting transcripts.

Supported line formats (timestamp is optional):
    Rahul: We should launch next month.
    [10:30] Rahul: We need to finalize the budget.
    (10:30:15) Rahul: ...        10:30 Rahul: ...        [10:30 AM] Rahul: ...
    Rahul [10:30 AM]: ...        Rahul (10:30): ...       (timestamp after name)
    Speaker 1: ...    SPEAKER_01: ...    Participant 2: ...
    Dr. Rahul Sharma: ...

Rules
1. A label is 1-4 words, max 40 characters, before the first colon of a line.
2. Name words must start with an uppercase letter (or a caseless script such
   as Devanagari/Chinese). Lowercase particles (van, de, bin...) are allowed.
   Digits and characters like , ; ? ! ( ) / are not allowed in names.
3. Generic labels (Speaker 1, SPEAKER_01, Participant A, Host...) are allowed.
4. Common header words (Note, Agenda, Date, Attendees, Action Items...) are never
   speaker labels.
5. If the whole text has exactly ONE candidate line and it is not generic and
   has no timestamp, it is not trusted as a speaker label.
6. Lines that follow a label (with no blank line) continue that speaker's turn.
   A blank line ends a turn. Unlabeled text after a blank line (or before the
   first label) becomes a separate turn with speaker = None (unknown). We never
   guess a speaker.
7. A label such as "Unknown:" gives speaker = None.
8. Turns with no text at all are dropped (nothing is lost).
9. Statement text is copied unchanged from the input.

Known limitations
* Lowercase names ("rahul: hi") and names with digits are not recognized.
* Labels with a role in brackets/parentheses ("Rahul (PM): ...") are not recognized.
* A timestamp after the name is recognized only inside brackets or parentheses
  ("Rahul [10:30]:"), not bare ("Rahul 10:30:").
* Two header-like lines such as "Budget: 5000" / "Venue: Hall A" that are not
  in the stop-list may be mistaken for speakers.
* A speaker's multi-paragraph statement separated by blank lines is split, and
  the later paragraphs are marked unknown rather than guessed.
* "Rahul" and "rahul" would count as different speakers.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from ingestion.models import SpeakerTurn

_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm]\b)?"
# Timestamp BEFORE the name:  [10:30] Rahul: ...
_TS_PART = r"(?:[\[(]\s*)?(?P<ts>" + _TIME + r")(?:\s*[\])])?"
# Timestamp in brackets AFTER the name:  Rahul [10:30]: ...
_TS_AFTER_LABEL = r"(?:\s*[\[(]\s*(?P<ts2>" + _TIME + r")\s*[\])])?"
_LINE_RE = re.compile(
    r"^(?:" + _TS_PART + r"[\s\-\u2013\u2014]*)?"
    r"(?P<label>[^:\[(\n]{1,40}?)" + _TS_AFTER_LABEL + r"\s*:\s*(?P<text>.*)$"
)

_GENERIC_RE = re.compile(
    r"^(?:speaker|spk|participant|person|guest|host|interviewer|interviewee|"
    r"moderator|presenter|attendee|unknown(?:[\s_\-]+speaker)?)"
    r"(?:(?:[\s_\-]+[A-Za-z0-9]{1,3})|\d{1,3})?$",
    re.IGNORECASE,
)

_NON_SPEAKER_LABELS = {
    "note", "notes", "agenda", "agenda items", "date", "time", "attendees",
    "present", "absent", "participants", "subject", "topic", "topics",
    "location", "venue", "meeting", "title", "duration", "minutes", "summary",
    "action", "actions", "action item", "action items", "decision", "decisions",
    "owner", "deadline", "status", "priority", "reminder", "warning",
    "important", "example", "from", "to", "cc", "bcc", "re", "fwd", "project",
    "next steps", "key points", "discussion", "objective", "objectives", "goal",
    "goals", "am", "pm", "tip", "caution", "update", "updates", "question",
    "questions", "answer", "answers", "result", "results", "conclusion",
    "background", "context", "description", "name", "email", "phone", "page",
}

_NAME_PARTICLES = {
    "van", "von", "de", "da", "di", "del", "bin", "ibn", "al", "la", "le",
    "du", "der", "den", "dos", "das", "el",
}

_DISALLOWED_CHARS = set(',;?!"\u201c\u201d()[]{}<>@/\\=+*#%&|~^$')
_UNKNOWN_LABELS = {"unknown", "unknown speaker"}


@dataclass
class _Candidate:
    label: str
    timestamp: Optional[str]
    text: str
    generic: bool


@dataclass
class _Draft:
    speaker: Optional[str]
    timestamp: Optional[str]
    lines: List[str] = field(default_factory=list)


def _is_name_word(word: str) -> bool:
    if not word or unicodedata.category(word[0])[0] != "L":
        return False
    for ch in word:
        if unicodedata.category(ch)[0] in ("L", "M") or ch in "'\u2019.-":
            continue
        return False
    if word[0].islower() and word.casefold() not in _NAME_PARTICLES:
        return False
    return True


def _validate_label(raw: str) -> Optional[tuple]:
    """Return (clean_label, is_generic) or None if raw is not a speaker label."""
    label = " ".join(raw.split())
    if not label or len(label) > 40:
        return None
    if label.casefold() in _NON_SPEAKER_LABELS:
        return None
    if _GENERIC_RE.match(label):
        return label, True
    if any(ch in _DISALLOWED_CHARS for ch in label):
        return None
    words = label.split(" ")
    if len(words) > 4 or not all(_is_name_word(w) for w in words):
        return None
    return label, False


def _parse_line(line: str) -> Optional[_Candidate]:
    match = _LINE_RE.match(line)
    if not match:
        return None
    validated = _validate_label(match.group("label"))
    if validated is None:
        return None
    label, generic = validated
    timestamp = match.group("ts") or match.group("ts2")
    return _Candidate(label, timestamp, match.group("text").strip(), generic)


def _accept(candidates: List[Optional[_Candidate]]) -> List[Optional[_Candidate]]:
    """Apply rule 5: distrust a single weak candidate line."""
    found = [c for c in candidates if c is not None]
    if len(found) == 1 and not (found[0].generic or found[0].timestamp):
        return [None] * len(candidates)
    return candidates


def _speaker_name(label: str) -> Optional[str]:
    norm = " ".join(label.casefold().replace("_", " ").replace("-", " ").split())
    return None if norm in _UNKNOWN_LABELS else label


def detect_speaker_turns(text: str) -> List[SpeakerTurn]:
    """Split (normalized) transcript text into speaker turns."""
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    candidates = _accept([_parse_line(line) for line in lines])

    drafts: List[_Draft] = []
    current: Optional[_Draft] = None
    for line, cand in zip(lines, candidates):
        if cand is not None:
            if current is not None:
                drafts.append(current)
            current = _Draft(
                speaker=_speaker_name(cand.label),
                timestamp=cand.timestamp,
                lines=[cand.text] if cand.text else [],
            )
        elif not line.strip():
            if current is not None:
                drafts.append(current)
                current = None
        elif current is None:
            current = _Draft(speaker=None, timestamp=None, lines=[line.strip()])
        else:
            current.lines.append(line.strip())
    if current is not None:
        drafts.append(current)

    turns: List[SpeakerTurn] = []
    for draft in drafts:
        body = "\n".join(draft.lines).strip()
        if not body:
            continue
        turns.append(
            SpeakerTurn(
                index=len(turns),
                speaker=draft.speaker,
                timestamp=draft.timestamp,
                text=body,
            )
        )
    if not turns:  # only empty labels: keep the text instead of losing it
        return [SpeakerTurn(index=0, speaker=None, timestamp=None, text=text.strip())]
    return turns


def get_unique_speakers(turns: List[SpeakerTurn]) -> List[str]:
    """Unique known speaker names in order of first appearance."""
    seen: List[str] = []
    for turn in turns:
        if turn.speaker is not None and turn.speaker not in seen:
            seen.append(turn.speaker)
    return seen
