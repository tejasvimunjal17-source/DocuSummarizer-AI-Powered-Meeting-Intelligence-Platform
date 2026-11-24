"""Pydantic models describing what DocuSummarizer extracts from a meeting.

Design rules:
* Information that may be missing from a transcript (owner, deadline, speaker,
  context) is Optional and defaults to None. We never invent values.
* Blank strings from an LLM (e.g. "") are converted to None for optional fields.
* Severity accepts any letter case ("HIGH", "High") and is stored as an Enum.
* Empty lists are valid (a meeting may have no risks, for example).
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Severity(str, Enum):
    """How serious a risk is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionStatus(str, Enum):
    """Completion status of an action item. UNKNOWN is used when not stated."""

    UNKNOWN = "unknown"
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class _Base(BaseModel):
    """Shared config: trim whitespace on all strings."""

    model_config = ConfigDict(str_strip_whitespace=True)


def _blank_to_none(value):
    """Turn empty/whitespace-only strings into None (LLMs often return "")."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


class ExecutiveSummary(_Base):
    """High-level summary of the whole meeting."""

    overview: str = Field(min_length=1, description="Overall meeting summary.")
    main_points: List[str] = Field(
        default_factory=list, description="Main discussion points."
    )
    outcomes: List[str] = Field(
        default_factory=list, description="Important outcomes of the meeting."
    )


class KeyDecision(_Base):
    """A decision made during the meeting."""

    decision: str = Field(min_length=1, description="What was decided.")
    speaker: Optional[str] = Field(
        default=None,
        description="Who made or announced the decision. None if unclear.",
    )
    context: Optional[str] = Field(
        default=None, description="Supporting context from the discussion."
    )

    _clean = field_validator("speaker", "context", mode="before")(_blank_to_none)


class ActionItem(_Base):
    """A task someone must do. Owner and deadline are often not stated."""

    task: str = Field(min_length=1, description="What needs to be done.")
    owner: Optional[str] = Field(
        default=None, description="Assigned person. None if not stated."
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Deadline as written in the transcript (e.g. 'next Friday'). "
        "None if not stated.",
    )
    context: Optional[str] = Field(
        default=None, description="Relevant context for the task."
    )
    status: ActionStatus = Field(
        default=ActionStatus.UNKNOWN, description="Completion status, if available."
    )

    _clean = field_validator("owner", "deadline", "context", mode="before")(
        _blank_to_none
    )


class RiskFlag(_Base):
    """A risk or concern raised in the meeting."""

    description: str = Field(min_length=1, description="What the risk is.")
    severity: Severity = Field(description="Risk severity: low, medium or high.")
    context: Optional[str] = Field(
        default=None, description="Supporting context from the discussion."
    )
    potential_impact: Optional[str] = Field(
        default=None, description="What could happen if the risk occurs."
    )

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, value):
        """Accept 'HIGH', ' High ' etc. Other invalid values still fail."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    _clean = field_validator("context", "potential_impact", mode="before")(
        _blank_to_none
    )


class TopicCluster(_Base):
    """A group of related discussion points under one topic."""

    name: str = Field(min_length=1, description="Short topic name.")
    description: Optional[str] = Field(
        default=None, description="What this topic was about."
    )
    related_points: List[str] = Field(
        default_factory=list, description="Discussion points under this topic."
    )

    _clean = field_validator("description", mode="before")(_blank_to_none)


class MeetingExtraction(_Base):
    """Complete structured result for one meeting transcript."""

    summary: ExecutiveSummary
    decisions: List[KeyDecision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    risks: List[RiskFlag] = Field(default_factory=list)
    topics: List[TopicCluster] = Field(default_factory=list)

    def to_json_dict(self) -> dict:
        """Return a JSON-compatible dict (enums become plain strings)."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON string."""
        return self.model_dump_json(indent=indent)
