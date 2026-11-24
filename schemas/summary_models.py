"""Pydantic models for Day 4: chunk summaries and the global executive summary.

Relationship to Day 1: schemas/extraction_models.py has a small ExecutiveSummary
(overview / main_points / outcomes) that Day 5 will place inside MeetingExtraction.
GlobalExecutiveSummary is the richer Day 4 result and can be converted with
GlobalExecutiveSummary.to_executive_summary(). Decisions and action items are NOT
produced here (Day 5).

Three kinds of models:
* *LLMOutput  - exactly what the LLM must return (validated, then retried if invalid).
* ChunkSummary / GlobalExecutiveSummary - pipeline results with metadata copied from
  the Day 3 chunks (speakers and timestamps come from the chunk, never from the LLM).
* SummarizationConfig - settings for the summarization service.
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, List, Mapping, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schemas.extraction_models import ExecutiveSummary
from services.summarization_utils import dedupe_preserve_order


class SummaryStatus(str, Enum):
    """Result of summarizing ONE chunk."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # nothing to summarize (empty chunk)


class GlobalSummaryStatus(str, Enum):
    """Result of the whole-meeting summary. Only COMPLETE means 'full coverage'."""

    COMPLETE = "complete"  # every chunk was summarized and the final summary was made
    PARTIAL = "partial"  # final summary made, but some chunks could not be summarized
    FAILED = "failed"  # no usable global summary
    EMPTY = "empty"  # nothing to summarize


def _clean_list(value: Any) -> Any:
    """Tolerate small LLM slips: None -> [], one string -> [string].

    Blank items are dropped and repeated items are removed (case-insensitive).
    Wrong types (for example numbers) are left alone so Pydantic rejects them.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            return value
        return dedupe_preserve_order([item.strip() for item in value if item.strip()])
    return value


class _Model(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


# ------------------------------------------------------------ LLM outputs
class ChunkSummaryLLMOutput(_Model):
    """JSON the LLM returns for one chunk (also used when merging summaries)."""

    summary: str = Field(
        min_length=1,
        description="Concise summary of this part of the meeting. Keep names, figures "
        "and dates. Do not invent anything.",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Important points, one short sentence each. Mention the speaker "
        "when it matters.",
    )
    unresolved_items: List[str] = Field(
        default_factory=list,
        description="Open questions, disagreements or differing opinions. Empty list if none.",
    )

    _clean_lists = field_validator("key_points", "unresolved_items", mode="before")(_clean_list)


class GlobalLLMOutput(_Model):
    """JSON the LLM returns for the whole meeting."""

    meeting_title: str = Field(
        min_length=1,
        description="Short descriptive title. If the topic is unclear use 'Meeting summary'.",
    )
    executive_summary: str = Field(
        min_length=1,
        description="One coherent narrative summary of the whole meeting (not a list).",
    )
    meeting_overview: str = Field(
        min_length=1,
        description="One or two sentences: the purpose and context of the meeting.",
    )
    major_topics: List[str] = Field(default_factory=list, description="Main topics, in order.")
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="Important developments or conclusions that were actually stated.",
    )
    important_discussions: List[str] = Field(
        default_factory=list, description="Notable discussions or exchanges."
    )
    unresolved_questions: List[str] = Field(
        default_factory=list, description="Open questions, disagreements, pending issues."
    )

    _clean_lists = field_validator(
        "major_topics",
        "key_takeaways",
        "important_discussions",
        "unresolved_questions",
        mode="before",
    )(_clean_list)


# --------------------------------------------------------- pipeline results
class ChunkSummary(_Model):
    """Summary of one Day 3 TextChunk plus its metadata."""

    chunk_id: str
    chunk_index: int = Field(ge=0)
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    unresolved_items: List[str] = Field(default_factory=list)
    speakers: List[str] = Field(default_factory=list, description="From the chunk metadata.")
    speaker_turn_ids: List[int] = Field(default_factory=list)
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    source_filename: Optional[str] = None
    word_count: int = Field(default=0, ge=0)
    status: SummaryStatus = SummaryStatus.SUCCESS
    error_message: Optional[str] = Field(
        default=None, description="Why the chunk failed or was skipped (secrets removed)."
    )
    attempts: int = Field(default=0, ge=0, description="Number of LLM calls made.")

    @model_validator(mode="after")
    def _check_status(self) -> "ChunkSummary":
        if self.status == SummaryStatus.SUCCESS and not self.summary.strip():
            raise ValueError("A successful chunk summary must contain text.")
        if self.status != SummaryStatus.SUCCESS and not self.error_message:
            raise ValueError("Failed or skipped chunk summaries need an error_message.")
        return self


class GlobalExecutiveSummary(_Model):
    """Executive summary of the complete meeting."""

    meeting_title: str = ""
    executive_summary: str = ""
    meeting_overview: str = ""
    major_topics: List[str] = Field(default_factory=list)
    key_takeaways: List[str] = Field(default_factory=list)
    important_discussions: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    speakers: List[str] = Field(
        default_factory=list, description="All known speakers (from chunk metadata)."
    )
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    source_filename: Optional[str] = None
    total_chunks: int = Field(default=0, ge=0)
    summarized_chunks: int = Field(default=0, ge=0)
    failed_chunk_ids: List[str] = Field(default_factory=list)
    reduction_levels: int = Field(
        default=0, ge=0, description="Extra merge rounds needed for very long meetings."
    )
    status: GlobalSummaryStatus = GlobalSummaryStatus.COMPLETE
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "GlobalExecutiveSummary":
        if self.summarized_chunks > self.total_chunks:
            raise ValueError("summarized_chunks cannot be larger than total_chunks.")
        if self.status in (GlobalSummaryStatus.COMPLETE, GlobalSummaryStatus.PARTIAL):
            if not self.executive_summary.strip():
                raise ValueError("A complete or partial summary must contain text.")
        if self.status == GlobalSummaryStatus.PARTIAL and not self.failed_chunk_ids:
            raise ValueError("A partial summary must list the failed chunks.")
        if self.status == GlobalSummaryStatus.FAILED and not self.error_message:
            raise ValueError("A failed summary needs an error_message.")
        return self

    def to_executive_summary(self) -> ExecutiveSummary:
        """Convert to the Day 1 ExecutiveSummary (used inside MeetingExtraction on Day 5).

        overview <- executive_summary, main_points <- major_topics,
        outcomes <- key_takeaways.
        """
        if self.status not in (GlobalSummaryStatus.COMPLETE, GlobalSummaryStatus.PARTIAL):
            raise ValueError(f"No summary available (status: {self.status.value}).")
        return ExecutiveSummary(
            overview=self.executive_summary,
            main_points=list(self.major_topics),
            outcomes=list(self.key_takeaways),
        )


class MeetingSummaryResult(_Model):
    """Everything the summarization pipeline returns."""

    source_filename: Optional[str] = None
    status: GlobalSummaryStatus
    chunk_summaries: List[ChunkSummary] = Field(default_factory=list)
    global_summary: GlobalExecutiveSummary
    llm_call_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    processing_seconds: float = Field(default=0.0, ge=0)
    warnings: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------ config
class SummarizationConfig(BaseModel):
    """Settings for the summarization service."""

    max_retries: int = Field(
        default=2, ge=0, le=10, description="Extra attempts after a failed LLM call."
    )
    retry_delay_seconds: float = Field(
        default=1.0, ge=0, description="First wait before a retry; doubles each time."
    )
    timeout_seconds: float = Field(default=60.0, gt=0, description="Wait limit per LLM call.")
    request_delay_seconds: float = Field(
        default=0.0, ge=0, description="Pause between chunk requests (free-tier limits)."
    )
    reduction_batch_chars: int = Field(
        default=12000,
        ge=1000,
        description="Max characters of summaries sent in one merge/global request "
        "(characters, not tokens).",
    )
    max_reduction_levels: int = Field(default=5, ge=1, le=20)
    include_overlap_context: bool = Field(
        default=True, description="Show the overlap text as 'previous context' only."
    )
    output_language: str = Field(default="English", min_length=1)

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None, load_dotenv_file: bool = True
    ) -> "SummarizationConfig":
        """Read optional SUMMARY_* variables (see .env.example)."""
        if env is None:
            if load_dotenv_file:
                load_dotenv()
            env = os.environ

        values: dict = {}
        converters = {
            "SUMMARY_MAX_RETRIES": ("max_retries", int),
            "SUMMARY_RETRY_DELAY_SECONDS": ("retry_delay_seconds", float),
            "SUMMARY_TIMEOUT_SECONDS": ("timeout_seconds", float),
            "SUMMARY_REQUEST_DELAY_SECONDS": ("request_delay_seconds", float),
            "SUMMARY_REDUCTION_BATCH_CHARS": ("reduction_batch_chars", int),
            "SUMMARY_OUTPUT_LANGUAGE": ("output_language", str),
        }
        for name, (field_name, convert) in converters.items():
            raw = env.get(name, "").strip()
            if not raw:
                continue
            try:
                values[field_name] = convert(raw)
            except ValueError as exc:
                raise ValueError(f"{name} has an invalid value: '{raw}'.") from exc
        return cls(**values)
