"""Pydantic validation tests for the Day 4 summary models (real Pydantic)."""
import json

import pytest
from pydantic import ValidationError

from schemas.extraction_models import ExecutiveSummary
from schemas.summary_models import (
    ChunkSummary,
    ChunkSummaryLLMOutput,
    GlobalExecutiveSummary,
    GlobalLLMOutput,
    GlobalSummaryStatus,
    MeetingSummaryResult,
    SummarizationConfig,
    SummaryStatus,
)


def global_output(**overrides):
    data = dict(
        meeting_title="Title", executive_summary="Summary.", meeting_overview="Overview.",
    )
    data.update(overrides)
    return GlobalLLMOutput(**data)


# ------------------------------------------------------------- LLM outputs
def test_valid_chunk_llm_output():
    out = ChunkSummaryLLMOutput(summary="Alice proposed a plan.", key_points=["Plan"], unresolved_items=[])
    assert out.summary == "Alice proposed a plan." and out.key_points == ["Plan"]


def test_blank_summary_rejected():
    with pytest.raises(ValidationError):
        ChunkSummaryLLMOutput(summary="   ")


def test_missing_summary_rejected():
    with pytest.raises(ValidationError):
        ChunkSummaryLLMOutput(key_points=["x"])


def test_single_string_becomes_list_and_blanks_are_dropped():
    out = ChunkSummaryLLMOutput(summary="ok", key_points="Only one point", unresolved_items=["", "  ", "Real question"])
    assert out.key_points == ["Only one point"]
    assert out.unresolved_items == ["Real question"]


def test_none_lists_become_empty():
    out = ChunkSummaryLLMOutput(summary="ok", key_points=None, unresolved_items=None)
    assert out.key_points == [] and out.unresolved_items == []


def test_duplicate_points_removed_case_insensitively_keeping_order():
    out = ChunkSummaryLLMOutput(summary="ok", key_points=["Ship on Friday", "Review budget", "ship on  friday"])
    assert out.key_points == ["Ship on Friday", "Review budget"]


def test_wrong_item_types_rejected_not_silently_accepted():
    with pytest.raises(ValidationError):
        ChunkSummaryLLMOutput(summary="ok", key_points=[1, 2])
    with pytest.raises(ValidationError):
        ChunkSummaryLLMOutput(summary="ok", key_points={"a": "b"})


def test_global_output_requires_title_summary_overview():
    assert global_output().meeting_title == "Title"
    for field in ("meeting_title", "executive_summary", "meeting_overview"):
        with pytest.raises(ValidationError):
            global_output(**{field: " "})


def test_global_output_lists_are_cleaned():
    out = global_output(major_topics=["A", "a", ""], key_takeaways="One", unresolved_questions=None)
    assert out.major_topics == ["A"] and out.key_takeaways == ["One"] and out.unresolved_questions == []


# --------------------------------------------------------- pipeline models
def test_successful_chunk_summary_needs_text():
    ok = ChunkSummary(chunk_id="chunk-0000", chunk_index=0, summary="Text.", status=SummaryStatus.SUCCESS)
    assert ok.status == SummaryStatus.SUCCESS
    with pytest.raises(ValidationError):
        ChunkSummary(chunk_id="chunk-0000", chunk_index=0, summary="", status=SummaryStatus.SUCCESS)


def test_failed_and_skipped_chunk_summaries_need_an_error_message():
    for status in (SummaryStatus.FAILED, SummaryStatus.SKIPPED):
        with pytest.raises(ValidationError):
            ChunkSummary(chunk_id="c", chunk_index=0, status=status)
        item = ChunkSummary(chunk_id="c", chunk_index=0, status=status, error_message="why")
        assert item.error_message == "why"


def test_negative_chunk_index_rejected():
    with pytest.raises(ValidationError):
        ChunkSummary(chunk_id="c", chunk_index=-1, summary="x")


def test_complete_global_summary_needs_text():
    with pytest.raises(ValidationError):
        GlobalExecutiveSummary(status=GlobalSummaryStatus.COMPLETE, executive_summary="")
    ok = GlobalExecutiveSummary(status=GlobalSummaryStatus.COMPLETE, executive_summary="Text.", total_chunks=2, summarized_chunks=2)
    assert ok.summarized_chunks == 2


def test_partial_summary_must_list_failed_chunks():
    with pytest.raises(ValidationError):
        GlobalExecutiveSummary(status=GlobalSummaryStatus.PARTIAL, executive_summary="Text.", total_chunks=2, summarized_chunks=1)
    ok = GlobalExecutiveSummary(
        status=GlobalSummaryStatus.PARTIAL, executive_summary="Text.", total_chunks=2,
        summarized_chunks=1, failed_chunk_ids=["chunk-0001"],
    )
    assert ok.failed_chunk_ids == ["chunk-0001"]


def test_failed_summary_needs_error_message_and_empty_is_allowed():
    with pytest.raises(ValidationError):
        GlobalExecutiveSummary(status=GlobalSummaryStatus.FAILED)
    assert GlobalExecutiveSummary(status=GlobalSummaryStatus.FAILED, error_message="boom").error_message == "boom"
    assert GlobalExecutiveSummary(status=GlobalSummaryStatus.EMPTY).executive_summary == ""


def test_summarized_chunks_cannot_exceed_total():
    with pytest.raises(ValidationError):
        GlobalExecutiveSummary(status=GlobalSummaryStatus.COMPLETE, executive_summary="x", total_chunks=1, summarized_chunks=2)


def test_conversion_to_day1_executive_summary():
    summary = GlobalExecutiveSummary(
        status=GlobalSummaryStatus.COMPLETE, executive_summary="The whole story.",
        major_topics=["A", "B"], key_takeaways=["Done"], total_chunks=1, summarized_chunks=1,
    )
    converted = summary.to_executive_summary()
    assert isinstance(converted, ExecutiveSummary)
    assert converted.overview == "The whole story."
    assert converted.main_points == ["A", "B"] and converted.outcomes == ["Done"]


def test_conversion_refused_when_no_summary_exists():
    with pytest.raises(ValueError):
        GlobalExecutiveSummary(status=GlobalSummaryStatus.EMPTY).to_executive_summary()


def test_meeting_result_is_json_serializable():
    summary = GlobalExecutiveSummary(status=GlobalSummaryStatus.EMPTY)
    result = MeetingSummaryResult(status=GlobalSummaryStatus.EMPTY, global_summary=summary)
    data = json.loads(json.dumps(result.model_dump(mode="json")))
    assert data["status"] == "empty" and data["global_summary"]["status"] == "empty"


# ------------------------------------------------------------------ config
def test_config_defaults():
    config = SummarizationConfig()
    assert config.max_retries == 2 and config.timeout_seconds == 60
    assert config.reduction_batch_chars == 12000 and config.output_language == "English"
    assert config.include_overlap_context is True


@pytest.mark.parametrize(
    "kwargs",
    [{"max_retries": -1}, {"timeout_seconds": 0}, {"reduction_batch_chars": 999}, {"retry_delay_seconds": -1}, {"max_reduction_levels": 0}],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValidationError):
        SummarizationConfig(**kwargs)


def test_config_from_env():
    config = SummarizationConfig.from_env(
        env={"SUMMARY_MAX_RETRIES": "4", "SUMMARY_TIMEOUT_SECONDS": "30", "SUMMARY_OUTPUT_LANGUAGE": "Hindi"}
    )
    assert (config.max_retries, config.timeout_seconds, config.output_language) == (4, 30.0, "Hindi")
    assert SummarizationConfig.from_env(env={}).max_retries == 2


def test_config_from_env_rejects_bad_numbers():
    with pytest.raises(ValueError, match="SUMMARY_MAX_RETRIES"):
        SummarizationConfig.from_env(env={"SUMMARY_MAX_RETRIES": "many"})
