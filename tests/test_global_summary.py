"""Global executive summary tests with a MOCKED LLM: ordering, duplicates, partial
failures, failures, and hierarchical reduction of large collections."""
import json
import re

import pytest

pytest.importorskip("langchain_core")

from schemas.summary_models import (  # noqa: E402
    ChunkSummary,
    GlobalSummaryStatus,
    SummarizationConfig,
    SummaryStatus,
)
from services.summarization_service import SummarizationService  # noqa: E402
from tests.llm_fakes import GLOBAL_OK, FakeLLM, parts_in, sequence  # noqa: E402


def make_summary(index, status=SummaryStatus.SUCCESS, speakers=("Alice",), text=None, start=None, end=None):
    ok = status == SummaryStatus.SUCCESS
    return ChunkSummary(
        chunk_id=f"chunk-{index:04d}", chunk_index=index,
        summary=(text or f"Summary text of part {index + 1}.") if ok else "",
        key_points=[f"Key point {index + 1}"] if ok else [],
        speakers=list(speakers),
        start_timestamp=start if start is not None else f"10:{index % 60:02d}",
        end_timestamp=end if end is not None else f"10:{index % 60:02d}",
        source_filename="meeting.txt", word_count=50, status=status,
        error_message=None if ok else "simulated failure", attempts=1,
    )


def make_service(fake, **config):
    config.setdefault("retry_delay_seconds", 0.0)
    config.setdefault("timeout_seconds", 5)
    return SummarizationService(fake.runnable(), SummarizationConfig(**config), sleep=lambda s: None)


# --------------------------------------------------------------- happy path
def test_global_summary_is_complete_and_uses_llm_text():
    fake = FakeLLM()
    summaries = [make_summary(0, speakers=("Alice", "Bob")), make_summary(1, speakers=("Bob", "Carol")),
                 make_summary(2, speakers=()), make_summary(3, speakers=("Alice",))]
    result = make_service(fake).generate_global_summary(summaries, source_filename="meeting.txt")
    assert result.status == GlobalSummaryStatus.COMPLETE
    assert result.meeting_title == "Test meeting"
    assert result.executive_summary == "The meeting covered several topics."
    assert result.major_topics == ["Topic A", "Topic B"] and result.unresolved_questions == ["Question 1"]
    assert result.speakers == ["Alice", "Bob", "Carol"]
    assert (result.start_timestamp, result.end_timestamp) == ("10:00", "10:03")
    assert result.source_filename == "meeting.txt"
    assert (result.total_chunks, result.summarized_chunks, result.failed_chunk_ids) == (4, 4, [])
    assert result.reduction_levels == 0 and result.error_message is None
    assert len(fake.by_stage["global"]) == 1 and fake.by_stage["reduce"] == []


def test_all_chunk_summaries_are_sent_to_the_final_prompt_in_order():
    fake = FakeLLM()
    summaries = [make_summary(i) for i in range(4)]
    make_service(fake).generate_global_summary(summaries)
    prompt = fake.by_stage["global"][0]
    assert parts_in(prompt) == [1, 2, 3, 4]
    for i in range(4):
        assert f"Summary text of part {i + 1}." in prompt
    assert "Coverage note: All parts are covered." in prompt


def test_shuffled_input_is_put_back_in_chronological_order():
    fake = FakeLLM()
    summaries = [make_summary(i) for i in (2, 0, 3, 1)]
    make_service(fake).generate_global_summary(summaries)
    assert parts_in(fake.by_stage["global"][0]) == [1, 2, 3, 4]


def test_duplicate_items_in_the_llm_answer_are_reduced():
    answer = json.loads(GLOBAL_OK)
    answer["key_takeaways"] = ["Ship on Friday", "ship on friday", "Review the budget"]
    answer["major_topics"] = ["Budget", "Budget", "budget "]
    fake = FakeLLM(glob=sequence(json.dumps(answer)))
    result = make_service(fake).generate_global_summary([make_summary(0)])
    assert result.key_takeaways == ["Ship on Friday", "Review the budget"]
    assert result.major_topics == ["Budget"]


def test_final_prompt_asks_for_repetition_to_be_removed():
    fake = FakeLLM()
    make_service(fake).generate_global_summary([make_summary(0), make_summary(1)])
    assert "remove unnecessary repetition" in fake.by_stage["global"][0]


def test_total_chunks_argument_is_used():
    fake = FakeLLM()
    result = make_service(fake).generate_global_summary([make_summary(0)], total_chunks=5)
    assert result.total_chunks == 5 and result.summarized_chunks == 1
    assert "Number of parts: 5" in fake.by_stage["global"][0]


def test_complete_summary_converts_to_day1_model():
    result = make_service(FakeLLM()).generate_global_summary([make_summary(0)])
    converted = result.to_executive_summary()
    assert converted.overview == "The meeting covered several topics."
    assert converted.main_points == ["Topic A", "Topic B"] and converted.outcomes == ["Takeaway 1"]


# ----------------------------------------------------------- partial / empty
def test_partial_failure_is_labelled_partial_and_the_gap_is_reported():
    fake = FakeLLM()
    summaries = [make_summary(0), make_summary(1), make_summary(2, SummaryStatus.FAILED), make_summary(3)]
    result = make_service(fake).generate_global_summary(summaries)
    prompt = fake.by_stage["global"][0]
    assert result.status == GlobalSummaryStatus.PARTIAL
    assert result.failed_chunk_ids == ["chunk-0002"] and result.summarized_chunks == 3 and result.total_chunks == 4
    assert "Parts 3 could not be summarized and are NOT covered below." in prompt
    assert "Parts covered by the summaries below: 1\u20132, 4" in prompt
    assert any("does not cover" in w for w in result.warnings)


def test_all_chunks_failed_gives_failed_status_without_calling_the_llm():
    fake = FakeLLM()
    summaries = [make_summary(i, SummaryStatus.FAILED) for i in range(3)]
    result = make_service(fake).generate_global_summary(summaries)
    assert result.status == GlobalSummaryStatus.FAILED
    assert "No chunk could be summarized" in result.error_message and "simulated failure" in result.error_message
    assert result.executive_summary == "" and fake.calls == 0
    assert result.failed_chunk_ids == ["chunk-0000", "chunk-0001", "chunk-0002"]


def test_no_summaries_gives_empty_status_without_calling_the_llm():
    fake = FakeLLM()
    result = make_service(fake).generate_global_summary([])
    assert result.status == GlobalSummaryStatus.EMPTY and result.total_chunks == 0 and fake.calls == 0


def test_only_skipped_summaries_give_empty_status():
    skipped = ChunkSummary(chunk_id="chunk-0000", chunk_index=0, status=SummaryStatus.SKIPPED, error_message="empty")
    fake = FakeLLM()
    result = make_service(fake).generate_global_summary([skipped])
    assert result.status == GlobalSummaryStatus.EMPTY and fake.calls == 0


# ----------------------------------------------------------- LLM failures
def test_global_llm_failure_is_reported_not_hidden():
    fake = FakeLLM(glob=sequence(RuntimeError("server error")))
    result = make_service(fake).generate_global_summary([make_summary(0), make_summary(1)])
    assert result.status == GlobalSummaryStatus.FAILED
    assert "failed after 3 attempt(s)" in result.error_message
    assert result.executive_summary == "" and result.summarized_chunks == 2


def test_global_malformed_answer_is_retried():
    fake = FakeLLM(glob=sequence("not json at all", GLOBAL_OK))
    result = make_service(fake).generate_global_summary([make_summary(0)])
    assert result.status == GlobalSummaryStatus.COMPLETE
    assert "IMPORTANT: your previous reply could not be used" in fake.by_stage["global"][1]


def test_global_answer_missing_required_field_fails_after_retries():
    answer = json.loads(GLOBAL_OK)
    answer["executive_summary"] = ""
    fake = FakeLLM(glob=sequence(json.dumps(answer)))
    result = make_service(fake).generate_global_summary([make_summary(0)])
    assert result.status == GlobalSummaryStatus.FAILED and "malformed_output" in result.error_message


# ------------------------------------------------------ large collections
def test_large_collection_is_reduced_in_stages_and_never_sent_at_once():
    summaries = [make_summary(i, text="Detailed discussion of topic. " * 20) for i in range(120)]
    raw_size = sum(len(s.summary) for s in summaries)
    fake = FakeLLM()
    result = make_service(fake, reduction_batch_chars=3000).generate_global_summary(summaries)

    assert result.status == GlobalSummaryStatus.COMPLETE
    assert result.reduction_levels >= 1 and len(fake.by_stage["reduce"]) > 0
    assert raw_size > 60_000
    assert max(len(p) for p in fake.prompts) < 15_000  # nothing close to the whole input
    for prompt in fake.by_stage["reduce"] + fake.by_stage["global"]:
        numbers = parts_in(prompt)
        assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)  # order kept
    assert parts_in(fake.by_stage["global"][0])[0] == 1
    assert result.summarized_chunks == 120 and result.total_chunks == 120


def test_merge_requests_list_each_original_part_at_most_once_and_in_order():
    summaries = [make_summary(i, text="Detailed discussion of topic. " * 20) for i in range(60)]
    fake = FakeLLM()
    make_service(fake, reduction_batch_chars=3000).generate_global_summary(summaries)
    singles = []
    for prompt in fake.by_stage["reduce"]:
        singles.extend(int(n) for n in re.findall(r"^Part (\d+) \(", prompt, re.MULTILINE))
    assert singles == sorted(set(singles))  # ascending, no repeats
    assert set(singles) <= set(range(1, 61)) and len(singles) >= 56


def test_failed_merges_keep_the_original_summaries_and_warn():
    summaries = [make_summary(i, text="Detailed discussion of topic. " * 20) for i in range(40)]
    fake = FakeLLM(reduce=sequence(RuntimeError("merge failed")))
    result = make_service(fake, reduction_batch_chars=3000, max_retries=0).generate_global_summary(summaries)
    assert result.status == GlobalSummaryStatus.COMPLETE and result.reduction_levels == 0
    assert any("could not be merged" in w for w in result.warnings)
    assert any("larger than the configured limit" in w for w in result.warnings)
    assert parts_in(fake.by_stage["global"][0]) == list(range(1, 41))  # nothing dropped


def test_reduction_stops_at_the_configured_maximum():
    summaries = [make_summary(i, text="Short text. " * 10) for i in range(60)]
    fake = FakeLLM()
    result = make_service(fake, reduction_batch_chars=1000, max_reduction_levels=1).generate_global_summary(summaries)
    assert result.reduction_levels == 1 and result.status == GlobalSummaryStatus.COMPLETE
    assert any("maximum of 1 reduction levels" in w for w in result.warnings)
