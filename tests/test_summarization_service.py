"""End-to-end Day 2 -> Day 3 -> Day 4 tests with a MOCKED LLM, expected-summary
fixture checks, and light regression checks of earlier days.

These tests run the REAL ingestion, chunking (LangChain RecursiveCharacterTextSplitter
where needed), prompt templates, chains and Pydantic models. Only the LLM is fake.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_text_splitters")

from chunking import (  # noqa: E402
    ChunkingConfig,
    chunk_file,
    chunk_ingestion_result,
    validate_chunks,
)
from ingestion import detect_speaker_turns, ingest_bytes, ingest_file, normalize_text  # noqa: E402
from schemas import MeetingExtraction  # noqa: E402
from schemas.summary_models import (  # noqa: E402
    GlobalExecutiveSummary,
    GlobalLLMOutput,
    GlobalSummaryStatus,
    SummarizationConfig,
    SummaryStatus,
)
from services.summarization_service import (  # noqa: E402
    SummarizationService,
    summarize_chunking_result,
    summarize_file,
    summarize_text,
)
from tests.llm_fakes import FakeLLM, chunk_json, parts_in, sequence  # noqa: E402

def context_of(prompt):
    """The PREVIOUS CONTEXT section of a chunk prompt."""
    after = prompt.split("PREVIOUS CONTEXT (do not summarize again):")[1]
    return after.split("TRANSCRIPT PART TO SUMMARIZE")[0].strip()


SAMPLES = Path(__file__).resolve().parent.parent / "sample_transcripts"
FIXTURES = SAMPLES / "expected_summaries"
FAST = SummarizationConfig(retry_delay_seconds=0.0, timeout_seconds=5, max_retries=0)

FIXTURE_FOR_SAMPLE = {
    "short_meeting": "chunking_short_meeting.txt",
    "multi_speaker_long_turns": "chunking_long_turns.txt",
    "missing_speaker_information": "chunking_mixed_unlabeled.txt",
    "multilingual_meeting": "chunking_multilingual.txt",
    "long_realistic_meeting": "chunking_realistic_meeting.txt",
}


def load_fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------ end to end
def test_short_meeting_end_to_end():
    fake = FakeLLM()
    result = summarize_file(SAMPLES / "chunking_short_meeting.txt", llm=fake.runnable(), config=FAST)
    assert result.status == GlobalSummaryStatus.COMPLETE
    assert len(result.chunk_summaries) == 1 and result.chunk_summaries[0].status == SummaryStatus.SUCCESS
    assert result.global_summary.speakers == ["Alice", "Bob", "Charlie"]
    assert (result.global_summary.start_timestamp, result.global_summary.end_timestamp) == ("10:00 AM", "10:03 AM")
    assert result.global_summary.source_filename == "chunking_short_meeting.txt"
    assert result.llm_call_count == 2 and result.retry_count == 0 and result.processing_seconds >= 0
    assert "[10:00 AM] Alice: Welcome everyone." in fake.by_stage["chunk"][0]


def test_realistic_meeting_end_to_end_keeps_chronological_order():
    fake = FakeLLM()
    chunking = chunk_file(SAMPLES / "chunking_realistic_meeting.txt")
    result = summarize_chunking_result(chunking, llm=fake.runnable(), config=FAST)
    total = chunking.total_chunks
    assert total > 5 and result.status == GlobalSummaryStatus.COMPLETE
    assert [s.chunk_index for s in result.chunk_summaries] == list(range(total))
    assert all(s.status == SummaryStatus.SUCCESS for s in result.chunk_summaries)
    assert result.llm_call_count == total + 1  # one per chunk plus the global summary
    for i, prompt in enumerate(fake.by_stage["chunk"]):
        assert f"Meeting part {i + 1} of {total}." in prompt
    assert parts_in(fake.by_stage["global"][0]) == list(range(1, total + 1))
    assert result.global_summary.total_chunks == total == result.global_summary.summarized_chunks
    assert set(result.global_summary.speakers) == {"Alice", "Bob", "Charlie", "Dana", "Eli"}
    assert context_of(fake.by_stage["chunk"][0]) == "(none)"  # first chunk has no overlap
    assert any(context_of(p) != "(none)" for p in fake.by_stage["chunk"][1:])  # overlap becomes context


def test_day2_day3_day4_chain_with_ingestion_result():
    ingestion = ingest_bytes("meeting.txt", b"Alice: Hello there.\n\nBob: Hi Alice, welcome.")
    chunking = chunk_ingestion_result(ingestion)
    result = summarize_chunking_result(chunking, llm=FakeLLM().runnable(), config=FAST)
    assert result.source_filename == "meeting.txt"
    assert result.chunk_summaries[0].source_filename == "meeting.txt"
    assert result.chunk_summaries[0].speakers == ["Alice", "Bob"]


def test_long_turn_continuations_reach_the_prompts():
    fake = FakeLLM()
    config = ChunkingConfig(chunk_size=400, chunk_overlap=80)
    result = summarize_file(SAMPLES / "chunking_long_turns.txt", llm=fake.runnable(), chunking_config=config, config=FAST)
    assert result.status == GlobalSummaryStatus.COMPLETE and len(result.chunk_summaries) >= 4
    assert any("begins in the middle" in p for p in fake.by_stage["chunk"])
    assert any("Dana" in s.speakers for s in result.chunk_summaries)


def test_meeting_without_speaker_information():
    fake = FakeLLM()
    result = summarize_text("Plain notes without any speaker names.\n\nMore plain notes here.", llm=fake.runnable(), config=FAST)
    assert result.status == GlobalSummaryStatus.COMPLETE
    assert result.chunk_summaries[0].speakers == [] and result.global_summary.speakers == []
    assert "not labeled" in fake.by_stage["chunk"][0]


def test_multilingual_meeting_keeps_speaker_names():
    fake = FakeLLM()
    result = summarize_file(SAMPLES / "chunking_multilingual.txt", llm=fake.runnable(), config=FAST)
    hindi, chinese = "\u0930\u093e\u0939\u0941\u0932", "\u674e\u96f7"
    assert hindi in result.global_summary.speakers and chinese in result.global_summary.speakers
    assert f"{hindi}:" in fake.by_stage["chunk"][0] and "Write in English" in fake.by_stage["chunk"][0]


def test_empty_transcript_gives_empty_status_and_no_llm_calls():
    fake = FakeLLM()
    result = summarize_text("   \n\n", llm=fake.runnable(), config=FAST)
    assert result.status == GlobalSummaryStatus.EMPTY and result.chunk_summaries == []
    assert result.llm_call_count == 0 and fake.calls == 0


# -------------------------------------------------------------- failures
def test_partial_failure_end_to_end_is_not_reported_as_complete():
    fake = FakeLLM(chunk=sequence(chunk_json(1), chunk_json(2), RuntimeError("boom"), chunk_json(4)))
    chunking = chunk_file(SAMPLES / "chunking_long_turns.txt", ChunkingConfig(chunk_size=400, chunk_overlap=80))
    assert chunking.total_chunks >= 4
    result = summarize_chunking_result(chunking, llm=fake.runnable(), config=FAST)
    assert result.status == GlobalSummaryStatus.PARTIAL
    assert result.global_summary.failed_chunk_ids == ["chunk-0002"]
    assert result.chunk_summaries[2].status == SummaryStatus.FAILED
    assert any("chunk-0002 failed" in w for w in result.warnings)


def test_global_failure_keeps_the_chunk_summaries():
    fake = FakeLLM(glob=sequence(RuntimeError("boom")))
    result = summarize_file(SAMPLES / "chunking_short_meeting.txt", llm=fake.runnable(), config=FAST)
    assert result.status == GlobalSummaryStatus.FAILED
    assert result.chunk_summaries[0].status == SummaryStatus.SUCCESS
    assert result.global_summary.error_message


def test_service_instance_can_be_reused_and_reports_per_run_counts():
    fake = FakeLLM()
    service = SummarizationService(fake.runnable(), FAST)
    chunking = chunk_file(SAMPLES / "chunking_short_meeting.txt")
    first = service.summarize_chunking_result(chunking)
    second = service.summarize_chunking_result(chunking)
    assert first.llm_call_count == 2 and second.llm_call_count == 2


# -------------------------------------------------------- expected fixtures
@pytest.mark.parametrize("name", sorted(FIXTURE_FOR_SAMPLE))
def test_expected_summary_fixture_is_valid_and_clearly_labelled(name):
    data = load_fixture(name)
    assert "NOT generated by an LLM" in data["fixture_notice"] and "hand" in data["fixture_notice"]
    summary = GlobalExecutiveSummary.model_validate(data)
    assert summary.status == GlobalSummaryStatus.COMPLETE and summary.executive_summary


@pytest.mark.parametrize("name", sorted(FIXTURE_FOR_SAMPLE))
def test_expected_summary_fixture_metadata_matches_the_real_pipeline(name):
    data = load_fixture(name)
    sample = SAMPLES / FIXTURE_FOR_SAMPLE[name]
    chunking = chunk_file(sample)
    assert data["source_file"] == sample.name
    assert data["total_chunks"] == chunking.total_chunks
    assert data["speakers"] == ingest_file(sample).speakers
    assert data["start_timestamp"] == chunking.chunks[0].start_timestamp
    assert data["end_timestamp"] == chunking.chunks[-1].end_timestamp


def test_fixture_used_as_mocked_llm_answer_flows_through_the_pipeline():
    data = load_fixture("short_meeting")
    answer = {name: data[name] for name in GlobalLLMOutput.model_fields}
    fake = FakeLLM(glob=sequence(json.dumps(answer)))
    result = summarize_file(SAMPLES / "chunking_short_meeting.txt", llm=fake.runnable(), config=FAST)
    assert result.global_summary.executive_summary == data["executive_summary"]
    assert result.global_summary.key_takeaways == data["key_takeaways"]
    assert result.global_summary.speakers == data["speakers"]


# ---------------------------------------------------- previous-day regression
def test_day1_models_accept_the_converted_summary():
    result = summarize_file(SAMPLES / "chunking_short_meeting.txt", llm=FakeLLM().runnable(), config=FAST)
    extraction = MeetingExtraction(summary=result.global_summary.to_executive_summary())
    assert extraction.summary.overview and extraction.decisions == []
    assert json.loads(extraction.to_json())["summary"]["overview"]


def test_day2_ingestion_still_works_as_before():
    ingestion = ingest_bytes("m.txt", b"[10:30] Rahul: Hi\n[10:31] Priya: Hello")
    assert ingestion.speakers == ["Rahul", "Priya"]
    assert [t.timestamp for t in ingestion.speaker_turns] == ["10:30", "10:31"]


def test_day3_chunks_used_by_day4_are_still_valid():
    path = SAMPLES / "chunking_realistic_meeting.txt"
    turns = detect_speaker_turns(normalize_text(path.read_text(encoding="utf-8")))
    chunking = chunk_file(path)
    check = validate_chunks(chunking.chunks, chunking.config, turns)
    assert check.is_valid, check.issues
