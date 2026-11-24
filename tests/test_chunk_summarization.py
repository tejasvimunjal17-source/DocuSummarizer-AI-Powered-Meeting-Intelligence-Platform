"""Per-chunk summarization tests with a MOCKED LLM (real LangChain chains, parsers
and Pydantic models; no network, no API key)."""
import inspect
import time

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402

import services.summarization_service as service_module  # noqa: E402
from chunking import ChunkSegment, TextChunk  # noqa: E402
from config.settings import ConfigurationError  # noqa: E402
from schemas.summary_models import SummarizationConfig, SummaryStatus  # noqa: E402
from services.summarization_service import SummarizationService  # noqa: E402
from tests.llm_fakes import FakeLLM, chunk_json, sequence  # noqa: E402


class RateLimitError(Exception):
    pass


def make_chunk(index, speakers=("Alice", "Bob"), timestamps=("10:00", "10:05"),
               overlap_text=None, is_continuation=False, filename="meeting.txt"):
    segments = []
    if overlap_text:
        segments.append(ChunkSegment(turn_index=index * 10, speaker=speakers[0] if speakers else None,
                                     text=overlap_text, is_overlap=True))
    for i in range(max(len(speakers), 1)):
        continued = is_continuation and i == 0
        segments.append(ChunkSegment(
            turn_index=index * 10 + i + 1,
            speaker=speakers[i] if speakers else None,
            timestamp=timestamps[i] if i < len(timestamps) else None,
            text=f"Statement {index} number {i}.",
            fragment_index=1 if continued else 0,
            fragment_count=2 if continued else 1,
        ))
    text = "\n".join(s.rendered_text() for s in segments)
    stamps = [s.timestamp for s in segments if s.timestamp]
    turn_ids = []
    for s in segments:
        if s.turn_index not in turn_ids:
            turn_ids.append(s.turn_index)
    return TextChunk(
        chunk_id=f"chunk-{index:04d}", chunk_index=index, text=text, character_count=len(text),
        word_count=len(text.split()), speakers=list(speakers), speaker_turn_ids=turn_ids,
        start_timestamp=stamps[0] if stamps else None, end_timestamp=stamps[-1] if stamps else None,
        is_continuation=is_continuation, overlap_character_count=len(overlap_text) if overlap_text else 0,
        source_filename=filename, segments=segments,
    )


def make_service(fake, sleeps=None, **config):
    config.setdefault("retry_delay_seconds", 0.5)
    config.setdefault("timeout_seconds", 5)
    sleeps = sleeps if sleeps is not None else []
    return SummarizationService(fake.runnable(), SummarizationConfig(**config), sleep=sleeps.append)


# ------------------------------------------------------------ valid summary
def test_valid_chunk_summary_copies_metadata_from_the_chunk():
    fake = FakeLLM()
    chunk = make_chunk(0)
    result = make_service(fake).summarize_chunk(chunk, total_chunks=3)
    assert result.status == SummaryStatus.SUCCESS and result.error_message is None
    assert result.chunk_id == "chunk-0000" and result.chunk_index == 0
    assert result.summary == "Summary of part 1." and result.key_points == ["Point from part 1"]
    assert result.speakers == ["Alice", "Bob"] and result.speaker_turn_ids == chunk.speaker_turn_ids
    assert (result.start_timestamp, result.end_timestamp) == ("10:00", "10:05")
    assert result.source_filename == "meeting.txt" and result.word_count == chunk.word_count
    assert result.attempts == 1 and fake.calls == 1


def test_prompt_contains_speakers_timestamps_and_statements():
    fake = FakeLLM()
    make_service(fake).summarize_chunk(make_chunk(0), total_chunks=3)
    prompt = fake.by_stage["chunk"][0]
    assert "Speakers in this part: Alice, Bob" in prompt
    assert "Time range: 10:00 \u2013 10:05" in prompt
    assert "[10:00] Alice: Statement 0 number 0." in prompt
    assert "Meeting part 1 of 3." in prompt and "Source file: meeting.txt" in prompt


def test_overlap_is_shown_as_context_not_as_text_to_summarize():
    fake = FakeLLM()
    make_service(fake).summarize_chunk(make_chunk(1, overlap_text="OVERLAP-SENTENCE repeated."), total_chunks=2)
    before, after = fake.by_stage["chunk"][0].split("TRANSCRIPT PART TO SUMMARIZE")
    assert "OVERLAP-SENTENCE" in before and "OVERLAP-SENTENCE" not in after
    assert "Statement 1 number 0." in after


def test_overlap_context_can_be_switched_off():
    fake = FakeLLM()
    make_service(fake, include_overlap_context=False).summarize_chunk(make_chunk(1, overlap_text="OVERLAP-SENTENCE"))
    assert "OVERLAP-SENTENCE" not in fake.by_stage["chunk"][0]


def test_continuation_note_only_for_continued_chunks():
    fake = FakeLLM()
    service = make_service(fake)
    service.summarize_chunk(make_chunk(1, is_continuation=True))
    service.summarize_chunk(make_chunk(2))
    assert "begins in the middle" in fake.by_stage["chunk"][0]
    assert "begins in the middle" not in fake.by_stage["chunk"][1]


def test_missing_speakers_and_timestamps_are_handled():
    fake = FakeLLM()
    result = make_service(fake).summarize_chunk(make_chunk(0, speakers=(), timestamps=()))
    prompt = fake.by_stage["chunk"][0]
    assert result.status == SummaryStatus.SUCCESS and result.speakers == []
    assert result.start_timestamp is None and result.end_timestamp is None
    assert "not labeled" in prompt and "Time range: not available" in prompt


# ---------------------------------------------------- empty / invalid chunks
def test_empty_chunk_is_skipped_without_calling_the_llm():
    fake = FakeLLM()
    empty = TextChunk(chunk_id="chunk-0000", chunk_index=0, text="   ", character_count=3, word_count=0)
    result = make_service(fake).summarize_chunk(empty)
    assert result.status == SummaryStatus.SKIPPED and "empty" in result.error_message
    assert fake.calls == 0


def test_chunk_with_only_overlap_is_skipped():
    fake = FakeLLM()
    segment = ChunkSegment(turn_index=0, speaker="Alice", text="Only overlap.", is_overlap=True)
    chunk = TextChunk(chunk_id="chunk-0001", chunk_index=1, text="Alice: Only overlap.", character_count=20,
                      word_count=3, speakers=["Alice"], segments=[segment])
    assert make_service(fake).summarize_chunk(chunk).status == SummaryStatus.SKIPPED
    assert fake.calls == 0


@pytest.mark.parametrize("bad", ["just a string", {"text": "dict"}, None, 42])
def test_invalid_chunk_objects_fail_safely(bad):
    fake = FakeLLM()
    result = make_service(fake).summarize_chunk(bad)
    assert result.status == SummaryStatus.FAILED and "Invalid chunk" in result.error_message
    assert result.chunk_id == "invalid-0" and fake.calls == 0


def test_chunk_without_id_fails_safely():
    fake = FakeLLM()
    chunk = make_chunk(0).model_copy(update={"chunk_id": "  "})
    result = make_service(fake).summarize_chunk(chunk, position=7)
    assert result.status == SummaryStatus.FAILED and result.chunk_id == "invalid-7"
    assert fake.calls == 0


def test_mixed_valid_and_invalid_chunks_keep_positions():
    fake = FakeLLM()
    results = make_service(fake).summarize_chunks([make_chunk(0), "bad", make_chunk(2)])
    assert [r.chunk_id for r in results] == ["chunk-0000", "invalid-1", "chunk-0002"]
    assert [r.status for r in results] == [SummaryStatus.SUCCESS, SummaryStatus.FAILED, SummaryStatus.SUCCESS]


# -------------------------------------------------- many chunks and ordering
def test_multiple_chunks_are_summarized_in_order():
    fake = FakeLLM()
    results = make_service(fake).summarize_chunks([make_chunk(i) for i in range(5)])
    assert [r.chunk_id for r in results] == [f"chunk-{i:04d}" for i in range(5)]
    assert [r.summary for r in results] == [f"Summary of part {i + 1}." for i in range(5)]
    assert len(fake.by_stage["chunk"]) == 5
    for i, prompt in enumerate(fake.by_stage["chunk"]):
        assert f"Meeting part {i + 1} of 5." in prompt


def test_out_of_order_chunks_are_sorted_with_a_warning():
    fake = FakeLLM()
    warnings = []
    results = make_service(fake).summarize_chunks([make_chunk(2), make_chunk(0), make_chunk(1)], warnings)
    assert [r.chunk_index for r in results] == [0, 1, 2]
    assert any("chronological" in w for w in warnings)


# ------------------------------------------------------- response formats
def test_json_in_a_code_fence_is_accepted():
    fake = FakeLLM(chunk=sequence("```json\n" + chunk_json(1) + "\n```"))
    assert make_service(fake).summarize_chunk(make_chunk(0)).status == SummaryStatus.SUCCESS


def test_chat_message_with_content_blocks_is_accepted():
    message = AIMessage(content=[{"type": "text", "text": chunk_json(1)}])
    fake = FakeLLM(chunk=sequence(message))
    result = make_service(fake).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.SUCCESS and result.summary == "Summary of part 1."


# -------------------------------------------------------- malformed output
def test_malformed_output_is_retried_with_a_correction_message():
    fake = FakeLLM(chunk=sequence("this is not json", chunk_json(1)))
    sleeps = []
    service = make_service(fake, sleeps)
    result = service.summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.SUCCESS and result.attempts == 2
    assert service.retry_count == 1 and sleeps == []
    assert "IMPORTANT: your previous reply could not be used" not in fake.by_stage["chunk"][0]
    assert "IMPORTANT: your previous reply could not be used" in fake.by_stage["chunk"][1]


def test_missing_required_field_counts_as_malformed():
    fake = FakeLLM(chunk=sequence('{"summary": "", "key_points": []}', chunk_json(1)))
    result = make_service(fake).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.SUCCESS and result.attempts == 2


def test_always_malformed_output_fails_without_crashing():
    fake = FakeLLM(chunk=sequence("garbage"))
    result = make_service(fake).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.FAILED and result.summary == ""
    assert "malformed_output" in result.error_message and result.attempts == 3
    assert fake.calls == 3


def test_wrong_types_in_output_are_rejected():
    fake = FakeLLM(chunk=sequence('{"summary": "ok", "key_points": [1, 2]}'))
    assert make_service(fake).summarize_chunk(make_chunk(0)).status == SummaryStatus.FAILED


def test_zero_retries_means_one_attempt():
    fake = FakeLLM(chunk=sequence("garbage"))
    result = make_service(fake, max_retries=0).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.FAILED and fake.calls == 1


# ----------------------------------------------------------- provider errors
def test_provider_error_is_retried_with_backoff():
    fake = FakeLLM(chunk=sequence(RuntimeError("503 service unavailable"), chunk_json(1)))
    sleeps = []
    result = make_service(fake, sleeps).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.SUCCESS and result.attempts == 2
    assert sleeps == [0.5]


def test_backoff_doubles_between_retries():
    fake = FakeLLM(chunk=sequence(RuntimeError("boom"), RuntimeError("boom"), chunk_json(1)))
    sleeps = []
    result = make_service(fake, sleeps).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.SUCCESS and sleeps == [0.5, 1.0]


def test_rate_limit_errors_are_retried():
    fake = FakeLLM(chunk=sequence(RateLimitError("slow down"), chunk_json(1)))
    result = make_service(fake).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.SUCCESS and result.attempts == 2


def test_exhausted_retries_return_a_failed_summary_not_an_exception():
    fake = FakeLLM(chunk=sequence(RuntimeError("server error")))
    result = make_service(fake).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.FAILED and "api_error" in result.error_message
    assert result.attempts == 3 and fake.calls == 3


def test_timeout_is_reported():
    fake = FakeLLM(chunk=sequence(lambda text, n: (time.sleep(0.5), chunk_json(1))[1]))
    result = make_service(fake, max_retries=0, timeout_seconds=0.05).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.FAILED and "timeout" in result.error_message


def test_empty_response_is_retried_then_reported():
    fake = FakeLLM(chunk=sequence("   "))
    result = make_service(fake).summarize_chunk(make_chunk(0))
    assert result.status == SummaryStatus.FAILED and "empty_response" in result.error_message
    assert fake.calls == 3


def test_authentication_error_is_not_retried_and_stops_further_calls():
    fake = FakeLLM(chunk=sequence(Exception("API key not valid. Please pass a valid API key.")))
    results = make_service(fake).summarize_chunks([make_chunk(i) for i in range(3)])
    assert [r.status for r in results] == [SummaryStatus.FAILED] * 3
    assert fake.calls == 1
    assert "Not attempted" in results[1].error_message
    assert "Authentication failed" in results[1].error_message


def test_api_keys_in_error_messages_are_redacted():
    secret = "AIza" + "S" * 35
    fake = FakeLLM(chunk=sequence(RuntimeError(f"request failed ?key={secret}")))
    result = make_service(fake, max_retries=0).summarize_chunk(make_chunk(0))
    assert secret not in result.error_message and "[REDACTED]" in result.error_message


def test_missing_configuration_error_is_not_swallowed(monkeypatch):
    def broken_build_llm(settings=None):
        raise ConfigurationError("GEMINI_API_KEY is missing")

    monkeypatch.setattr(service_module, "build_llm", broken_build_llm)
    with pytest.raises(ConfigurationError):
        SummarizationService(config=SummarizationConfig()).summarize_chunk(make_chunk(0))


def test_summarization_service_does_not_name_a_provider():
    source = inspect.getsource(service_module)
    for forbidden in ("langchain_google_genai", "langchain_openai", "ChatGoogleGenerativeAI", "ChatOpenAI"):
        assert forbidden not in source


def test_call_counters_track_requests_and_retries():
    fake = FakeLLM(chunk=sequence(RuntimeError("x"), chunk_json(1)))
    service = make_service(fake)
    service.summarize_chunk(make_chunk(0))
    assert service.llm_call_count == 2 and service.retry_count == 1
