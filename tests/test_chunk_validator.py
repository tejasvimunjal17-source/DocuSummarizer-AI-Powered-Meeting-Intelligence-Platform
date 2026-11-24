"""Tests for chunk validation (real Pydantic / LangChain)."""
import pytest

pytest.importorskip("langchain_text_splitters")

from chunking import ChunkingConfig, SpeakerAwareSplitter, validate_chunks  # noqa: E402
from ingestion.models import SpeakerTurn  # noqa: E402

CONFIG = ChunkingConfig(chunk_size=300, chunk_overlap=100)


def make_turns(count=12):
    return [
        SpeakerTurn(
            index=i,
            speaker=("Alice", "Bob")[i % 2],
            timestamp=f"10:{i:02d}",
            text=f"Statement number {i} about the schedule and the budget plan.",
        )
        for i in range(count)
    ]


def make_chunks(turns=None, config=CONFIG):
    return SpeakerAwareSplitter(config).split(turns or make_turns()).chunks


def test_valid_chunks_pass_with_no_issues():
    turns = make_turns()
    chunks = make_chunks(turns)
    result = validate_chunks(chunks, CONFIG, turns)
    assert result.is_valid, result.issues
    assert result.issues == []
    assert result.chunk_count == len(chunks) >= 3


def test_no_chunks_is_valid_without_source_but_reports_missing_content_with_source():
    assert validate_chunks([], CONFIG).is_valid
    result = validate_chunks([], CONFIG, make_turns(2))
    assert not result.is_valid and "content_missing" in result.codes()


def test_empty_chunk_rejected():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"text": "   "})] + chunks[1:]
    result = validate_chunks(bad, CONFIG)
    assert not result.is_valid and "empty_chunk" in result.codes()


def test_invalid_chunk_index_reported():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"chunk_index": 5})] + chunks[1:]
    result = validate_chunks(bad, CONFIG)
    assert "invalid_index" in result.codes()


def test_incorrect_character_count_reported():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"character_count": 1})] + chunks[1:]
    assert "wrong_character_count" in validate_chunks(bad, CONFIG).codes()


def test_incorrect_word_count_reported():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"word_count": 999})] + chunks[1:]
    assert "wrong_word_count" in validate_chunks(bad, CONFIG).codes()


def test_missing_speaker_metadata_reported():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"speakers": []})] + chunks[1:]
    assert "missing_speaker_metadata" in validate_chunks(bad, CONFIG).codes()


def test_wrong_speaker_metadata_reported():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"speakers": ["Mallory"]})] + chunks[1:]
    assert "invalid_speaker_metadata" in validate_chunks(bad, CONFIG).codes()


def test_wrong_timestamp_metadata_reported():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"start_timestamp": "23:59"})] + chunks[1:]
    assert "invalid_timestamps" in validate_chunks(bad, CONFIG).codes()


def test_invalid_overlap_configuration_reported():
    bad_config = ChunkingConfig.model_construct(chunk_size=100, chunk_overlap=100)
    result = validate_chunks([], bad_config)
    assert not result.is_valid and "invalid_overlap" in result.codes()


def test_overlap_larger_than_configured_reported():
    chunks = make_chunks()
    strict = ChunkingConfig(chunk_size=300, chunk_overlap=10)
    result = validate_chunks(chunks, strict)
    assert "excessive_overlap_content" in result.codes()


def test_oversized_multi_segment_chunk_is_an_error():
    chunks = make_chunks()
    tiny = ChunkingConfig(chunk_size=50, chunk_overlap=0)
    result = validate_chunks(chunks, tiny)
    assert not result.is_valid
    assert "chunk_too_large" in result.codes()


def test_unsplittable_oversized_chunk_is_reported_as_warning_not_hidden():
    turns = [SpeakerTurn(index=0, speaker="Alice", text="x" * 500)]
    config = ChunkingConfig(chunk_size=200, chunk_overlap=0)
    chunks = SpeakerAwareSplitter(config).split(turns).chunks
    result = validate_chunks(chunks, config, turns)
    assert result.is_valid  # content is intact
    assert "oversized_unsplittable" in result.codes()
    assert len(result.warnings) == 1 and result.errors == []


def test_lost_content_detected():
    turns = make_turns()
    chunks = make_chunks(turns)
    result = validate_chunks(chunks[:-1], CONFIG, turns)
    assert not result.is_valid and "content_missing" in result.codes()


def test_attribution_change_detected():
    turns = make_turns()
    chunks = make_chunks(turns)
    changed = [t.model_copy(update={"speaker": "Mallory"}) if t.index == 3 else t for t in turns]
    result = validate_chunks(chunks, CONFIG, changed)
    assert "attribution_mismatch" in result.codes()


def test_out_of_order_chunks_detected():
    chunks = make_chunks()
    first = chunks[0].model_copy(update={"chunk_index": 1})
    second = chunks[1].model_copy(update={"chunk_index": 0})
    result = validate_chunks([second, first], CONFIG)
    assert "out_of_order" in result.codes()


def test_text_not_matching_segments_detected():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"text": chunks[0].text + " extra"})] + chunks[1:]
    assert "text_mismatch" in validate_chunks(bad, CONFIG).codes()


def test_result_separates_errors_and_warnings():
    chunks = make_chunks()
    bad = [chunks[0].model_copy(update={"word_count": 999})] + chunks[1:]
    result = validate_chunks(bad, CONFIG)
    assert result.errors and result.is_valid is False
