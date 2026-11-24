"""Unit tests for chunk models and configuration (real Pydantic)."""
import json

import pytest
from pydantic import ValidationError

from chunking import ChunkingConfig, ChunkingResult, ChunkSegment, TextChunk
from chunking.text_utils import compact, count_words, ends_with_sentence_boundary, label_prefix


def test_default_config_values():
    config = ChunkingConfig()
    assert config.chunk_size == 1500
    assert config.chunk_overlap == 200


def test_custom_config_accepted():
    config = ChunkingConfig(chunk_size=400, chunk_overlap=100)
    assert (config.chunk_size, config.chunk_overlap) == (400, 100)


def test_zero_overlap_is_allowed():
    assert ChunkingConfig(chunk_size=200, chunk_overlap=0).chunk_overlap == 0


@pytest.mark.parametrize("overlap", [300, 301, 1000])
def test_overlap_not_smaller_than_size_rejected(overlap):
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=300, chunk_overlap=overlap)


def test_excessive_overlap_rejected():
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=300, chunk_overlap=200)


def test_negative_overlap_rejected():
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=300, chunk_overlap=-1)


def test_chunk_size_below_minimum_rejected():
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=10, chunk_overlap=0)


def test_config_from_env_defaults_when_not_set():
    config = ChunkingConfig.from_env(env={})
    assert (config.chunk_size, config.chunk_overlap) == (1500, 200)


def test_config_from_env_reads_values():
    config = ChunkingConfig.from_env(env={"CHUNK_SIZE": "800", "CHUNK_OVERLAP": "100"})
    assert (config.chunk_size, config.chunk_overlap) == (800, 100)


def test_config_from_env_rejects_non_numbers():
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        ChunkingConfig.from_env(env={"CHUNK_SIZE": "big"})


def test_config_from_env_validates_overlap():
    with pytest.raises(ValueError):
        ChunkingConfig.from_env(env={"CHUNK_SIZE": "200", "CHUNK_OVERLAP": "200"})


def test_label_prefix_variants():
    assert label_prefix("Alice", "10:00 AM") == "[10:00 AM] Alice: "
    assert label_prefix("Alice", None) == "Alice: "
    assert label_prefix(None, "10:00") == "[10:00] "
    assert label_prefix(None, None) == ""


def test_segment_rendering_and_continuation_flag():
    first = ChunkSegment(turn_index=0, speaker="Alice", timestamp="10:00", text="Hello.")
    later = ChunkSegment(
        turn_index=0, speaker="Alice", timestamp="10:00", text="More.",
        fragment_index=1, fragment_count=2,
    )
    assert first.rendered_text() == "[10:00] Alice: Hello."
    assert first.is_continuation is False
    assert later.is_continuation is True


def test_unknown_speaker_segment_has_no_invented_name():
    seg = ChunkSegment(turn_index=3, text="Plain notes.")
    assert seg.speaker is None
    assert seg.rendered_text() == "Plain notes."


def test_chunk_is_json_serializable():
    seg = ChunkSegment(turn_index=0, speaker="Alice", text="Hi.")
    chunk = TextChunk(
        chunk_id="chunk-0000", chunk_index=0, text="Alice: Hi.", character_count=10,
        word_count=2, speakers=["Alice"], speaker_turn_ids=[0], segments=[seg],
    )
    data = json.loads(json.dumps(chunk.model_dump(mode="json")))
    assert data["chunk_id"] == "chunk-0000"
    assert data["segments"][0]["speaker"] == "Alice"
    assert data["start_timestamp"] is None


def test_negative_chunk_index_rejected():
    with pytest.raises(ValidationError):
        TextChunk(chunk_id="x", chunk_index=-1, text="a", character_count=1, word_count=1)


def test_result_defaults_are_empty():
    result = ChunkingResult(config=ChunkingConfig())
    assert result.chunks == [] and result.total_chunks == 0 and result.warnings == []


def test_text_utils_helpers():
    assert ends_with_sentence_boundary("Done.")
    assert ends_with_sentence_boundary('He said "yes."')
    assert not ends_with_sentence_boundary("and then we")
    assert compact(" a \n b\tc ") == "abc"
    assert count_words("one  two\nthree") == 3
