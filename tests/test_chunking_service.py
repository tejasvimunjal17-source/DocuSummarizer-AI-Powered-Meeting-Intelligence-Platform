"""Integration tests: ingestion -> chunking -> validation (real Pydantic / LangChain)."""
from pathlib import Path

import pytest

pytest.importorskip("langchain_text_splitters")

from chunking import (  # noqa: E402
    ChunkingConfig,
    chunk_file,
    chunk_ingestion_result,
    chunk_text,
    chunk_turns,
    validate_chunks,
)
from chunking.text_utils import compact  # noqa: E402
from ingestion import detect_speaker_turns, ingest_bytes, normalize_text  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "sample_transcripts"


def test_chunk_text_detects_speakers_and_timestamps():
    result = chunk_text("[10:00 AM] Alice: Hello there everyone.\n[10:01 AM] Bob: Hi!")
    assert result.total_chunks == 1
    assert result.chunks[0].speakers == ["Alice", "Bob"]
    assert result.chunks[0].start_timestamp == "10:00 AM"


def test_chunk_text_supports_name_then_timestamp_format():
    result = chunk_text("Alice [10:00 AM]: Welcome everyone.\n\nBob [10:01 AM]: I will do it.")
    assert result.chunks[0].speakers == ["Alice", "Bob"]
    assert result.chunks[0].end_timestamp == "10:01 AM"


@pytest.mark.parametrize("text", ["", "   \n\n  ", "\u200b \u00a0"])
def test_empty_transcript_is_handled_safely(text):
    result = chunk_text(text)
    assert result.chunks == [] and result.total_chunks == 0
    assert result.source_turn_count == 0


def test_statistics_are_reported():
    turns = detect_speaker_turns(normalize_text(
        "Alice: " + "Sentence one is here. " * 40 + "\n\nBob: Short reply."
    ))
    result = chunk_turns(turns, ChunkingConfig(chunk_size=300, chunk_overlap=50))
    assert result.total_chunks == len(result.chunks) >= 2
    assert result.source_turn_count == 2
    assert result.split_turn_count == 1
    assert result.source_character_count == sum(len(t.text) for t in turns)
    assert result.total_chunk_characters == sum(c.character_count for c in result.chunks)
    assert result.processing_seconds >= 0


def test_ingestion_result_integration_sets_source_filename():
    ingestion = ingest_bytes("meeting.txt", b"Alice: Hello there.\n\nBob: Hi Alice.")
    result = chunk_ingestion_result(ingestion)
    assert result.source_filename == "meeting.txt"
    assert all(c.source_filename == "meeting.txt" for c in result.chunks)
    assert result.chunks[0].speakers == ["Alice", "Bob"]


def test_chunk_file_with_short_sample():
    result = chunk_file(SAMPLES / "chunking_short_meeting.txt")
    assert result.total_chunks == 1
    assert result.chunks[0].speakers == ["Alice", "Bob", "Charlie"]
    assert result.chunks[0].start_timestamp == "10:00 AM"
    assert result.chunks[0].end_timestamp == "10:03 AM"


def test_chunk_file_with_long_turns_sample_divides_long_turn():
    config = ChunkingConfig(chunk_size=400, chunk_overlap=80)
    result = chunk_file(SAMPLES / "chunking_long_turns.txt", config)
    ingestion_turns = detect_speaker_turns(normalize_text(
        (SAMPLES / "chunking_long_turns.txt").read_text(encoding="utf-8")
    ))
    assert result.split_turn_count >= 1
    assert result.oversized_chunk_count == 0
    check = validate_chunks(result.chunks, config, ingestion_turns)
    assert check.is_valid, check.issues
    dana_fragments = [
        s for c in result.chunks for s in c.segments
        if not s.is_overlap and s.speaker == "Dana" and s.fragment_count > 1
    ]
    assert dana_fragments and all(s.timestamp == "09:01" for s in dana_fragments)


def test_chunk_file_with_unlabeled_sections_sample():
    result = chunk_file(SAMPLES / "chunking_mixed_unlabeled.txt", ChunkingConfig(chunk_size=300, chunk_overlap=50))
    assert result.total_chunks >= 2
    unknown = [s for c in result.chunks for s in c.segments if not s.is_overlap and s.speaker is None]
    assert len(unknown) == 2
    assert any("Weekly Sync Notes" in s.text for s in unknown)
    all_speakers = {n for c in result.chunks for n in c.speakers}
    assert all_speakers == {"Gita", "Hari"}


def test_chunk_file_with_multilingual_sample():
    result = chunk_file(SAMPLES / "chunking_multilingual.txt")
    names = result.chunks[0].speakers
    assert "\u0930\u093e\u0939\u0941\u0932" in names and "\u674e\u96f7" in names and "Maria" in names


def test_realistic_meeting_with_default_settings():
    path = SAMPLES / "chunking_realistic_meeting.txt"
    turns = detect_speaker_turns(normalize_text(path.read_text(encoding="utf-8")))
    result = chunk_file(path)
    assert result.config.chunk_size == 1500 and result.config.chunk_overlap == 200
    assert result.total_chunks > 5
    assert result.source_turn_count == len(turns)
    assert set(result.chunks[0].speakers) <= {"Alice", "Bob", "Charlie", "Dana", "Eli"}
    check = validate_chunks(result.chunks, result.config, turns)
    assert check.is_valid, check.issues
    assert result.oversized_chunk_count == 0


def test_realistic_meeting_with_small_chunks_loses_nothing():
    path = SAMPLES / "chunking_realistic_meeting.txt"
    turns = detect_speaker_turns(normalize_text(path.read_text(encoding="utf-8")))
    config = ChunkingConfig(chunk_size=500, chunk_overlap=100)
    result = chunk_file(path, config)
    assert result.split_turn_count > 0  # some turns are longer than 500 characters
    check = validate_chunks(result.chunks, config, turns)
    assert check.is_valid, check.issues
    rebuilt = compact("".join(s.text for c in result.chunks for s in c.segments if not s.is_overlap))
    assert rebuilt == compact("".join(t.text for t in turns))


def test_docx_file_through_chunking(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for i in range(30):
        document.add_paragraph(f"[10:{i:02d}] Alice: Point {i} about the plan and the budget for the quarter.")
        document.add_paragraph(f"[10:{i:02d}] Bob: Agreed on point {i}, and I will follow up with the team.")
    path = tmp_path / "meeting.docx"
    document.save(str(path))
    result = chunk_file(path, ChunkingConfig(chunk_size=500, chunk_overlap=100))
    assert result.total_chunks > 1
    assert {n for c in result.chunks for n in c.speakers} == {"Alice", "Bob"}


def test_large_transcript_of_50000_words_is_chunked_and_validated():
    """Real run of the full chunking path on >= 50,000 words (not an API test)."""
    speakers = ["Alice", "Bob", "Charlie", "Dana", "Eli"]
    lines, words, i = [], 0, 0
    while words < 50_000:
        body = " ".join(
            f"Item {i}-{j} needs a review by the team before the next milestone."
            for j in range(4)
        )
        lines.append(f"[{10 + (i // 60) % 12}:{i % 60:02d}] {speakers[i % 5]}: {body}")
        words += len(body.split())
        i += 1
    text = "\n\n".join(lines)
    turns = detect_speaker_turns(normalize_text(text))
    result = chunk_turns(turns, ChunkingConfig())
    assert result.source_word_count >= 50_000
    assert result.total_chunks > 100
    assert result.oversized_chunk_count == 0
    check = validate_chunks(result.chunks, result.config, turns)
    assert check.is_valid, check.issues[:3]
    assert result.processing_seconds < 120
