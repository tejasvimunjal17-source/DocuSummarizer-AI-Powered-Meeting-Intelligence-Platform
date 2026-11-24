"""Tests for chunk previews (real Pydantic / LangChain)."""
import pytest

pytest.importorskip("langchain_text_splitters")

from chunking import (  # noqa: E402
    ChunkingConfig,
    build_chunk_previews,
    chunk_text,
    chunk_turns,
    format_chunk_preview,
    print_chunk_preview,
)
from ingestion.models import SpeakerTurn  # noqa: E402


def many_turns(count=40):
    return [
        SpeakerTurn(index=i, speaker=("Alice", "Bob")[i % 2], timestamp=f"10:{i % 60:02d}",
                    text=f"Statement number {i} about the schedule and the budget plan.")
        for i in range(count)
    ]


def make_result(count=40, size=300, overlap=60):
    return chunk_turns(many_turns(count), ChunkingConfig(chunk_size=size, chunk_overlap=overlap))


def test_preview_limits_number_of_chunks():
    result = make_result()
    assert result.total_chunks > 5
    previews = build_chunk_previews(result.chunks, max_chunks=3)
    assert len(previews) == 3
    assert [p.chunk_index for p in previews] == [0, 1, 2]


def test_preview_contains_required_information():
    result = make_result()
    first = build_chunk_previews(result.chunks, max_chunks=1)[0]
    chunk = result.chunks[0]
    assert first.character_count == chunk.character_count
    assert first.word_count == chunk.word_count
    assert first.speakers == chunk.speakers
    assert first.is_continuation is False
    assert first.chunk_id == "chunk-0000"


def test_preview_text_is_shortened_and_flattened():
    result = make_result()
    preview = build_chunk_previews(result.chunks, max_chunks=1, preview_chars=40)[0].preview
    assert len(preview) <= 41  # 40 characters plus the ellipsis
    assert preview.endswith("\u2026")
    assert "\n" not in preview


def test_short_chunk_preview_is_not_cut():
    result = chunk_text("Alice: Hello there.")
    preview = build_chunk_previews(result.chunks)[0].preview
    assert preview == "Alice: Hello there."


def test_preview_does_not_modify_chunks():
    result = make_result()
    before = [c.text for c in result.chunks]
    build_chunk_previews(result.chunks, max_chunks=2, preview_chars=10)
    assert [c.text for c in result.chunks] == before


def test_invalid_preview_arguments_rejected():
    result = make_result()
    with pytest.raises(ValueError):
        build_chunk_previews(result.chunks, max_chunks=-1)
    with pytest.raises(ValueError):
        build_chunk_previews(result.chunks, preview_chars=0)


def test_formatted_preview_has_totals_and_limit():
    result = make_result()
    text = format_chunk_preview(result, max_chunks=2, preview_chars=60)
    assert f"Total chunks: {result.total_chunks}" in text
    assert f"Showing 2 of {result.total_chunks} chunks" in text
    assert "[0]" in text and "[1]" in text and "[2]" not in text
    assert "speakers: Alice, Bob" in text
    assert "continuation: no" in text


def test_formatted_preview_marks_continuations():
    turns = [SpeakerTurn(index=0, speaker="Alice", text=" ".join(
        f"Sentence number {i} explains one separate point clearly." for i in range(30)))]
    result = chunk_turns(turns, ChunkingConfig(chunk_size=400, chunk_overlap=0))
    assert "continuation: yes" in format_chunk_preview(result, max_chunks=3)


def test_formatted_preview_of_empty_transcript():
    text = format_chunk_preview(chunk_text(""))
    assert "Total chunks: 0" in text and "No chunks" in text


def test_preview_of_large_transcript_stays_small():
    result = make_result(count=2000, size=1500, overlap=200)
    text = format_chunk_preview(result, max_chunks=3, preview_chars=80)
    assert result.total_chunks > 10
    assert len(text) < 1500
    assert len(text) < sum(c.character_count for c in result.chunks) / 20


def test_print_chunk_preview_writes_to_stdout(capsys):
    print_chunk_preview(make_result(), max_chunks=1)
    out = capsys.readouterr().out
    assert "Total chunks:" in out and "[0]" in out
