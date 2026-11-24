"""Tests for the speaker-aware splitter (uses the REAL LangChain
RecursiveCharacterTextSplitter; skipped automatically if it is not installed)."""
import pytest

pytest.importorskip("langchain_text_splitters")

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from chunking import ChunkingConfig, SpeakerAwareSplitter  # noqa: E402
from chunking.speaker_aware_splitter import create_turn_splitter  # noqa: E402
from chunking.text_utils import compact  # noqa: E402
from ingestion.models import SpeakerTurn  # noqa: E402

HINDI_NAME = "\u0930\u093e\u0939\u0941\u0932"
CHINESE_NAME = "\u674e\u96f7"


def turn(index, speaker, text, timestamp=None):
    return SpeakerTurn(index=index, speaker=speaker, timestamp=timestamp, text=text)


def split(turns, size=300, overlap=0):
    config = ChunkingConfig(chunk_size=size, chunk_overlap=overlap)
    return SpeakerAwareSplitter(config).split(turns)


def new_segments(chunks):
    return [s for c in chunks for s in c.segments if not s.is_overlap]


def assert_no_content_lost(turns, chunks):
    by_turn = {}
    for seg in new_segments(chunks):
        by_turn.setdefault(seg.turn_index, []).append(seg.text)
    for t in turns:
        if t.text.strip():
            assert compact("".join(by_turn[t.index])) == compact(t.text)


def statements(count, speakers=("Alice", "Bob"), with_time=False):
    return [
        turn(
            i,
            speakers[i % len(speakers)],
            f"Statement number {i} about the schedule and the budget plan.",
            f"10:{i:02d}" if with_time else None,
        )
        for i in range(count)
    ]


def long_text(sentences=30):
    return " ".join(f"Sentence number {i} explains one separate point clearly." for i in range(sentences))


# ------------------------------------------------------------ real LangChain
def test_real_recursive_character_text_splitter_is_used():
    splitter = create_turn_splitter(100)
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 100
    assert splitter._chunk_overlap == 0


# ------------------------------------------------------------------- basics
def test_empty_transcript_gives_no_chunks():
    output = split([])
    assert output.chunks == [] and output.warnings == []


def test_whitespace_only_turns_are_skipped():
    output = split([turn(0, "Alice", "   ")])
    assert output.chunks == []


def test_short_transcript_is_one_chunk_with_all_speakers():
    turns = [
        turn(0, "Alice", "Welcome everyone. Today we will discuss the project timeline.", "10:00 AM"),
        turn(1, "Bob", "I will complete the backend implementation by Friday.", "10:01 AM"),
        turn(2, "Alice", "Great. We also need to review the security requirements.", "10:02 AM"),
        turn(3, "Charlie", "I will prepare the security checklist.", "10:03 AM"),
    ]
    chunks = split(turns, size=1500, overlap=200).chunks
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "chunk-0000" and chunk.chunk_index == 0
    assert chunk.speakers == ["Alice", "Bob", "Charlie"]
    assert chunk.speaker_turn_ids == [0, 1, 2, 3]
    assert chunk.start_timestamp == "10:00 AM" and chunk.end_timestamp == "10:03 AM"
    assert "[10:00 AM] Alice: Welcome everyone." in chunk.text
    assert chunk.is_continuation is False and chunk.overlap_character_count == 0
    assert chunk.character_count == len(chunk.text)
    assert chunk.word_count == len(chunk.text.split())


def test_source_filename_is_stored_on_chunks():
    config = ChunkingConfig(chunk_size=300, chunk_overlap=0)
    output = SpeakerAwareSplitter(config).split(statements(6), source_filename="m.txt")
    assert all(c.source_filename == "m.txt" for c in output.chunks)


def test_many_turns_make_multiple_chunks_within_limit():
    turns = statements(12)
    chunks = split(turns, size=350, overlap=0).chunks
    assert len(chunks) > 1
    assert all(c.character_count <= 350 for c in chunks)
    assert_no_content_lost(turns, chunks)


@pytest.mark.parametrize("size", [150, 300, 600])
def test_configurable_chunk_size_is_respected(size):
    turns = statements(30)
    output = split(turns, size=size, overlap=0)
    assert all(c.character_count <= size for c in output.chunks)
    assert output.warnings == []
    assert_no_content_lost(turns, output.chunks)


def test_smaller_size_gives_more_chunks():
    turns = statements(30)
    assert len(split(turns, size=200).chunks) > len(split(turns, size=800).chunks)


# ---------------------------------------------------- speaker-aware boundaries
def test_chunks_split_between_whole_turns():
    turns = statements(12)
    chunks = split(turns, size=300, overlap=0).chunks
    for seg in new_segments(chunks):
        assert seg.fragment_count == 1 and seg.fragment_index == 0
    for chunk in chunks:
        assert chunk.is_continuation is False


def test_one_speaker_consecutive_statements_stay_separate_turns():
    turns = [turn(i, "Alice", f"Alice makes point number {i} about the roadmap and the plan.") for i in range(8)]
    chunks = split(turns, size=250).chunks
    assert len(chunks) > 1
    assert all(c.speakers == ["Alice"] for c in chunks)
    assert [s.turn_index for s in new_segments(chunks)] == list(range(8))


def test_many_speakers_are_all_preserved():
    names = [f"Person{chr(65 + i)}" for i in range(12)]
    turns = [turn(i, names[i], f"This is what {names[i]} wants to say today.") for i in range(12)]
    chunks = split(turns, size=300).chunks
    assert len(chunks) > 1
    seen = []
    for c in chunks:
        for name in c.speakers:
            if name not in seen:
                seen.append(name)
    assert seen == names


def test_missing_speaker_labels_stay_unknown():
    turns = [turn(i, None, f"Plain note line number {i} without any speaker.") for i in range(4)]
    chunk = split(turns, size=1500).chunks[0]
    assert chunk.speakers == []
    assert chunk.start_timestamp is None
    assert chunk.text.startswith("Plain note line number 0")
    assert ": " not in chunk.text.split("\n")[0].replace("Plain note", "")


def test_mixed_unknown_and_known_speakers():
    turns = [turn(0, None, "Introductory notes."), turn(1, "Alice", "Hello team.")]
    chunk = split(turns, size=1500).chunks[0]
    assert chunk.speakers == ["Alice"]
    assert chunk.segments[0].speaker is None
    assert chunk.text == "Introductory notes.\nAlice: Hello team."


def test_multiline_statement_stays_together():
    text = "First line of the point\nsecond line of the point\nthird line of the point"
    chunk = split([turn(0, "Alice", text)], size=1500).chunks[0]
    assert text in chunk.text


def test_timestamps_are_preserved_in_order():
    turns = statements(12, with_time=True)
    chunks = split(turns, size=300).chunks
    assert [s.timestamp for s in new_segments(chunks)] == [t.timestamp for t in turns]
    for chunk in chunks:
        stamps = [s.timestamp for s in chunk.segments if s.timestamp]
        assert chunk.start_timestamp == stamps[0]
        assert chunk.end_timestamp == stamps[-1]


def test_non_latin_speaker_names_are_preserved():
    turns = [
        turn(0, HINDI_NAME, "Hello from the first speaker."),
        turn(1, CHINESE_NAME, "Hello from the second speaker."),
    ]
    chunk = split(turns, size=1500).chunks[0]
    assert chunk.speakers == [HINDI_NAME, CHINESE_NAME]
    assert f"{HINDI_NAME}: Hello" in chunk.text


# ------------------------------------------------------------ long turns
def test_long_turn_is_divided_at_sentence_boundaries_and_keeps_speaker():
    text = long_text(30)
    turns = [turn(0, "Alice", text, "10:05")]
    output = split(turns, size=400)
    chunks = output.chunks
    assert len(chunks) >= 4
    assert output.split_turn_count == 1
    segments = new_segments(chunks)
    assert [s.fragment_index for s in segments] == list(range(len(segments)))
    assert all(s.fragment_count == len(segments) for s in segments)
    assert all(s.speaker == "Alice" and s.timestamp == "10:05" for s in segments)
    assert all(c.character_count <= 400 for c in chunks)
    # every fragment except the last ends at a sentence boundary
    assert all(s.text.endswith(".") for s in segments[:-1])
    assert not any(s.ends_mid_sentence for s in segments)
    assert output.warnings == []
    assert_no_content_lost(turns, chunks)


def test_continuation_chunks_repeat_the_speaker_label_and_are_flagged():
    chunks = split([turn(0, "Alice", long_text(30), "10:05")], size=400).chunks
    assert chunks[0].is_continuation is False
    assert all(c.is_continuation for c in chunks[1:])
    assert all(c.text.startswith("[10:05] Alice: ") for c in chunks)
    assert all(c.speakers == ["Alice"] for c in chunks)


def test_long_turn_between_short_turns_keeps_order():
    turns = [
        turn(0, "Bob", "Short opening remark."),
        turn(1, "Alice", long_text(25)),
        turn(2, "Bob", "Short closing remark."),
    ]
    chunks = split(turns, size=400).chunks
    order = [(s.turn_index, s.fragment_index) for s in new_segments(chunks)]
    assert order == sorted(order)
    assert order[0] == (0, 0) and order[-1][0] == 2
    assert_no_content_lost(turns, chunks)


def test_sentence_longer_than_limit_is_divided_at_word_boundaries_and_reported():
    text = " ".join(f"word{i}" for i in range(150))  # no punctuation at all
    turns = [turn(0, "Bob", text)]
    output = split(turns, size=200)
    assert all(c.character_count <= 200 for c in output.chunks)
    segments = new_segments(output.chunks)
    assert len(segments) > 1
    assert all(s.ends_mid_sentence for s in segments[:-1])
    assert any("sentence longer" in w for w in output.warnings)
    assert_no_content_lost(turns, output.chunks)


def test_unbreakable_text_is_kept_in_full_and_reported():
    text = "x" * 500
    output = split([turn(0, "Alice", text)], size=200)
    assert len(output.chunks) == 1
    assert text in output.chunks[0].text  # nothing truncated
    assert output.chunks[0].character_count > 200
    assert any("cannot be divided" in w or "could not be divided" in w for w in output.warnings)


# -------------------------------------------------------------------- overlap
def test_overlap_repeats_whole_turns_from_previous_chunk():
    turns = statements(10)
    chunks = split(turns, size=300, overlap=140).chunks
    assert len(chunks) >= 3
    second = chunks[1]
    overlap_segments = [s for s in second.segments if s.is_overlap]
    assert overlap_segments and second.segments[0].is_overlap
    previous_new = [s for s in chunks[0].segments if not s.is_overlap]
    assert [s.turn_index for s in overlap_segments] == [
        s.turn_index for s in previous_new[-len(overlap_segments):]
    ]
    assert 0 < second.overlap_character_count <= 140
    assert chunks[0].overlap_character_count == 0
    # overlap never causes duplicated NEW content
    assert [s.turn_index for s in new_segments(chunks)] == list(range(10))
    assert_no_content_lost(turns, chunks)


def test_zero_overlap_creates_no_overlap_segments():
    chunks = split(statements(10), size=300, overlap=0).chunks
    assert not any(s.is_overlap for c in chunks for s in c.segments)


def test_chronological_order_is_maintained():
    chunks = split(statements(20, with_time=True), size=250, overlap=100).chunks
    indexes = [s.turn_index for s in new_segments(chunks)]
    assert indexes == sorted(indexes)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
