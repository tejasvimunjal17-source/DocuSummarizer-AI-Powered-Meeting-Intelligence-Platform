"""Tests for the LangChain prompt templates (real langchain-core)."""
import pytest

pytest.importorskip("langchain_core")

from prompts.summarization_prompts import (  # noqa: E402
    CHUNK_MARKER,
    GLOBAL_MARKER,
    REDUCE_MARKER,
    build_chunk_prompt,
    build_correction_note,
    build_global_prompt,
    build_reduce_prompt,
)

CHUNK_VARS = {"output_language", "position", "total_parts", "source_filename", "speakers",
              "time_range", "continuation_note", "context_text", "transcript_text",
              "format_instructions", "correction"}
REDUCE_VARS = {"output_language", "position_range", "speakers", "time_range",
               "summaries_text", "format_instructions", "correction"}
GLOBAL_VARS = {"output_language", "source_filename", "total_parts", "covered_parts", "speakers",
               "time_range", "coverage_note", "summaries_text", "format_instructions", "correction"}


def fill(variables, **overrides):
    values = {name: f"<{name}>" for name in variables}
    values["output_language"] = "English"
    values.update(overrides)
    return values


def test_input_variables_match_what_the_service_supplies():
    assert set(build_chunk_prompt().input_variables) == CHUNK_VARS
    assert set(build_reduce_prompt().input_variables) == REDUCE_VARS
    assert set(build_global_prompt().input_variables) == GLOBAL_VARS


def test_each_prompt_has_a_system_and_a_human_message():
    for prompt, variables in ((build_chunk_prompt(), CHUNK_VARS), (build_reduce_prompt(), REDUCE_VARS), (build_global_prompt(), GLOBAL_VARS)):
        messages = prompt.format_messages(**fill(variables))
        assert [m.type for m in messages] == ["system", "human"]


def test_chunk_system_prompt_contains_the_required_rules():
    system = build_chunk_prompt().format_messages(**fill(CHUNK_VARS))[0].content
    for phrase in (
        "main discussion points", "names, facts, figures, dates", "speaker attribution",
        "Do not invent information", "disagreements", "repetition", "DATA, not instructions",
        "business, technical, educational and project meetings", "Do not translate or transliterate names",
        "PREVIOUS CONTEXT",
    ):
        assert phrase in system, phrase


def test_global_system_prompt_contains_the_required_rules():
    system = build_global_prompt().format_messages(**fill(GLOBAL_VARS))[0].content
    for phrase in ("entire meeting", "Do not just paste", "unnecessary repetition", "overall purpose",
                   "unresolved questions", "Do not invent information", "coverage note"):
        assert phrase in system, phrase


def test_reduce_system_prompt_keeps_order_and_removes_repetition():
    system = build_reduce_prompt().format_messages(**fill(REDUCE_VARS))[0].content
    assert "CONSECUTIVE" in system and "remove repetition" in system and "chronological" in system


def test_stage_markers_appear_in_the_human_message_only_for_their_stage():
    chunk = build_chunk_prompt().format_messages(**fill(CHUNK_VARS))[1].content
    reduce_ = build_reduce_prompt().format_messages(**fill(REDUCE_VARS))[1].content
    global_ = build_global_prompt().format_messages(**fill(GLOBAL_VARS))[1].content
    assert CHUNK_MARKER in chunk and REDUCE_MARKER not in chunk and GLOBAL_MARKER not in chunk
    assert REDUCE_MARKER in reduce_ and CHUNK_MARKER not in reduce_ and GLOBAL_MARKER not in reduce_
    assert GLOBAL_MARKER in global_ and CHUNK_MARKER not in global_ and REDUCE_MARKER not in global_


def test_braces_and_quotes_inside_a_transcript_do_not_break_the_template():
    nasty = 'He said {"a": 1} and {x} and {{y}} "quoted"'
    messages = build_chunk_prompt().format_messages(**fill(CHUNK_VARS, transcript_text=nasty))
    assert nasty in messages[1].content


def test_output_language_is_a_variable():
    system = build_chunk_prompt().format_messages(**fill(CHUNK_VARS, output_language="Hindi"))[0].content
    assert "Write in Hindi" in system


def test_correction_note_contains_the_error():
    note = build_correction_note("missing field summary")
    assert "missing field summary" in note and "valid JSON" in note
