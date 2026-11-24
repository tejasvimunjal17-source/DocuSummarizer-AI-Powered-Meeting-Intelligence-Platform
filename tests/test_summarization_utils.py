"""Tests for the pure helper functions (standard library only)."""
import time

import pytest

from services.summarization_utils import (
    RETRYABLE_KINDS,
    LLMCallError,
    LLMErrorKind,
    SummarizationError,
    SummaryNode,
    batch_nodes,
    call_with_timeout,
    classify_exception,
    dedupe_preserve_order,
    extract_text,
    format_part_ranges,
    format_time_range,
    merge_node_metadata,
    node_size,
    redact_secrets,
    render_node,
    retry_delay,
    safe_error_message,
    total_size,
)


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class ResourceExhausted(Exception):
    pass


class FakeMessage:
    def __init__(self, content):
        self.content = content


def node(i, text="s" * 80, speakers=("Alice",), start=None, end=None):
    return SummaryNode(
        summary=text, key_points=["k"], speakers=list(speakers), start_timestamp=start,
        end_timestamp=end, first_chunk_index=i, last_chunk_index=i, chunk_ids=[f"chunk-{i:04d}"],
    )


@pytest.mark.parametrize(
    "exc, kind",
    [
        (TimeoutError("x"), LLMErrorKind.TIMEOUT),
        (Exception("Request timed out"), LLMErrorKind.TIMEOUT),
        (RateLimitError("slow down"), LLMErrorKind.RATE_LIMIT),
        (ResourceExhausted("quota"), LLMErrorKind.RATE_LIMIT),
        (Exception("HTTP 429 Too Many Requests"), LLMErrorKind.RATE_LIMIT),
        (AuthenticationError("bad"), LLMErrorKind.AUTH),
        (Exception("API key not valid. Please pass a valid API key."), LLMErrorKind.AUTH),
        (Exception("Error code: 401"), LLMErrorKind.AUTH),
        (Exception("server exploded"), LLMErrorKind.API_ERROR),
        (Exception("used 1401 tokens"), LLMErrorKind.API_ERROR),
    ],
)
def test_classify_exception(exc, kind):
    assert classify_exception(exc) == kind


def test_only_auth_errors_are_not_retryable():
    assert LLMErrorKind.AUTH not in RETRYABLE_KINDS
    assert {LLMErrorKind.TIMEOUT, LLMErrorKind.RATE_LIMIT, LLMErrorKind.API_ERROR,
            LLMErrorKind.EMPTY_RESPONSE, LLMErrorKind.MALFORMED_OUTPUT} <= RETRYABLE_KINDS


def test_secrets_are_redacted():
    key = "AIza" + "B" * 35
    assert key not in redact_secrets(f"bad request ?key={key}")
    assert "sk-" + "a" * 30 not in redact_secrets("token sk-" + "a" * 30)
    assert "abcdefghijk123" not in redact_secrets("Authorization: Bearer abcdefghijk123")
    assert "supersecretvalue1" not in redact_secrets("api_key: supersecretvalue1")
    assert key not in safe_error_message(RuntimeError(f"failed {key}"))


def test_redaction_shortens_and_cleans_whitespace():
    assert len(redact_secrets("x" * 1000)) <= 301
    assert redact_secrets("a   b\n c") == "a b c"
    assert safe_error_message(RuntimeError("boom")) == "RuntimeError: boom"


def test_retry_delay_doubles():
    assert [retry_delay(n, 0.5) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]


def test_call_with_timeout_returns_value_and_passes_errors():
    assert call_with_timeout(lambda: 42, 1) == 42
    with pytest.raises(ValueError):
        call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("inner")), 1)


def test_call_with_timeout_raises_timeout_error_quickly():
    started = time.perf_counter()
    with pytest.raises(LLMCallError) as info:
        call_with_timeout(lambda: time.sleep(0.5), 0.05)
    assert info.value.kind == LLMErrorKind.TIMEOUT
    assert time.perf_counter() - started < 0.4


def test_llm_call_error_is_a_summarization_error():
    err = LLMCallError(LLMErrorKind.TIMEOUT, "m", attempts=3)
    assert isinstance(err, SummarizationError) and err.attempts == 3


def test_extract_text_variants():
    assert extract_text("hi") == "hi"
    assert extract_text(FakeMessage("yo")) == "yo"
    assert extract_text(FakeMessage([{"type": "text", "text": "a"}, "b", {"type": "image"}, {"text": "c"}])) == "abc"
    assert extract_text(FakeMessage(None)) == "" and extract_text(FakeMessage([])) == ""


def test_dedupe_preserve_order():
    assert dedupe_preserve_order(["A  b", "a b", "c", "C", "d"]) == ["A  b", "c", "d"]
    assert dedupe_preserve_order([]) == [] and dedupe_preserve_order(["  ", ""]) == []


def test_format_time_range():
    assert format_time_range("10:00", "10:05") == "10:00 \u2013 10:05"
    assert format_time_range("10:00", "10:00") == "10:00"
    assert format_time_range(None, "10:05") == "10:05"
    assert format_time_range(None, None) == "not available"


def test_format_part_ranges():
    assert format_part_ranges([1, 2, 3, 5]) == "1\u20133, 5"
    assert format_part_ranges([4]) == "4" and format_part_ranges([]) == "none"
    assert format_part_ranges([3, 1, 2, 2]) == "1\u20133"


def test_render_node_single_and_merged():
    text = render_node(node(2, "hello", ("Bob", "Eve"), "10:00", "10:05"))
    assert text.startswith("Part 3 (time: 10:00 \u2013 10:05; speakers: Bob, Eve)")
    assert "Summary: hello" in text and "- k" in text and "Unresolved" not in text
    assert "not labeled" in render_node(node(0, speakers=()))
    merged = merge_node_metadata([node(0), node(1), node(2)], "m", ["p"], ["u"])
    assert render_node(merged).startswith("Parts 1\u20133") and "Unresolved or disputed:" in render_node(merged)


def test_merge_node_metadata_combines_in_order():
    merged = merge_node_metadata(
        [node(0, speakers=("A",), start="10:00", end="10:01"),
         node(1, speakers=("B", "A"), end="10:09"),
         node(2, speakers=(), start="10:10")],
        "merged", ["p"], [],
    )
    assert merged.speakers == ["A", "B"]
    assert (merged.start_timestamp, merged.end_timestamp) == ("10:00", "10:09")
    assert (merged.first_chunk_index, merged.last_chunk_index) == (0, 2)
    assert merged.chunk_ids == ["chunk-0000", "chunk-0001", "chunk-0002"]


@pytest.mark.parametrize("budget", [500, 1000, 3000, 100000])
def test_batching_keeps_order_loses_nothing_and_respects_budget(budget):
    nodes = [node(i) for i in range(50)]
    batches = batch_nodes(nodes, budget)
    assert [n for b in batches for n in b] == nodes
    assert all(len(b) == 1 or total_size(b) <= budget for b in batches)


def test_batching_edge_cases():
    nodes = [node(i) for i in range(5)]
    assert len(batch_nodes(nodes, 100000)) == 1 and batch_nodes([], 100) == []
    big = node(1, "z" * 5000)
    groups = batch_nodes([node(0), big, node(2)], 1000)
    assert len(groups) == 3 and groups[1] == [big]
    assert total_size([]) == 0
    assert total_size(nodes[:2]) == node_size(nodes[0]) + node_size(nodes[1]) + 2
