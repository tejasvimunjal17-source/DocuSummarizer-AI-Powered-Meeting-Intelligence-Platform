"""Pure-Python helpers for the summarization service (standard library only).

Kept free of LangChain/Pydantic imports so the logic is simple to test:
error classification, secret redaction, timeouts, response-text extraction,
and the "summary node" helpers used to combine many chunk summaries.
"""
from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence


# ------------------------------------------------------------------ errors
class SummarizationError(Exception):
    """Base class for summarization problems."""


class LLMErrorKind(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    EMPTY_RESPONSE = "empty_response"
    MALFORMED_OUTPUT = "malformed_output"
    API_ERROR = "api_error"


# Everything except an authentication problem is worth retrying.
RETRYABLE_KINDS = frozenset(k for k in LLMErrorKind if k != LLMErrorKind.AUTH)


class LLMCallError(SummarizationError):
    """A single LLM call failed. `kind` says why; the message is already redacted."""

    def __init__(self, kind: LLMErrorKind, message: str, attempts: int = 0) -> None:
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts


# --------------------------------------------------------- secret redaction
_SECRET_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]{8,}"), r"\1[REDACTED]"),
    (
        re.compile(r"(?i)((?:api[_-]?key|key|token|secret)\s*[=:]\s*['\"]?)[A-Za-z0-9_\-\.]{8,}"),
        r"\1[REDACTED]",
    ),
]


def redact_secrets(text: str, max_length: int = 300) -> str:
    """Hide anything that looks like an API key and shorten long messages."""
    cleaned = str(text)
    for pattern, replacement in _SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip() + "\u2026"
    return cleaned


def safe_error_message(exc: BaseException) -> str:
    """'ExceptionName: message' with secrets removed."""
    return f"{type(exc).__name__}: {redact_secrets(str(exc))}"


_TIMEOUT_RE = re.compile(r"timed? ?out|deadline exceeded")
_RATE_RE = re.compile(r"\b429\b|rate.?limit|quota|resource.?exhausted|too many requests")
_AUTH_RE = re.compile(
    r"\b40[13]\b|invalid api key|api key not valid|api_key_invalid|incorrect api key"
    r"|unauthenticated|permission denied|authentication"
)


def classify_exception(exc: BaseException) -> LLMErrorKind:
    """Guess why an LLM call failed (works across providers by name/message)."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in name or _TIMEOUT_RE.search(message):
        return LLMErrorKind.TIMEOUT
    if "ratelimit" in name or "resourceexhausted" in name or _RATE_RE.search(message):
        return LLMErrorKind.RATE_LIMIT
    if (
        "authentication" in name
        or "permissiondenied" in name
        or "unauthenticated" in name
        or _AUTH_RE.search(message)
    ):
        return LLMErrorKind.AUTH
    return LLMErrorKind.API_ERROR


def retry_delay(attempt: int, base_seconds: float) -> float:
    """Exponential backoff: base, 2*base, 4*base ... (attempt starts at 1)."""
    return base_seconds * (2 ** (attempt - 1))


def call_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run fn in a worker thread and stop WAITING after timeout_seconds.

    Limitation: Python cannot kill a thread, so a hung request may keep running
    in the background; the caller simply moves on (and may retry).
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise LLMCallError(
            LLMErrorKind.TIMEOUT,
            f"The model did not answer within {timeout_seconds:g} seconds.",
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------ text helpers
def extract_text(response: Any) -> str:
    """Get plain text from a string, a LangChain message, or content blocks."""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    """Remove repeated items (ignoring case and extra spaces); keep first, keep order."""
    seen = set()
    result: List[str] = []
    for item in items:
        key = " ".join(item.split()).casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def format_part_ranges(numbers: Sequence[int]) -> str:
    """[1, 2, 3, 5] -> '1\u20133, 5' (used to say which parts a summary covers)."""
    ordered = sorted(set(numbers))
    if not ordered:
        return "none"
    groups: List[str] = []
    start = previous = ordered[0]
    for number in ordered[1:]:
        if number == previous + 1:
            previous = number
            continue
        groups.append(str(start) if start == previous else f"{start}\u2013{previous}")
        start = previous = number
    groups.append(str(start) if start == previous else f"{start}\u2013{previous}")
    return ", ".join(groups)


def format_time_range(start: Optional[str], end: Optional[str]) -> str:
    if start and end and start != end:
        return f"{start} \u2013 {end}"
    if start or end:
        return start or end or ""
    return "not available"


# -------------------------------------------------------- summary "nodes"
@dataclass
class SummaryNode:
    """A summary of one chunk, or of several consecutive chunks after merging."""

    summary: str
    key_points: List[str] = field(default_factory=list)
    unresolved_items: List[str] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    first_chunk_index: int = 0
    last_chunk_index: int = 0
    chunk_ids: List[str] = field(default_factory=list)


def render_node(node: SummaryNode) -> str:
    """Text shown to the LLM for one node (part numbers are chunk_index + 1)."""
    if node.first_chunk_index == node.last_chunk_index:
        header = f"Part {node.first_chunk_index + 1}"
    else:
        header = f"Parts {node.first_chunk_index + 1}\u2013{node.last_chunk_index + 1}"
    speakers = ", ".join(node.speakers) if node.speakers else "not labeled"
    lines = [
        f"{header} (time: {format_time_range(node.start_timestamp, node.end_timestamp)}; "
        f"speakers: {speakers})",
        f"Summary: {node.summary}",
    ]
    if node.key_points:
        lines.append("Key points:")
        lines.extend(f"- {p}" for p in node.key_points)
    if node.unresolved_items:
        lines.append("Unresolved or disputed:")
        lines.extend(f"- {p}" for p in node.unresolved_items)
    return "\n".join(lines)


def node_size(node: SummaryNode) -> int:
    return len(render_node(node))


_NODE_SEPARATOR = "\n\n"


def batch_nodes(nodes: Sequence[SummaryNode], budget_chars: int) -> List[List[SummaryNode]]:
    """Group CONSECUTIVE nodes so each group's rendered size stays within budget.

    Order is never changed. A node larger than the budget gets its own group.
    """
    batches: List[List[SummaryNode]] = []
    current: List[SummaryNode] = []
    current_size = 0
    for node in nodes:
        size = node_size(node)
        extra = size + (len(_NODE_SEPARATOR) if current else 0)
        if current and current_size + extra > budget_chars:
            batches.append(current)
            current, current_size = [], 0
            extra = size
        current.append(node)
        current_size += extra
    if current:
        batches.append(current)
    return batches


def total_size(nodes: Sequence[SummaryNode]) -> int:
    if not nodes:
        return 0
    return sum(node_size(n) for n in nodes) + len(_NODE_SEPARATOR) * (len(nodes) - 1)


def _unique(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def merge_node_metadata(
    nodes: Sequence[SummaryNode],
    summary: str,
    key_points: List[str],
    unresolved_items: List[str],
) -> SummaryNode:
    """New node covering all given nodes (speakers/time range/ids combined in order)."""
    starts = [n.start_timestamp for n in nodes if n.start_timestamp]
    ends = [n.end_timestamp for n in nodes if n.end_timestamp]
    return SummaryNode(
        summary=summary,
        key_points=key_points,
        unresolved_items=unresolved_items,
        speakers=_unique([s for n in nodes for s in n.speakers]),
        start_timestamp=starts[0] if starts else None,
        end_timestamp=ends[-1] if ends else None,
        first_chunk_index=min(n.first_chunk_index for n in nodes),
        last_chunk_index=max(n.last_chunk_index for n in nodes),
        chunk_ids=[cid for n in nodes for cid in n.chunk_ids],
    )
