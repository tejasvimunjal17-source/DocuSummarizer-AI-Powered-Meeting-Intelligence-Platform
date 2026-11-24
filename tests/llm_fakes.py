"""Fake LLM for Day 4 tests (MOCKED responses; no network, no API key).

The fake is a real LangChain Runnable (RunnableLambda), so the real prompt
templates, chains and output parsers run exactly as they would with Gemini/OpenAI.
It looks at the prompt text to see which stage is calling (chunk / reduce / global)
and answers with JSON, or raises an error on demand.
"""
import json
import re
from typing import Callable, Dict, List

from langchain_core.runnables import RunnableLambda

from prompts.summarization_prompts import CHUNK_MARKER, GLOBAL_MARKER, REDUCE_MARKER

GLOBAL_OK = json.dumps(
    {
        "meeting_title": "Test meeting",
        "executive_summary": "The meeting covered several topics.",
        "meeting_overview": "A test meeting.",
        "major_topics": ["Topic A", "Topic B"],
        "key_takeaways": ["Takeaway 1"],
        "important_discussions": ["Discussion 1"],
        "unresolved_questions": ["Question 1"],
    }
)


def chunk_json(part: int) -> str:
    return json.dumps(
        {
            "summary": f"Summary of part {part}.",
            "key_points": [f"Point from part {part}"],
            "unresolved_items": [],
        }
    )


def stage_of(prompt_text: str) -> str:
    if GLOBAL_MARKER in prompt_text:
        return "global"
    if REDUCE_MARKER in prompt_text:
        return "reduce"
    if CHUNK_MARKER in prompt_text:
        return "chunk"
    return "unknown"


def parts_in(prompt_text: str) -> List[int]:
    """Part numbers at the start of each summary block, in the order shown."""
    return [int(n) for n in re.findall(r"^Parts? (\d+)", prompt_text, re.MULTILINE)]


def default_chunk(text: str, call_number: int) -> str:
    match = re.search(r"Meeting part (\d+) of", text)
    return chunk_json(int(match.group(1)) if match else call_number)


def default_reduce(text: str, call_number: int) -> str:
    numbers = []
    for first, last in re.findall(r"^Parts? (\d+)(?:\u2013(\d+))?", text, re.MULTILINE):
        numbers.append(int(first))
        if last:
            numbers.append(int(last))
    low, high = min(numbers), max(numbers)
    return json.dumps(
        {
            "summary": f"Merged parts {low}-{high}.",
            "key_points": [f"Merged point {low}-{high}"],
            "unresolved_items": [],
        }
    )


def default_global(text: str, call_number: int) -> str:
    return GLOBAL_OK


def sequence(*items):
    """Handler that returns/raises item 1 on call 1, item 2 on call 2 ... (last repeats).

    An item can be a string (returned), an exception instance (raised) or a
    function (text, call_number) -> string.
    """

    def handler(text: str, call_number: int):
        item = items[min(call_number, len(items)) - 1]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(text, call_number)
        return item

    return handler


class FakeLLM:
    def __init__(self, chunk: Callable = None, reduce: Callable = None, glob: Callable = None):
        self.handlers: Dict[str, Callable] = {
            "chunk": chunk or default_chunk,
            "reduce": reduce or default_reduce,
            "global": glob or default_global,
        }
        self.prompts: List[str] = []
        self.by_stage: Dict[str, List[str]] = {"chunk": [], "reduce": [], "global": [], "unknown": []}

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def _respond(self, prompt_value):
        text = prompt_value.to_string()
        stage = stage_of(text)
        self.prompts.append(text)
        self.by_stage[stage].append(text)
        handler = self.handlers.get(stage)
        if handler is None:
            raise AssertionError("Prompt of unknown stage received by FakeLLM")
        return handler(text, len(self.by_stage[stage]))

    def runnable(self):
        def fn(prompt_value):
            return self._respond(prompt_value)

        return RunnableLambda(fn)
