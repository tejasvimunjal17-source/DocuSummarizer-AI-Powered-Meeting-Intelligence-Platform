"""LangChain (LCEL) chains for summarization.

Each chain is:   prompt | chat model | text extraction
so it returns plain text. Parsing into Pydantic models happens in the service,
which lets it retry with a correction message when the JSON is invalid.

The chat model is any LangChain chat model / Runnable (Gemini, OpenAI, or a fake
model in tests). Nothing here names a provider; services/llm_service.py chooses it.
"""
from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from prompts.summarization_prompts import (
    build_chunk_prompt,
    build_global_prompt,
    build_reduce_prompt,
)
from schemas.summary_models import ChunkSummaryLLMOutput, GlobalLLMOutput
from services.summarization_utils import extract_text


def get_chunk_parser() -> PydanticOutputParser:
    """Parser for chunk summaries (also used for merged summaries)."""
    return PydanticOutputParser(pydantic_object=ChunkSummaryLLMOutput)


def get_global_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=GlobalLLMOutput)


def _text_step() -> Runnable:
    return RunnableLambda(extract_text)


def build_chunk_chain(llm: Any) -> Runnable:
    return build_chunk_prompt() | llm | _text_step()


def build_reduce_chain(llm: Any) -> Runnable:
    return build_reduce_prompt() | llm | _text_step()


def build_global_chain(llm: Any) -> Runnable:
    return build_global_prompt() | llm | _text_step()
