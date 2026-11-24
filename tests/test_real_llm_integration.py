"""REAL API test (opt-in). Skipped unless you explicitly enable it.

It makes real requests with YOUR configured key and may use free-tier quota:
    RUN_REAL_LLM_TESTS=1 python -m pytest tests/test_real_llm_integration.py -v
Everything else in the suite uses a mocked LLM and needs no key.
"""
import os
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_text_splitters")

from schemas.summary_models import GlobalSummaryStatus  # noqa: E402
from services.summarization_service import summarize_file  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_LLM_TESTS") != "1",
    reason="Real API tests are opt-in: set RUN_REAL_LLM_TESTS=1 (uses your API key and quota).",
)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_transcripts" / "chunking_short_meeting.txt"


def test_real_provider_summarizes_the_short_sample():
    result = summarize_file(SAMPLE)  # uses the provider configured in .env
    assert result.status == GlobalSummaryStatus.COMPLETE, result.warnings
    assert result.global_summary.executive_summary.strip()
    assert result.global_summary.speakers == ["Alice", "Bob", "Charlie"]
