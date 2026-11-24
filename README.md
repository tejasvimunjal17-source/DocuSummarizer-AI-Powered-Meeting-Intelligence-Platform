# DocuSummarizer — AI-Powered Meeting Intelligence Platform

CODE-A-NOVA AI Internship · Month 2 · Task 2

> **Status: Day 4 implemented (chunk summaries + global executive summary).** The full
> AI application is NOT finished. Day 1: config + schemas. Day 2: ingestion,
> normalization, speaker turns. Day 3: chunking. Day 4: per-chunk summarization and the
> global executive summary (LLM calls happen only here). Decisions, action items and
> risks are Day 5; the Streamlit UI is Day 7.
> **Verification status:** the automated tests were written but have not yet been run in
> an environment with all dependencies installed, and no real LLM call has been made.
> See "Verification status" in the Day 4 section.

## Project overview and objective
DocuSummarizer will read raw meeting transcripts and extract an executive
summary, key decisions (with speaker), action items (owner + deadline), risk
flags (with severity) and topic clusters, shown in a Streamlit app with JSON and
PDF export. The full plan is the official 10-day CODE-A-NOVA brief.

## Technology stack
| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM | Google Gemini (default, free tier) — OpenAI GPT-4o supported by config |
| Framework | LangChain + LangChain Community |
| Validation | Pydantic v2 |
| Config | python-dotenv |
| Testing | pytest |
| Later days | Streamlit, python-docx, PyMuPDF, ReportLab, rouge-score |

## Initial architecture
- `config/` loads and validates settings from environment variables.
- `schemas/` Pydantic models for the extraction result.
- `services/llm_service.py` creates a LangChain chat model for the chosen provider.
- Other folders are empty placeholders for Days 2–10.

## Folder structure
```
DocuSummarizer/
├── config/            settings.py (secure env config)
├── schemas/           extraction_models.py (Day 1), summary_models.py (Day 4)
├── services/          llm_service.py (LLM factory), summarization_service.py + summarization_utils.py (Day 4)
├── prompts/           summarization_prompts.py (Day 4; extraction prompts come Day 5)
├── ingestion/         parsers, normalizer, speaker detector, service (Day 2)
├── chunking/          speaker-aware chunking engine (Day 3)
├── chains/            summarization_chains.py (Day 4)
├── evaluation/        (Day 9)
├── tests/             Day 1 + Day 2 test files
├── sample_transcripts/  chunking samples (Day 3); evaluation samples come Day 9-10
├── reports/           (Day 8-9)
├── requirements.txt  .env.example  .gitignore  pytest.ini  README.md
```
`app.py` (Streamlit) is intentionally not created yet; it belongs to Day 7.

## Install
Requires Python 3.11 or newer.
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
(On Android, Termux or a cloud environment such as GitHub Codespaces works.)

## Environment and API key setup
```bash
cp .env.example .env
```
Open `.env` and replace `your_api_key_here` with your real Gemini key
(free key: Google AI Studio). Settings used:

| Variable | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `gemini` or `openai` | `gemini` |
| `GEMINI_API_KEY` | your Gemini key | required for gemini |
| `OPENAI_API_KEY` | your OpenAI key | required for openai |
| `LLM_MODEL` | model name | `gemini-2.5-flash` / `gpt-4o` |
| `LLM_TEMPERATURE` | 0.0 – 2.0 | `0.0` |

The default Gemini model name is unverified; if the API rejects it, set `LLM_MODEL`.

## API key security
- Keys are read only from environment variables / `.env`.
- `.env` is in `.gitignore`. Never commit it, never share it in chat or screenshots.
- Settings objects hide the key when printed or logged.
- If a key leaks, delete it in Google AI Studio and create a new one.

## Run the tests
```bash
python -m pytest -v
```
These tests use no network and no API key.

## Current implementation status
See also the Day 2 section below. Implemented in Day 1: folder structure, requirements, `.env.example`, `.gitignore`,
secure settings loader, LLM factory (`build_llm`), Pydantic schemas, unit tests.

Not yet verified: a real Gemini API call (never performed), `pip install` of
`requirements.txt` in a clean environment.

## Features not yet implemented
Days 3 and 4 are implemented (see below). Still to do: structured
extraction (5), token tracking and retries (6), Streamlit UI (7), JSON/PDF export
(8), ROUGE evaluation (9), optimization, sample data and demo (10).

## Troubleshooting
If pip reports a version conflict between LangChain packages, install them
together without pins and then freeze the working versions:
`pip install langchain langchain-community langchain-google-genai pydantic python-dotenv pytest`

---

# Day 2: Document ingestion

## Supported file formats
| Format | Library | Notes |
|---|---|---|
| `.txt` | Python built-ins | UTF-8, UTF-8 BOM, UTF-16/32 (with BOM), Windows-1252 fallback (with warning) |
| `.docx` | python-docx | Body paragraphs and table rows, in document order |
| `.pdf` | PyMuPDF | Text layer only, page by page |

Anything else (for example `.doc`, `.csv`) is rejected with `UnsupportedFileFormatError`.

## Architecture
```
file / bytes
   -> ingestion_service.detect_file_format()      (by extension)
   -> TxtParser | DocxParser | PdfParser          (raw text, warnings)
   -> text_normalizer.normalize_text()            (clean text)
   -> speaker_detector.detect_speaker_turns()     (list of SpeakerTurn)
   -> IngestionResult (Pydantic)
```
Files in `ingestion/`: `errors.py`, `models.py`, `base_parser.py`, `txt_parser.py`,
`docx_parser.py`, `pdf_parser.py`, `text_normalizer.py`, `speaker_detector.py`,
`ingestion_service.py`.

## How to use
```python
from ingestion import ingest_file, ingest_bytes

result = ingest_file("sample_transcripts/meeting.txt")
print(result.speakers, result.speaker_count, result.warnings)
for turn in result.speaker_turns:
    print(turn.index, turn.speaker, turn.timestamp, turn.text)
```
`IngestionResult` holds: `filename`, `file_format`, `raw_text` (exactly as extracted),
`normalized_text`, `speaker_turns`, `speakers`, `speaker_count`, `has_speaker_labels`,
`page_count` (PDF), `word_count`, `status`, `warnings`. Failures raise a subclass of
`IngestionError` (`UnsupportedFileFormatError`, `InvalidDocumentError`,
`EmptyDocumentError`, `OCRRequiredError`).

## Text normalization
Applies Unicode NFC, unifies line endings, turns non-breaking spaces into spaces,
removes zero-width characters and control characters, collapses repeated spaces and
extra blank lines, and trims. It never removes punctuation, speaker labels,
timestamps, or merges lines/speakers. The original text is kept in `raw_text`.

## Speaker-turn detection
Recognized: `Rahul: text`, `[10:30] Rahul: text`, `(10:30:15)`, `10:30 Rahul:`,
`[10:30 AM] Rahul:`, `Speaker 1:`, `SPEAKER_01:`, `Participant A:`, `Dr. Rahul Sharma:`,
and non-Latin names. Multi-line statements are supported (lines directly after a label
belong to that speaker). Text with no label gets `speaker = None` (unknown); names are
never invented. Header words such as `Note:`, `Agenda:`, `Date:` are not speakers.
The complete rule list and limitations are in the docstring of
`ingestion/speaker_detector.py`.

## Install and run the ingestion tests
```bash
pip install -r requirements.txt
python -m pytest tests -v
```
Run only Day 2 tests:
```bash
python -m pytest tests/test_txt_parser.py tests/test_docx_parser.py tests/test_pdf_parser.py tests/test_pdf_parser_mocked.py tests/test_text_normalizer.py tests/test_speaker_detector.py tests/test_ingestion_service.py -v
```
`test_pdf_parser.py` uses real PyMuPDF and is skipped automatically if PyMuPDF is not
installed. `test_pdf_parser_mocked.py` uses a fake PyMuPDF to test our own logic only.

## Known limitations (Day 2)
- **Scanned PDFs (no text layer) need OCR, which is not implemented.** They raise
  `OCRRequiredError`. OCR is not part of the mandatory scope.
- DOCX: headers, footers, footnotes, text boxes, comments and images are not extracted.
  Tables are read row by row (cells joined by tabs), so a Speaker|Text table is not
  turned into speaker turns.
- PDF: page headers, footers and page numbers appear as normal text. Password-protected
  PDFs are rejected. Reading order for multi-column layouts depends on PyMuPDF.
- Speaker detection is rule-based (see limitations in `speaker_detector.py`): lowercase
  names, labels with brackets like `Rahul (PM):`, and timestamps after the name are not
  recognized. A blank line ends a speaker's turn.
- File type is decided by the extension, not by file content.
- Chunking is not implemented yet (Day 3).

---

# Day 3: Speaker-aware chunking

## Why chunking is necessary
LLMs can only read a limited amount of text at once, and a 50,000-word transcript is far
too long for one prompt. Chunking cuts the transcript into smaller pieces that can be
summarized one by one (Day 4). Cutting carelessly would split a person's sentence or
lose who said it, so our chunker works on **speaker turns** from the Day 2 ingestion layer.

## How it works
```
IngestionResult.speaker_turns
  -> units: each turn is one unit; a turn that is too long for one chunk is divided
            by LangChain's RecursiveCharacterTextSplitter into fragments
  -> packing: units are placed into chunks in order; a chunk is closed BEFORE a unit
            that would exceed chunk_size, so boundaries fall between turns
  -> overlap: the next chunk starts with a copy of the last whole turn(s)/fragments of
            the previous chunk, up to chunk_overlap characters
  -> TextChunk objects with metadata
```
**Where LangChain is used:** `RecursiveCharacterTextSplitter` (package
`langchain-text-splitters`) divides every turn that does not fit into one chunk. Its
separators are tried in this order: blank line, line break, sentence end (`. ! ? ...`
and CJK/Hindi marks), clause end (`, ; :`), space. There is no empty-string separator,
so a word is never cut in half.
**What is our own code:** the whole-turn packing and the turn-level overlap
(`chunking/packing.py`), because the LangChain splitter does not know about speakers.

## Speaker-aware boundaries
- Chunks break between turns whenever possible; a turn is not cut if it fits.
- A turn that is too long is cut at sentence boundaries where possible.
- Every fragment of a divided turn repeats the speaker name and timestamp
  (e.g. `[10:05] Alice: ...`) and is marked as a continuation
  (`segment.fragment_index > 0`, chunk `is_continuation=True`).
- Unlabeled text keeps `speaker=None`; no names are invented.
- **Not every turn stays intact:** a long turn is divided (reported as `split_turn_count`).
- A single sentence longer than the limit is cut at a clause or word boundary; the
  fragment gets `ends_mid_sentence=True` and a warning is recorded.
- A single "word" longer than the limit (for example a very long URL) cannot be cut. Its
  chunk becomes larger than `chunk_size`. It is reported (warning + validation issue)
  and is never truncated.

## Chunk size and overlap (CHARACTERS, not tokens)
`chunk_size` (default 1500) and `chunk_overlap` (default 200) count **characters** of the
rendered chunk text, labels included. They are not tokens (token counting comes on Day 6).
Rules: `chunk_size >= 50`, `chunk_overlap < chunk_size`, and `chunk_overlap <= chunk_size/2`.
Configure in code `ChunkingConfig(chunk_size=800, chunk_overlap=100)` or with the optional
environment variables `CHUNK_SIZE` and `CHUNK_OVERLAP` (`ChunkingConfig.from_env()`).
Overlap copies whole turns/fragments only, never half a sentence, so it can be smaller
than `chunk_overlap` (or zero) when the last turn is longer than the overlap setting.

## Chunk metadata
Each `TextChunk` has: `chunk_id`, `chunk_index`, `text`, `character_count`, `word_count`
(whitespace-separated words), `speakers`, `speaker_turn_ids`, `start_timestamp`,
`end_timestamp`, `is_continuation`, `overlap_character_count`, `source_filename` and
`segments` (the structured statements; overlap copies are marked `is_overlap=True`).
Chunk text format: `[timestamp] Speaker: statement`, one statement per line.

## Usage example
```python
from chunking import (ChunkingConfig, chunk_ingestion_result,
                      format_chunk_preview, validate_chunks)
from ingestion import ingest_file

ingestion = ingest_file("sample_transcripts/chunking_realistic_meeting.txt")
config = ChunkingConfig(chunk_size=1500, chunk_overlap=200)
result = chunk_ingestion_result(ingestion, config)

print(format_chunk_preview(result, max_chunks=3, preview_chars=100))
report = validate_chunks(result.chunks, result.config, ingestion.speaker_turns)
print(report.is_valid, report.codes())
```
Other entry points: `chunk_file(path)`, `chunk_text(text)`, `chunk_turns(turns)`.

## Chunk preview
`format_chunk_preview(result, max_chunks=5, preview_chars=120)` returns text with the
total chunk count and, for the first chunks, index, character count, word count,
speakers, continuation flag, overlap size and a shortened text. `print_chunk_preview`
prints it; `build_chunk_previews` returns structured `ChunkPreview` objects.

## Size and content validation
`validate_chunks(chunks, config, source_turns=None)` returns a `ChunkValidationResult`
(`is_valid`, `issues`, `errors`, `warnings`, `codes()`). It checks: empty chunks, sequential
indexes, character and word counts, speaker/turn/timestamp metadata, chunk text matching its
segments, size limit, overlap rules and overlap content, chronological order and (when
`source_turns` is given) that no turn text was lost, duplicated or re-attributed.
An oversized chunk that consists of one undividable piece is a **warning**
(`oversized_unsplittable`), any other oversized chunk is an **error**.

## Run the chunking tests
```bash
pip install -r requirements.txt
python -m pytest tests -v                  # whole suite (Day 1 + 2 + 3)
python -m pytest tests/test_chunk_models.py tests/test_speaker_aware_splitter.py \
  tests/test_chunking_service.py tests/test_chunk_validator.py tests/test_chunk_preview.py -v
```
Modules that need LangChain skip themselves automatically if `langchain-text-splitters`
is not installed.

## Sample data for chunking
`sample_transcripts/`: `chunking_short_meeting.txt`, `chunking_long_turns.txt`,
`chunking_mixed_unlabeled.txt`, `chunking_multilingual.txt` and
`chunking_realistic_meeting.txt` (about 4,500 words, 97 turns, 5 speakers).

## Change to Day 2 code (additive)
`ingestion/speaker_detector.py` now also recognizes a timestamp in brackets or
parentheses **after** the name: `Alice [10:00 AM]: ...`, `Rahul (10:30): ...`. All
previously supported formats are unchanged. New tests were added; none were removed or edited.

## Known limitations (Day 3)
- Sizes are characters, not tokens.
- Overlap is turn-level (whole turns/fragments), not an exact character count.
- The sentence separator also triggers after abbreviations such as "Dr." when a long
  turn has to be divided.
- Word counts use whitespace splitting; Chinese/Japanese text without spaces counts as few words.
- Quality depends on Day 2 speaker detection (see its limitations). Text after a blank line
  with no label is a separate unknown-speaker turn.
- Chunks are not sent to any LLM yet (Day 4).

---

# Day 4: Per-chunk summaries and the global executive summary

## Architecture
```
Day 2 ingestion -> Day 3 chunks (TextChunk) -> Day 4 map: one summary per chunk
   -> (only if the summaries are too big) hierarchical merge, level by level
   -> one final request: global executive summary  -> MeetingSummaryResult
```
| File | Role |
|---|---|
| `prompts/summarization_prompts.py` | All prompt text (chunk, merge, global) as LangChain `ChatPromptTemplate`s, documented |
| `chains/summarization_chains.py` | LCEL chains `prompt \| chat model \| text`, plus `PydanticOutputParser`s |
| `schemas/summary_models.py` | Pydantic models: LLM outputs, `ChunkSummary`, `GlobalExecutiveSummary`, `MeetingSummaryResult`, `SummarizationConfig` |
| `services/summarization_service.py` | `SummarizationService` and the convenience functions |
| `services/summarization_utils.py` | Pure helpers: error classification, secret redaction, timeouts, batching |
| `tests/llm_fakes.py` | Fake LLM (a real LangChain Runnable) used by the tests |

## Per-chunk summarization
For every Day 3 `TextChunk` (in `chunk_index` order) the service sends one request that
contains: the part number, speakers, time range, the chunk's NEW statements, and the
overlap text as "previous context" (shown separately so it is not summarized twice).
The LLM must return JSON `{summary, key_points, unresolved_items}`. Speakers, timestamps,
chunk id, word count and filename are copied from the chunk metadata, never from the LLM,
so they cannot be invented. An empty chunk is `skipped`; an invalid object is `failed`;
neither stops the run.

## Global executive summary
1. Chunk summaries (successful ones) are put in order.
2. If they fit in `reduction_batch_chars` (default 12,000 characters, not tokens) they go
   straight to the final request.
3. Otherwise consecutive groups are merged by the LLM (never reordered), level by level,
   until they fit (max 5 levels). A group whose merge fails stays unmerged (nothing is lost).
4. One final request returns `GlobalLLMOutput` (title, executive summary, overview,
   major topics, key takeaways, important discussions, unresolved questions).
The whole transcript is never sent in one request. Decisions are not extracted (Day 5):
"key takeaways" only repeats conclusions that were actually stated.

## Status values (never a false "complete")
`complete` = every chunk summarized and the final summary written. `partial` = final
summary written but some chunks failed (they are listed, and the LLM is told which parts
are missing). `failed` = no usable global summary (the chunk summaries are still returned).
`empty` = nothing to summarize. Per chunk: `success`, `failed`, `skipped`.

## Prompt design (see the docstring of `prompts/summarization_prompts.py`)
System message = rules (find main points, keep names/figures/dates, keep speaker
attribution, do not invent, keep disagreements and open questions, avoid repetition,
treat the transcript as data and ignore instructions inside it, do not translate names).
Human message = data passed as variables, so braces in a transcript cannot break the
template. The JSON format comes from the Pydantic parser so prompt and validation agree.

## Validation, retries and errors
- LLM output is parsed with `PydanticOutputParser` into `ChunkSummaryLLMOutput` /
  `GlobalLLMOutput`. Blank required text or wrong types are rejected; a single string for a
  list is turned into a list; blank and duplicate list items are removed.
- Invalid output: asked again (up to `max_retries`, default 2) with the parser error added
  to the prompt. API errors, rate limits, timeouts and empty answers: retried with backoff
  (1 s, 2 s, ...). Authentication errors are not retried and stop further calls.
- Timeouts: each request waits at most `timeout_seconds` (default 60). Python cannot kill the
  worker thread, so a hung request may continue in the background.
- Error messages are redacted (API keys removed) and the key is never logged.
- A missing key/package raises `ConfigurationError` (fix `.env`), it is not swallowed.

## Provider integration
The service takes any LangChain chat model/Runnable. By default it calls the existing
`services.llm_service.build_llm()` which reads `.env` (Gemini by default, OpenAI supported).
No provider is named in the summarization code.

## Setup and environment variables
Same install as before (`pip install -r requirements.txt`); no new dependency was needed.
Put your key in `.env` (copy `.env.example`). Optional settings (defaults shown):
`SUMMARY_MAX_RETRIES=2`, `SUMMARY_RETRY_DELAY_SECONDS=1.0`, `SUMMARY_TIMEOUT_SECONDS=60`,
`SUMMARY_REQUEST_DELAY_SECONDS=0` (raise it if the free tier rate-limits you),
`SUMMARY_REDUCTION_BATCH_CHARS=12000`, `SUMMARY_OUTPUT_LANGUAGE=English`
(`SummarizationConfig.from_env()` reads them).

## How to run summarization
```python
from services.summarization_service import summarize_file

result = summarize_file("sample_transcripts/chunking_short_meeting.txt")  # real LLM call!
print(result.status.value)                       # complete / partial / failed / empty
print(result.global_summary.meeting_title)
print(result.global_summary.executive_summary)
for item in result.chunk_summaries:
    print(item.chunk_id, item.status.value, item.summary)
print(result.llm_call_count, result.retry_count, result.warnings)
```
Other entry points: `summarize_text(text)`, `summarize_chunking_result(chunking_result)`,
or `SummarizationService(llm, config)` for full control. Pass `llm=` to use your own model.
`result.global_summary.to_executive_summary()` converts to the Day 1 `ExecutiveSummary`.

## Expected-summary examples
`sample_transcripts/expected_summaries/*.json` (short, multi-speaker long turns, missing
speakers, multilingual, long realistic meeting). **They are hand-written examples, not
LLM output**; each file says so. Tests check they are valid and match the real chunk
counts, speakers and timestamps of the sample transcripts.

## Run the tests
```bash
python -m pytest tests -v                       # everything (mocked LLM, no key needed)
python -m pytest tests/test_summary_models.py tests/test_summarization_utils.py \
  tests/test_summarization_prompts.py tests/test_chunk_summarization.py \
  tests/test_global_summary.py tests/test_summarization_service.py -v
RUN_REAL_LLM_TESTS=1 python -m pytest tests/test_real_llm_integration.py -v   # optional REAL API call
```

## Verification status (honest)
Written: Day 4 tests with a mocked LLM. The standard-library helper module was executed
with plain Python checks. Not yet verified by the author: pytest runs, the real LangChain
chain/parser behaviour, any real Gemini/OpenAI call, and the full 50,000-word pipeline.

## Known limitations (Day 4)
- Sizes are characters, not tokens (token tracking is Day 6).
- Requests are sequential (kind to free-tier limits, but slow for very long meetings).
- Output quality depends on the model; JSON compliance is enforced by validation and retries.
- The timeout stops waiting but cannot cancel a running request.
- If merging fails repeatedly the final request can exceed the size limit (a warning is added).
- Summaries cover content only; decisions/actions/risks (Day 5) are not extracted yet.
