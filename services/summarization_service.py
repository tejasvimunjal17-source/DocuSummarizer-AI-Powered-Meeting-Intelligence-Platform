"""Summarization service (Day 4): per-chunk summaries + global executive summary.

    Day 3 TextChunk objects
      -> summarize each chunk with the LLM             (map)
      -> if the summaries are too large for one request, merge consecutive
         groups of summaries, level by level            (hierarchical reduce)
      -> one LLM call writes the global executive summary

The LLM is any LangChain chat model/Runnable. By default the project's existing
provider factory (services.llm_service.build_llm) is used, so this file never
mentions Gemini or OpenAI.

Reliability rules
* Expected problems never crash a run: a bad chunk becomes a FAILED/SKIPPED
  ChunkSummary and the run continues.
* Failed LLM calls are retried with backoff; unparsable output is retried with a
  correction message. Authentication errors are not retried and stop further calls.
* The global summary is COMPLETE only if every chunk was summarized. If some chunks
  failed it is PARTIAL (and says which). If nothing could be summarized it is FAILED.
* Nothing is truncated: if summaries are too large they are merged, never cut.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from chains.summarization_chains import (
    build_chunk_chain,
    build_global_chain,
    build_reduce_chain,
    get_chunk_parser,
    get_global_parser,
)
from chunking import ChunkingConfig, ChunkingResult, TextChunk, chunk_file, chunk_text
from prompts.summarization_prompts import NO_CORRECTION, build_correction_note
from schemas.summary_models import (
    ChunkSummary,
    ChunkSummaryLLMOutput,
    GlobalExecutiveSummary,
    GlobalSummaryStatus,
    MeetingSummaryResult,
    SummarizationConfig,
    SummaryStatus,
)
from services.llm_service import build_llm
from services.summarization_utils import (
    RETRYABLE_KINDS,
    LLMCallError,
    LLMErrorKind,
    SummaryNode,
    batch_nodes,
    call_with_timeout,
    classify_exception,
    format_part_ranges,
    format_time_range,
    merge_node_metadata,
    render_node,
    retry_delay,
    safe_error_message,
    total_size,
)

_NODE_JOIN = "\n\n"


def _unique(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _node_from_summary(summary: ChunkSummary) -> SummaryNode:
    return SummaryNode(
        summary=summary.summary,
        key_points=list(summary.key_points),
        unresolved_items=list(summary.unresolved_items),
        speakers=list(summary.speakers),
        start_timestamp=summary.start_timestamp,
        end_timestamp=summary.end_timestamp,
        first_chunk_index=summary.chunk_index,
        last_chunk_index=summary.chunk_index,
        chunk_ids=[summary.chunk_id],
    )


class SummarizationService:
    """Runs the Day 4 pipeline with any LangChain chat model."""

    def __init__(
        self,
        llm: Any = None,
        config: Optional[SummarizationConfig] = None,
        settings: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or SummarizationConfig()
        self._llm = llm
        self._settings = settings
        self._sleep = sleep
        self._chains: Dict[str, Any] = {}
        self._chunk_parser = get_chunk_parser()
        self._global_parser = get_global_parser()
        self.llm_call_count = 0
        self.retry_count = 0
        self._fatal_error: Optional[str] = None

    # ------------------------------------------------------------- LLM access
    def _chain(self, stage: str) -> Any:
        if stage not in self._chains:
            if self._llm is None:
                # May raise ConfigurationError (missing key/package): the user must fix it.
                self._llm = build_llm(self._settings)
            builders = {
                "chunk": build_chunk_chain,
                "reduce": build_reduce_chain,
                "global": build_global_chain,
            }
            self._chains[stage] = builders[stage](self._llm)
        return self._chains[stage]

    def _invoke_chain(self, chain: Any, inputs: Dict[str, Any]) -> str:
        self.llm_call_count += 1
        try:
            return call_with_timeout(
                lambda: chain.invoke(inputs), self.config.timeout_seconds
            )
        except LLMCallError:
            raise
        except Exception as exc:  # provider errors differ; classify and redact
            raise LLMCallError(classify_exception(exc), safe_error_message(exc)) from exc

    def _run_stage(
        self, stage: str, inputs: Dict[str, Any], parser: Any, label: str
    ) -> Tuple[Any, int]:
        """Call the LLM and parse its answer, with retries. Returns (parsed, attempts)."""
        chain = self._chain(stage)
        max_attempts = self.config.max_retries + 1
        correction = NO_CORRECTION
        last_error: Optional[LLMCallError] = None
        attempt = 0
        for attempt in range(1, max_attempts + 1):
            call_inputs = dict(inputs, correction=correction)
            try:
                text = self._invoke_chain(chain, call_inputs)
                if not isinstance(text, str) or not text.strip():
                    raise LLMCallError(
                        LLMErrorKind.EMPTY_RESPONSE, "The model returned an empty response."
                    )
                try:
                    return parser.parse(text), attempt
                except Exception as exc:
                    raise LLMCallError(
                        LLMErrorKind.MALFORMED_OUTPUT,
                        f"The model's answer was not valid: {safe_error_message(exc)}",
                    ) from exc
            except LLMCallError as exc:
                last_error = exc
                if exc.kind not in RETRYABLE_KINDS or attempt == max_attempts:
                    break
                self.retry_count += 1
                if exc.kind == LLMErrorKind.MALFORMED_OUTPUT:
                    correction = build_correction_note(str(exc))
                else:
                    correction = NO_CORRECTION
                    self._sleep(retry_delay(attempt, self.config.retry_delay_seconds))
        assert last_error is not None
        raise LLMCallError(
            last_error.kind,
            f"{label} failed after {attempt} attempt(s) ({last_error.kind.value}): {last_error}",
            attempts=attempt,
        )

    # --------------------------------------------------------- chunk summaries
    def summarize_chunk(
        self,
        chunk: Any,
        total_chunks: Optional[int] = None,
        position: int = 0,
    ) -> ChunkSummary:
        """Summarize one Day 3 TextChunk. Never raises for a bad chunk or a failed call."""
        if not isinstance(chunk, TextChunk):
            return self._invalid_summary(
                position, f"Invalid chunk: expected a TextChunk, got {type(chunk).__name__}."
            )
        if not str(chunk.chunk_id).strip():
            return self._invalid_summary(position, "Invalid chunk: chunk_id is missing.", chunk)

        inputs = self._chunk_inputs(chunk, total_chunks)
        if inputs is None:
            return self._chunk_result(chunk, SummaryStatus.SKIPPED, "Chunk is empty; nothing to summarize.")
        if self._fatal_error:
            return self._chunk_result(
                chunk,
                SummaryStatus.FAILED,
                f"Not attempted because of an earlier error: {self._fatal_error}",
            )
        try:
            output, attempts = self._run_stage(
                "chunk", inputs, self._chunk_parser, f"Summarizing {chunk.chunk_id}"
            )
        except LLMCallError as exc:
            if exc.kind == LLMErrorKind.AUTH:
                self._fatal_error = (
                    "Authentication failed. Check the API key in your .env file "
                    "(the key is never shown)."
                )
            return self._chunk_result(chunk, SummaryStatus.FAILED, str(exc), attempts=exc.attempts)
        return self._chunk_result(chunk, SummaryStatus.SUCCESS, None, output, attempts)

    def summarize_chunks(
        self, chunks: Sequence[Any], warnings: Optional[List[str]] = None
    ) -> List[ChunkSummary]:
        """Summarize all chunks, in chronological order, one request per chunk."""
        warnings = warnings if warnings is not None else []
        items = list(chunks)
        self._fatal_error = None
        keys = [c.chunk_index if isinstance(c, TextChunk) else i for i, c in enumerate(items)]
        order = list(range(len(items)))
        if keys != sorted(keys):
            warnings.append(
                "Chunks were not in chronological order; they were sorted by chunk_index."
            )
            order = sorted(order, key=lambda i: keys[i])

        summaries: List[ChunkSummary] = []
        for n, i in enumerate(order):
            if n > 0 and self.config.request_delay_seconds > 0:
                self._sleep(self.config.request_delay_seconds)
            summaries.append(self.summarize_chunk(items[i], len(items), i))
        return summaries

    def _chunk_inputs(self, chunk: TextChunk, total_chunks: Optional[int]) -> Optional[Dict[str, Any]]:
        if chunk.segments:
            new = [s for s in chunk.segments if not s.is_overlap]
            overlap = [s for s in chunk.segments if s.is_overlap]
            transcript = "\n".join(s.rendered_text() for s in new)
            context = (
                "\n".join(s.rendered_text() for s in overlap)
                if self.config.include_overlap_context
                else ""
            )
        else:
            transcript, context = chunk.text, ""
        if not transcript.strip():
            return None
        return {
            "output_language": self.config.output_language,
            "position": chunk.chunk_index + 1,
            "total_parts": total_chunks if total_chunks else "unknown",
            "source_filename": chunk.source_filename or "unknown",
            "speakers": ", ".join(chunk.speakers)
            or "not labeled (speaker names are not available)",
            "time_range": format_time_range(chunk.start_timestamp, chunk.end_timestamp),
            "continuation_note": (
                "Note: this part begins in the middle of a long statement that started "
                "in an earlier part."
                if chunk.is_continuation
                else ""
            ),
            "context_text": context.strip() or "(none)",
            "transcript_text": transcript,
            "format_instructions": self._chunk_parser.get_format_instructions(),
        }

    @staticmethod
    def _chunk_result(
        chunk: TextChunk,
        status: SummaryStatus,
        error: Optional[str],
        output: Optional[ChunkSummaryLLMOutput] = None,
        attempts: int = 0,
    ) -> ChunkSummary:
        return ChunkSummary(
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            summary=output.summary if output else "",
            key_points=list(output.key_points) if output else [],
            unresolved_items=list(output.unresolved_items) if output else [],
            speakers=list(chunk.speakers),
            speaker_turn_ids=list(chunk.speaker_turn_ids),
            start_timestamp=chunk.start_timestamp,
            end_timestamp=chunk.end_timestamp,
            source_filename=chunk.source_filename,
            word_count=chunk.word_count,
            status=status,
            error_message=error,
            attempts=attempts,
        )

    @staticmethod
    def _invalid_summary(position: int, message: str, chunk: Optional[TextChunk] = None) -> ChunkSummary:
        return ChunkSummary(
            chunk_id=f"invalid-{position}",
            chunk_index=position,
            speakers=list(chunk.speakers) if chunk else [],
            source_filename=chunk.source_filename if chunk else None,
            status=SummaryStatus.FAILED,
            error_message=message,
        )

    # --------------------------------------------------------- global summary
    def generate_global_summary(
        self,
        chunk_summaries: Sequence[ChunkSummary],
        source_filename: Optional[str] = None,
        total_chunks: Optional[int] = None,
        warnings: Optional[List[str]] = None,
    ) -> GlobalExecutiveSummary:
        """Combine chunk summaries into the executive summary of the whole meeting."""
        summaries = sorted(chunk_summaries, key=lambda s: s.chunk_index)
        local_warnings = list(warnings or [])
        total = total_chunks if total_chunks is not None else len(summaries)
        ok = [s for s in summaries if s.status == SummaryStatus.SUCCESS]
        failed = [s for s in summaries if s.status == SummaryStatus.FAILED]
        starts = [s.start_timestamp for s in summaries if s.start_timestamp]
        ends = [s.end_timestamp for s in summaries if s.end_timestamp]
        base: Dict[str, Any] = dict(
            source_filename=source_filename or next((s.source_filename for s in summaries if s.source_filename), None),
            total_chunks=total,
            summarized_chunks=len(ok),
            failed_chunk_ids=[s.chunk_id for s in failed],
            speakers=_unique([name for s in summaries for name in s.speakers]),
            start_timestamp=starts[0] if starts else None,
            end_timestamp=ends[-1] if ends else None,
        )

        if not ok:
            if failed:
                return GlobalExecutiveSummary(
                    status=GlobalSummaryStatus.FAILED,
                    error_message=(
                        "No chunk could be summarized, so no global summary was generated. "
                        f"First error: {failed[0].error_message}"
                    ),
                    warnings=local_warnings,
                    **base,
                )
            return GlobalExecutiveSummary(
                status=GlobalSummaryStatus.EMPTY, warnings=local_warnings, **base
            )

        nodes = [_node_from_summary(s) for s in ok]
        nodes, levels = self._reduce_to_fit(nodes, local_warnings)
        budget = self.config.reduction_batch_chars
        if total_size(nodes) > budget:
            local_warnings.append(
                f"The combined summaries ({total_size(nodes)} characters) are larger than "
                f"the configured limit ({budget}); the final request may be too large."
            )

        covered = format_part_ranges([s.chunk_index + 1 for s in ok])
        if failed:
            missing = ", ".join(str(s.chunk_index + 1) for s in failed)
            coverage_note = f"Parts {missing} could not be summarized and are NOT covered below."
        else:
            coverage_note = "All parts are covered."
        inputs = {
            "output_language": self.config.output_language,
            "source_filename": base["source_filename"] or "unknown",
            "total_parts": total,
            "covered_parts": covered,
            "speakers": ", ".join(base["speakers"]) or "not labeled",
            "time_range": format_time_range(base["start_timestamp"], base["end_timestamp"]),
            "coverage_note": coverage_note,
            "summaries_text": _NODE_JOIN.join(render_node(n) for n in nodes),
            "format_instructions": self._global_parser.get_format_instructions(),
        }
        try:
            output, _ = self._run_stage(
                "global", inputs, self._global_parser, "Writing the executive summary"
            )
        except LLMCallError as exc:
            return GlobalExecutiveSummary(
                status=GlobalSummaryStatus.FAILED,
                error_message=str(exc),
                reduction_levels=levels,
                warnings=local_warnings,
                **base,
            )

        if failed:
            local_warnings.append(
                f"This summary does not cover part(s) {', '.join(str(s.chunk_index + 1) for s in failed)} "
                "because they could not be summarized."
            )
        return GlobalExecutiveSummary(
            meeting_title=output.meeting_title,
            executive_summary=output.executive_summary,
            meeting_overview=output.meeting_overview,
            major_topics=output.major_topics,
            key_takeaways=output.key_takeaways,
            important_discussions=output.important_discussions,
            unresolved_questions=output.unresolved_questions,
            status=GlobalSummaryStatus.PARTIAL if failed else GlobalSummaryStatus.COMPLETE,
            reduction_levels=levels,
            warnings=local_warnings,
            **base,
        )

    def _reduce_to_fit(self, nodes: List[SummaryNode], warnings: List[str]) -> Tuple[List[SummaryNode], int]:
        """Merge groups of consecutive summaries until they fit in one request."""
        budget = self.config.reduction_batch_chars
        levels = 0
        while (
            total_size(nodes) > budget
            and len(nodes) > 1
            and levels < self.config.max_reduction_levels
        ):
            batches = batch_nodes(nodes, budget)
            if len(batches) >= len(nodes):
                warnings.append(
                    "Summaries could not be grouped further within the size limit."
                )
                break
            merged_nodes: List[SummaryNode] = []
            for batch in batches:
                if len(batch) == 1:
                    merged_nodes.append(batch[0])
                    continue
                try:
                    merged_nodes.append(self._merge_batch(batch))
                except LLMCallError as exc:
                    first, last = batch[0].first_chunk_index + 1, batch[-1].last_chunk_index + 1
                    warnings.append(
                        f"Parts {first}\u2013{last} could not be merged ({exc}); "
                        "their summaries were kept unmerged."
                    )
                    merged_nodes.extend(batch)
            if len(merged_nodes) >= len(nodes):
                warnings.append("Merging did not shorten the summaries; stopping reduction.")
                break
            nodes = merged_nodes
            levels += 1
        if total_size(nodes) > budget and levels >= self.config.max_reduction_levels:
            warnings.append(
                f"Reached the maximum of {self.config.max_reduction_levels} reduction levels."
            )
        return nodes, levels

    def _merge_batch(self, batch: List[SummaryNode]) -> SummaryNode:
        first = min(n.first_chunk_index for n in batch) + 1
        last = max(n.last_chunk_index for n in batch) + 1
        speakers = _unique([s for n in batch for s in n.speakers])
        starts = [n.start_timestamp for n in batch if n.start_timestamp]
        ends = [n.end_timestamp for n in batch if n.end_timestamp]
        inputs = {
            "output_language": self.config.output_language,
            "position_range": f"{first}\u2013{last}",
            "speakers": ", ".join(speakers) or "not labeled",
            "time_range": format_time_range(starts[0] if starts else None, ends[-1] if ends else None),
            "summaries_text": _NODE_JOIN.join(render_node(n) for n in batch),
            "format_instructions": self._chunk_parser.get_format_instructions(),
        }
        output, _ = self._run_stage(
            "reduce", inputs, self._chunk_parser, f"Merging parts {first}\u2013{last}"
        )
        return merge_node_metadata(batch, output.summary, output.key_points, output.unresolved_items)

    # --------------------------------------------------------------- pipeline
    def summarize_chunking_result(self, result: ChunkingResult) -> MeetingSummaryResult:
        """Run the full Day 4 pipeline on the output of the Day 3 chunking service."""
        started = time.perf_counter()
        calls_before, retries_before = self.llm_call_count, self.retry_count
        warnings: List[str] = [f"Chunking: {w}" for w in result.warnings]

        summaries = self.summarize_chunks(result.chunks, warnings)
        for s in summaries:
            if s.status == SummaryStatus.FAILED:
                warnings.append(f"{s.chunk_id} failed: {s.error_message}")
            elif s.status == SummaryStatus.SKIPPED:
                warnings.append(f"{s.chunk_id} skipped: {s.error_message}")

        global_summary = self.generate_global_summary(
            summaries, result.source_filename, len(result.chunks), warnings
        )
        return MeetingSummaryResult(
            source_filename=result.source_filename,
            status=global_summary.status,
            chunk_summaries=summaries,
            global_summary=global_summary,
            llm_call_count=self.llm_call_count - calls_before,
            retry_count=self.retry_count - retries_before,
            processing_seconds=time.perf_counter() - started,
            warnings=list(global_summary.warnings),
        )


# ------------------------------------------------------ convenience functions
def summarize_chunking_result(
    result: ChunkingResult,
    llm: Any = None,
    config: Optional[SummarizationConfig] = None,
    settings: Any = None,
) -> MeetingSummaryResult:
    """Summarize the output of chunk_file / chunk_text / chunk_ingestion_result."""
    return SummarizationService(llm, config, settings).summarize_chunking_result(result)


def summarize_text(
    text: str,
    llm: Any = None,
    chunking_config: Optional[ChunkingConfig] = None,
    config: Optional[SummarizationConfig] = None,
    settings: Any = None,
) -> MeetingSummaryResult:
    """Normalize, chunk and summarize a transcript string."""
    return summarize_chunking_result(chunk_text(text, chunking_config), llm, config, settings)


def summarize_file(
    path: Union[str, Path],
    llm: Any = None,
    chunking_config: Optional[ChunkingConfig] = None,
    config: Optional[SummarizationConfig] = None,
    settings: Any = None,
) -> MeetingSummaryResult:
    """Ingest a TXT/DOCX/PDF file, chunk it and summarize it (Day 2 -> 3 -> 4)."""
    return summarize_chunking_result(chunk_file(path, chunking_config), llm, config, settings)
