"""Dialogue alignment and deterministic Conversation assembly."""

from __future__ import annotations

from pydantic import Field, model_validator
from rapidfuzz import fuzz

from rar_agent.domain.models import (
    Conversation,
    DomainModel,
    Plot,
    PlotChunk,
    PlotChunkRef,
    PlotRef,
    TextChunkSpanRef,
    Utterance,
)

PLOTS_PATH = "work/plots.json"


class RawUtterance(DomainModel):
    speaker: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_environment(self) -> RawUtterance:
        if self.speaker == "Environment" and not (
            self.content.startswith("*(")
            and self.content.endswith(")*")
            and self.content[2:-2].strip()
        ):
            raise ValueError("Environment content must use a non-empty *(...)* form")
        return self


class DialogueExtractionResult(DomainModel):
    utterances: list[RawUtterance]

    @property
    def is_empty(self) -> bool:
        return not self.utterances


class DialogueBatch(DomainModel):
    plot_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    utterances: list[Utterance]


class DialogueAligner:
    def __init__(self, *, fuzzy_threshold: float = 90.0, fuzzy_min_chars: int = 6) -> None:
        if not 0 <= fuzzy_threshold <= 100:
            raise ValueError("fuzzy_threshold must be between 0 and 100")
        if fuzzy_min_chars < 1:
            raise ValueError("fuzzy_min_chars must be positive")
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_min_chars = fuzzy_min_chars

    def align(
        self,
        plot_index: int,
        chunk: PlotChunk,
        raw_utterances: list[RawUtterance],
        *,
        aliases: dict[str, str],
    ) -> DialogueBatch:
        cursor = 0
        utterances: list[Utterance] = []
        plot_ref = PlotChunkRef(
            path=PLOTS_PATH, plot_index=plot_index, chunk_index=chunk.index
        )
        for index, raw in enumerate(raw_utterances):
            speaker = aliases.get(raw.speaker, raw.speaker)
            source_refs: list[PlotChunkRef | TextChunkSpanRef] = [plot_ref]
            if speaker != "Environment":
                match = self._locate(raw.content, chunk.text, cursor)
                if match is not None:
                    start, end = match
                    source_refs.extend(self._map_to_text_chunks(chunk, start, end))
                    cursor = end
            utterances.append(
                Utterance(
                    index=index,
                    speaker=speaker,
                    content=raw.content,
                    source_refs=source_refs,
                )
            )
        return DialogueBatch(
            plot_index=plot_index, chunk_index=chunk.index, utterances=utterances
        )

    def _locate(self, content: str, source: str, cursor: int) -> tuple[int, int] | None:
        exact = source.find(content, cursor)
        if exact >= 0:
            return exact, exact + len(content)
        compact_length = sum(not char.isspace() for char in content)
        if compact_length < self.fuzzy_min_chars or cursor >= len(source):
            return None
        alignment = fuzz.partial_ratio_alignment(
            content,
            source[cursor:],
            score_cutoff=self.fuzzy_threshold,
        )
        if alignment is None:
            return None
        start = cursor + int(alignment.dest_start)
        end = cursor + int(alignment.dest_end)
        if not cursor <= start < end <= len(source):
            return None
        return start, end

    @staticmethod
    def _map_to_text_chunks(
        chunk: PlotChunk, match_start: int, match_end: int
    ) -> list[TextChunkSpanRef]:
        refs: list[TextChunkSpanRef] = []
        plot_cursor = 0
        for source_ref in chunk.source_refs:
            source_length = source_ref.end - source_ref.start
            plot_end = plot_cursor + source_length
            overlap_start = max(match_start, plot_cursor)
            overlap_end = min(match_end, plot_end)
            if overlap_start < overlap_end:
                refs.append(
                    TextChunkSpanRef(
                        path=source_ref.path,
                        index=source_ref.index,
                        start=source_ref.start + overlap_start - plot_cursor,
                        end=source_ref.start + overlap_end - plot_cursor,
                    )
                )
            plot_cursor = plot_end
        return refs


class ConversationAssembler:
    def assemble(self, plot: Plot, batches: list[DialogueBatch]) -> Conversation:
        expected = {chunk.index for chunk in plot.chunks}
        actual = {batch.chunk_index for batch in batches}
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"dialogue batches do not match Plot chunks: missing={missing}, extra={extra}"
            )
        ordered = sorted(batches, key=lambda batch: batch.chunk_index)
        utterances: list[Utterance] = []
        for batch in ordered:
            for item in batch.utterances:
                utterances.append(item.model_copy(update={"index": len(utterances)}))
        return Conversation(
            plot_ref=PlotRef(path=PLOTS_PATH, index=plot.index),
            utterances=utterances,
        )
