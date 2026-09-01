"""Deterministically rebuild source-grounded Plots from model boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rar_agent.domain.models import (
    CharacterRef,
    Plot,
    PlotChunk,
    PlotExtractionResult,
    PlotsDocument,
    TextChunk,
    TextChunkSpanRef,
)
from rar_agent.text.chunking import split_sentences
from rar_agent.text.tokenizer import Tokenizer

TEXT_CHUNKS_PATH = "work/text_chunks.jsonl"


@dataclass(frozen=True, slots=True)
class _SourcePiece:
    text_chunk_position: int
    source_start: int
    source_end: int
    text: str


@dataclass(slots=True)
class _OpenPlot:
    meta: dict[str, object]
    pieces: list[_SourcePiece] = field(default_factory=list)
    character_refs: list[CharacterRef] = field(default_factory=list)


class PlotRebuilder:
    def __init__(self, tokenizer: Tokenizer, *, target_tokens: int) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens

    def rebuild(
        self,
        chunks: list[TextChunk],
        results: list[PlotExtractionResult],
        candidate_character_refs: dict[int, CharacterRef] | None = None,
    ) -> PlotsDocument:
        if len(chunks) != len(results):
            raise ValueError("every TextChunk requires one Plot extraction result")
        refs_by_candidate = candidate_character_refs or {}
        candidate_offsets = self._candidate_offsets(results)
        recovered: list[_OpenPlot] = []
        open_plot: _OpenPlot | None = None

        def close_open() -> None:
            nonlocal open_plot
            if open_plot is None:
                return
            if not open_plot.pieces:
                raise ValueError("cannot close an empty Plot")
            recovered.append(open_plot)
            open_plot = None

        for position, (chunk, result) in enumerate(zip(chunks, results, strict=True)):
            self._validate_boundaries(chunk, result)
            section_end = position == len(chunks) - 1 or chunks[position + 1].meta != chunk.meta
            boundaries = list(result.plots)
            refs_for = self._refs_for_result(
                result, candidate_offsets[position], refs_by_candidate
            )

            if open_plot is not None and open_plot.meta != chunk.meta:
                close_open()

            cursor = 0
            if open_plot is not None:
                open_plot.character_refs = _unique_refs(
                    open_plot.character_refs + refs_for(None)
                )
                if not boundaries:
                    open_plot.pieces.append(self._piece(position, chunk, 0, len(chunk.text)))
                    if section_end:
                        close_open()
                    continue

                _continuation_start, continuation_end = boundaries.pop(0)
                if continuation_end is None:
                    open_plot.pieces.append(self._piece(position, chunk, 0, len(chunk.text)))
                    if boundaries:
                        raise ValueError(
                            "an open continuation cannot precede later Plot boundaries"
                        )
                    if section_end:
                        close_open()
                    continue
                continuation_at = self._find_boundary(
                    chunk.text, continuation_end, 0, "continuation end"
                )
                cursor = continuation_at + len(continuation_end)
                open_plot.pieces.append(self._piece(position, chunk, 0, cursor))
                open_plot.character_refs = _unique_refs(
                    open_plot.character_refs + refs_for(0)
                )
                close_open()

            if not boundaries:
                open_plot = _OpenPlot(
                    meta=dict(chunk.meta),
                    pieces=[self._piece(position, chunk, cursor, len(chunk.text))],
                    character_refs=refs_for(None),
                )
                if section_end:
                    close_open()
                continue

            consumed_count = len(result.plots) - len(boundaries)
            for remaining_index, (start, end) in enumerate(boundaries):
                local_index = consumed_count + remaining_index
                start_at = cursor if start is None else self._find_boundary(
                    chunk.text, start, cursor, f"plots[{local_index}] start"
                )
                if end is None:
                    open_plot = _OpenPlot(
                        meta=dict(chunk.meta),
                        pieces=[self._piece(position, chunk, start_at, len(chunk.text))],
                        character_refs=refs_for(local_index),
                    )
                    cursor = len(chunk.text)
                    if remaining_index != len(boundaries) - 1:
                        raise ValueError("an unclosed Plot cannot precede later Plot boundaries")
                    continue
                end_at = self._find_boundary(
                    chunk.text, end, start_at, f"plots[{local_index}] end"
                ) + len(end)
                recovered.append(
                    _OpenPlot(
                        meta=dict(chunk.meta),
                        pieces=[self._piece(position, chunk, start_at, end_at)],
                        character_refs=refs_for(local_index),
                    )
                )
                cursor = end_at

            if section_end:
                close_open()

        close_open()
        plots = [self._build_plot(index, value) for index, value in enumerate(recovered)]
        return PlotsDocument(plots=plots)

    @staticmethod
    def _candidate_offsets(results: list[PlotExtractionResult]) -> list[int]:
        offsets: list[int] = []
        current = 0
        for result in results:
            offsets.append(current)
            current += len(result.characters)
        return offsets

    @staticmethod
    def _refs_for_result(
        result: PlotExtractionResult,
        candidate_offset: int,
        refs_by_candidate: dict[int, CharacterRef],
    ) -> Callable[[int | None], list[CharacterRef]]:
        def resolve(local_plot_index: int | None) -> list[CharacterRef]:
            refs: list[CharacterRef] = []
            for local_candidate_index, candidate in enumerate(result.characters):
                applies = candidate.plot_indexes is None or (
                    local_plot_index is not None
                    and local_plot_index in candidate.plot_indexes
                )
                ref = refs_by_candidate.get(candidate_offset + local_candidate_index)
                if applies and ref is not None:
                    refs.append(ref)
            return _unique_refs(refs)

        return resolve

    @staticmethod
    def _piece(
        position: int, chunk: TextChunk, start: int, end: int
    ) -> _SourcePiece:
        if not 0 <= start < end <= len(chunk.text):
            raise ValueError("reconstructed source slice is empty or outside TextChunk")
        return _SourcePiece(position, start, end, chunk.text[start:end])

    @staticmethod
    def _find_boundary(text: str, boundary: str, start: int, label: str) -> int:
        position = text.find(boundary, start)
        if position < 0:
            raise ValueError(f"cannot locate {label}: {boundary!r}")
        return position

    @staticmethod
    def _validate_boundaries(chunk: TextChunk, result: PlotExtractionResult) -> None:
        for plot_index, boundary in enumerate(result.plots):
            for label, value in zip(("start", "end"), boundary, strict=True):
                if value is not None and value not in chunk.text:
                    raise ValueError(f"plots[{plot_index}] {label} is not in TextChunk")

    def _build_plot(self, index: int, recovered: _OpenPlot) -> Plot:
        return Plot(
            index=index,
            meta=recovered.meta,
            character_refs=recovered.character_refs,
            chunks=self._rechunk(recovered.pieces),
        )

    def _rechunk(self, pieces: list[_SourcePiece]) -> list[PlotChunk]:
        full_text = "".join(piece.text for piece in pieces)
        sentence_ranges: list[tuple[int, int]] = []
        cursor = 0
        for sentence in split_sentences(full_text):
            end = cursor + len(sentence)
            sentence_ranges.append((cursor, end))
            cursor = end

        ranges: list[tuple[int, int]] = []
        pending_start: int | None = None
        pending_end = 0
        for start, end in sentence_ranges:
            candidate_start = start if pending_start is None else pending_start
            candidate = full_text[candidate_start:end]
            if pending_start is not None and self.tokenizer.count(candidate) > self.target_tokens:
                ranges.append((pending_start, pending_end))
                pending_start = start
            elif pending_start is None:
                pending_start = start
            pending_end = end
        if pending_start is not None:
            ranges.append((pending_start, pending_end))

        return [
            PlotChunk(
                index=chunk_index,
                text=full_text[start:end],
                token_count=self.tokenizer.count(full_text[start:end]),
                source_refs=self._source_refs(pieces, start, end),
            )
            for chunk_index, (start, end) in enumerate(ranges)
        ]

    @staticmethod
    def _source_refs(
        pieces: list[_SourcePiece], output_start: int, output_end: int
    ) -> list[TextChunkSpanRef]:
        refs: list[TextChunkSpanRef] = []
        full_cursor = 0
        for piece in pieces:
            piece_start = full_cursor
            piece_end = piece_start + len(piece.text)
            overlap_start = max(piece_start, output_start)
            overlap_end = min(piece_end, output_end)
            if overlap_start < overlap_end:
                source_start = piece.source_start + overlap_start - piece_start
                source_end = piece.source_start + overlap_end - piece_start
                refs.append(
                    TextChunkSpanRef(
                        path=TEXT_CHUNKS_PATH,
                        index=piece.text_chunk_position,
                        start=source_start,
                        end=source_end,
                    )
                )
            full_cursor = piece_end
        return refs


def _unique_refs(refs: list[CharacterRef]) -> list[CharacterRef]:
    result: list[CharacterRef] = []
    seen: set[tuple[str, int]] = set()
    for ref in refs:
        key = (ref.path, ref.index)
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result
