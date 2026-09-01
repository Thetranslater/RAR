"""Recover plot text from LLM boundaries and rechunk each plot by tokens."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from .chunker import TiktokenCounter, split_sentences
except ImportError:  # Support direct execution from the project root.
    from chunker import TiktokenCounter, split_sentences


class TokenCounter(Protocol):
    @property
    def name(self) -> str: ...

    def count(self, text: str) -> int: ...


Boundary = tuple[str | None, str | None]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class Character:
    names: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    volume: str | None
    chapter: str | None
    text: str


@dataclass(frozen=True)
class PlotRecord:
    chunk_id: str
    state: str
    plots: tuple[Boundary, ...]
    characters: tuple[Character, ...] = ()


@dataclass(frozen=True)
class RecoveredPlot:
    volume: str | None
    chapter: str | None
    text: str
    characters: tuple[Character, ...] = ()


@dataclass
class OpenPlot:
    volume: str | None
    chapter: str | None
    parts: list[str]
    characters: tuple[Character, ...]


def load_source_chunks(path: Path) -> tuple[str, list[SourceChunk]]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(root, dict) or not isinstance(root.get("chunks"), list):
        raise ValueError("Source chunk JSON must contain a chunks array.")

    chunks: list[SourceChunk] = []
    for index, item in enumerate(root["chunks"]):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"chunks[{index}].text must be a string.")
        chunks.append(
            SourceChunk(
                chunk_id=f"chunk-{index:06d}",
                volume=item.get("volume"),
                chapter=item.get("chapter"),
                text=item["text"],
            )
        )
    title = root.get("title")
    return (title if isinstance(title, str) else path.stem), chunks


def load_plot_records(path: Path) -> dict[str, PlotRecord]:
    records: dict[str, PlotRecord] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        chunk_id = raw.get("chunk_id")
        result = raw.get("result")
        if not isinstance(chunk_id, str) or not isinstance(result, dict):
            raise ValueError(f"Invalid plot record at line {line_number}.")
        if chunk_id in records:
            raise ValueError(f"Duplicate plot record for {chunk_id}.")

        state = result.get("state")
        raw_plots = result.get("plots")
        raw_characters = result.get("characters")
        if (
            state not in {"truncated", "finished"}
            or not isinstance(raw_plots, list)
            or not isinstance(raw_characters, list)
        ):
            raise ValueError(f"Invalid result for {chunk_id}.")
        characters: list[Character] = []
        for character_index, raw_character in enumerate(raw_characters):
            if (
                not isinstance(raw_character, dict)
                or set(raw_character) != {"name", "description"}
                or not isinstance(raw_character["name"], list)
                or not raw_character["name"]
                or any(
                    not isinstance(name, str) or not name.strip() for name in raw_character["name"]
                )
                or not isinstance(raw_character["description"], str)
                or not raw_character["description"].strip()
            ):
                raise ValueError(f"{chunk_id}.characters[{character_index}] is invalid.")
            characters.append(
                Character(
                    names=tuple(raw_character["name"]),
                    description=raw_character["description"],
                )
            )
        boundaries: list[Boundary] = []
        for plot_index, pair in enumerate(raw_plots):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(value is not None and not isinstance(value, str) for value in pair)
            ):
                raise ValueError(f"{chunk_id}.plots[{plot_index}] is invalid.")
            start, end = pair
            if start == "" or end == "":
                raise ValueError(f"{chunk_id}.plots[{plot_index}] contains an empty boundary.")
            boundaries.append((start, end))
        records[chunk_id] = PlotRecord(
            chunk_id,
            state,
            tuple(boundaries),
            merge_characters(characters),
        )
    return records


def select_source_chunks(
    source_chunks: list[SourceChunk],
    records: dict[str, PlotRecord],
    *,
    allow_partial: bool,
    warnings: list[str],
) -> list[SourceChunk]:
    source_ids = {chunk.chunk_id for chunk in source_chunks}
    unknown = sorted(set(records) - source_ids)
    if unknown:
        raise ValueError(f"Plot records contain unknown chunk IDs: {', '.join(unknown[:5])}.")

    if not allow_partial:
        missing = [chunk.chunk_id for chunk in source_chunks if chunk.chunk_id not in records]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Missing {len(missing)} plot record(s), beginning with {preview}.")
        return source_chunks

    selected: list[SourceChunk] = []
    for chunk in source_chunks:
        if chunk.chunk_id not in records:
            break
        selected.append(chunk)
    if not selected:
        raise ValueError("No contiguous plot records starting at chunk-000000.")
    if len(selected) < len(source_chunks):
        missing_id = source_chunks[len(selected)].chunk_id
        ignored = sum(chunk.chunk_id in records for chunk in source_chunks[len(selected) + 1 :])
        warnings.append(
            f"Partial mode stopped before {missing_id}; ignored {ignored} later record(s)."
        )
    return selected


def _find_boundary(text: str, boundary: str, start: int, label: str) -> int:
    index = text.find(boundary, start)
    if index < 0:
        raise ValueError(f"Cannot find {label} boundary after offset {start}: {boundary!r}")
    return index


def _join_parts(parts: list[str]) -> str:
    return "".join(parts)


def safe_filename(value: str) -> str:
    return SAFE_FILENAME_RE.sub("_", value).strip(" .") or "untitled"


def warning_log_path(title: str) -> Path:
    return PROJECT_ROOT / "debug" / "plots" / safe_filename(title) / "rebuild-warnings-log.txt"


def write_warning_log(
    path: Path,
    *,
    title: str,
    source_chunks: Path,
    plot_results: Path,
    output_json: Path,
    warnings: list[str],
) -> None:
    lines = [
        "RLFF plot rebuild warning log",
        "",
        f"source_title: {title}",
        f"source_chunks: {source_chunks.resolve()}",
        f"plot_results: {plot_results.resolve()}",
        f"output_json: {output_json.resolve()}",
        f"warning_count: {len(warnings)}",
        "",
        "=== WARNINGS ===",
    ]
    lines.extend(f"WARNING: {warning}" for warning in warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_characters(
    *groups: Iterable[Character],
) -> tuple[Character, ...]:
    """Merge character groups by overlapping aliases while preserving order."""

    merged: list[Character] = []
    for group in groups:
        for character in group:
            aliases = set(character.names)
            overlapping = [
                index
                for index, existing in enumerate(merged)
                if aliases.intersection(existing.names)
            ]
            if not overlapping:
                merged.append(character)
                continue

            first = overlapping[0]
            combined = [merged[index] for index in overlapping]
            combined.append(character)
            names: list[str] = []
            for item in combined:
                for name in item.names:
                    if name not in names:
                        names.append(name)
            description = max(
                (item.description for item in combined),
                key=len,
            )
            merged[first] = Character(tuple(names), description)
            for index in reversed(overlapping[1:]):
                del merged[index]
    return tuple(merged)


def recover_plots(
    source_chunks: list[SourceChunk],
    records: dict[str, PlotRecord],
    *,
    allow_incomplete_tail: bool = False,
    warnings: list[str] | None = None,
) -> list[RecoveredPlot]:
    warnings = warnings if warnings is not None else []
    recovered: list[RecoveredPlot] = []
    open_plot: OpenPlot | None = None

    effective_boundaries: dict[str, list[Boundary]] = {}
    effective_states: dict[str, str] = {}
    for chunk in source_chunks:
        record = records[chunk.chunk_id]
        boundaries = list(record.plots)
        state = record.state

        if state == "truncated" and boundaries and boundaries[-1][1] is not None:
            start, _end = boundaries[-1]
            boundaries[-1] = (start, None)
            warnings.append(
                f"{chunk.chunk_id} ended with state=truncated but a non-null tail; "
                "the tail boundary was ignored."
            )
        elif state == "finished" and boundaries and boundaries[-1][1] is None:
            warnings.append(
                f"{chunk.chunk_id} ended with state=finished but a null tail; "
                "the plot remains open."
            )

        boundary_index = 1
        while boundary_index < len(boundaries):
            start, end = boundaries[boundary_index]
            previous_start, previous_end = boundaries[boundary_index - 1]
            if start is None and previous_end is not None:
                boundaries[boundary_index - 1] = (previous_start, end)
                del boundaries[boundary_index]
                warnings.append(
                    f"{chunk.chunk_id}.plots[{boundary_index}] had an orphan null start; "
                    "the previous finished plot was extended and both plots were merged."
                )
                continue
            boundary_index += 1

        effective_boundaries[chunk.chunk_id] = boundaries
        effective_states[chunk.chunk_id] = state

    for chunk_index, chunk in enumerate(source_chunks):
        boundaries = effective_boundaries[chunk.chunk_id]
        if not boundaries or boundaries[0][0] is not None:
            continue

        for previous_index in range(chunk_index - 1, -1, -1):
            previous_chunk = source_chunks[previous_index]
            previous_boundaries = effective_boundaries[previous_chunk.chunk_id]
            if not previous_boundaries:
                continue
            previous_start, previous_end = previous_boundaries[-1]
            if previous_end is not None:
                previous_boundaries[-1] = (previous_start, None)
                effective_states[previous_chunk.chunk_id] = "truncated"
                warnings.append(
                    f"{chunk.chunk_id}.plots[0] had an orphan null start; "
                    f"{previous_chunk.chunk_id}'s previous finished plot was changed "
                    "to truncated and extended."
                )
            break

    def warn_chapter_change(chunk: SourceChunk) -> None:
        if open_plot and (chunk.volume != open_plot.volume or chunk.chapter != open_plot.chapter):
            warnings.append(
                f"{chunk.chunk_id} continues a plot across a volume/chapter boundary; "
                "the starting metadata was kept."
            )

    def close_open() -> None:
        nonlocal open_plot
        assert open_plot is not None
        text = _join_parts(open_plot.parts)
        if not text:
            raise ValueError("Recovered an empty cross-chunk plot.")
        recovered.append(
            RecoveredPlot(
                open_plot.volume,
                open_plot.chapter,
                text,
                open_plot.characters,
            )
        )
        open_plot = None

    for chunk in source_chunks:
        record = records[chunk.chunk_id]
        boundaries = list(effective_boundaries[chunk.chunk_id])
        state = effective_states[chunk.chunk_id]

        cursor = 0
        if open_plot is not None:
            warn_chapter_change(chunk)
            open_plot.characters = merge_characters(
                open_plot.characters,
                record.characters,
            )
            if not boundaries:
                open_plot.parts.append(chunk.text)
                continue

            continuation_start, continuation_end = boundaries.pop(0)
            if continuation_start is not None:
                warnings.append(
                    f"{chunk.chunk_id} continuation started with a non-null boundary; "
                    "it was treated as null."
                )
            if continuation_end is None:
                if boundaries:
                    raise ValueError(f"{chunk.chunk_id} has plots after an unclosed continuation.")
                open_plot.parts.append(chunk.text)
                continue

            end_index = _find_boundary(
                chunk.text, continuation_end, 0, f"{chunk.chunk_id} continuation end"
            )
            cursor = end_index + len(continuation_end)
            open_plot.parts.append(chunk.text[:cursor])
            close_open()

        if not boundaries:
            if state == "truncated" and open_plot is None:
                raise ValueError(f"{chunk.chunk_id} has state=truncated and no plot to continue.")
            continue

        for plot_index, (start, end) in enumerate(boundaries):
            if start is None:
                raise ValueError(f"{chunk.chunk_id}.plots[{plot_index}] has an orphan null start.")
            start_index = _find_boundary(
                chunk.text, start, cursor, f"{chunk.chunk_id}.plots[{plot_index}] start"
            )
            if end is None:
                if plot_index != len(boundaries) - 1:
                    raise ValueError(
                        f"{chunk.chunk_id}.plots[{plot_index}] is unclosed before later plots."
                    )
                open_plot = OpenPlot(
                    volume=chunk.volume,
                    chapter=chunk.chapter,
                    parts=[chunk.text[start_index:]],
                    characters=record.characters,
                )
                cursor = len(chunk.text)
                continue

            end_index = _find_boundary(
                chunk.text, end, start_index, f"{chunk.chunk_id}.plots[{plot_index}] end"
            )
            cursor = end_index + len(end)
            text = chunk.text[start_index:cursor]
            if not text.strip():
                raise ValueError(f"{chunk.chunk_id}.plots[{plot_index}] recovered empty text.")
            recovered.append(
                RecoveredPlot(
                    chunk.volume,
                    chunk.chapter,
                    text,
                    record.characters,
                )
            )

    if open_plot is not None:
        if not allow_incomplete_tail:
            raise ValueError("The final plot is still truncated at the end of available chunks.")
        warnings.append("Discarded one incomplete plot at the end of the available chunks.")
    return recovered


def rechunk_plot(text: str, budget: int, counter: TokenCounter) -> list[dict[str, Any]]:
    if budget < 1:
        raise ValueError("token budget must be a positive integer.")
    chunks: list[dict[str, Any]] = []
    pending: list[str] = []

    def emit() -> None:
        chunk_text = "".join(pending)
        chunks.append({"text": chunk_text, "token": counter.count(chunk_text)})
        pending.clear()

    for sentence in split_sentences(text):
        pending.append(sentence)
        if counter.count("".join(pending)) >= budget:
            emit()
    if pending:
        emit()
    return chunks


def build_output(
    title: str,
    plots: list[RecoveredPlot],
    budget: int,
    counter: TokenCounter,
) -> dict[str, Any]:
    return {
        "title": title,
        "chunking": {
            "token_budget": budget,
            "token_encoding": counter.name,
        },
        "plots": [
            {
                "volume": plot.volume,
                "chapter": plot.chapter,
                "characters": [
                    {
                        "name": list(character.names),
                        "description": character.description,
                    }
                    for character in plot.characters
                ],
                "chunks": rechunk_plot(plot.text, budget, counter),
            }
            for plot in plots
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover complete plots from extracted boundaries and rechunk them."
    )
    parser.add_argument("source_chunks", type=Path, help="Original chunk JSON.")
    parser.add_argument("plot_results", type=Path, help="Plot extraction JSONL.")
    parser.add_argument("output_json", type=Path, help="Rebuilt plot JSON.")
    parser.add_argument(
        "--token",
        type=int,
        default=512,
        help="Token budget for each rebuilt chunk (default: 512).",
    )
    parser.add_argument(
        "--token-encoding",
        default="o200k_base",
        help="tiktoken encoding used for counting (default: o200k_base).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use only the contiguous result prefix (the default).",
    )
    parser.add_argument(
        "--require-complete",
        dest="allow_partial",
        action="store_false",
        help="Require a plot result for every source chunk.",
    )
    parser.set_defaults(allow_partial=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.token < 1:
        raise SystemExit("--token must be a positive integer.")

    warnings: list[str] = []
    try:
        title, all_chunks = load_source_chunks(args.source_chunks)
        records = load_plot_records(args.plot_results)
        selected_chunks = select_source_chunks(
            all_chunks,
            records,
            allow_partial=args.allow_partial,
            warnings=warnings,
        )
        plots = recover_plots(
            selected_chunks,
            records,
            allow_incomplete_tail=args.allow_partial,
            warnings=warnings,
        )
        counter = TiktokenCounter(args.token_encoding)
        output = build_output(title, plots, args.token, counter)
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = warning_log_path(title)
    write_warning_log(
        log_path,
        title=title,
        source_chunks=args.source_chunks,
        plot_results=args.plot_results,
        output_json=args.output_json,
        warnings=warnings,
    )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(
        f"Wrote {len(plots)} plots from {len(selected_chunks)} source chunks to {args.output_json}"
    )
    print(f"Wrote {len(warnings)} warning(s) to {log_path}")


if __name__ == "__main__":
    main()
