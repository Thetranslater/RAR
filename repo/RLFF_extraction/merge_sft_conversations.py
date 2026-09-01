"""Merge SFT messages into canonical four-chunk conversation JSONL records."""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .build_rlff_dataset import (
        IssueLog,
        MessageUnit,
        ParsedSftRecord,
        PlotText,
        _match_unit_positions,
        _normalize_with_positions,
        choose_plot,
        load_sft_records,
        normalize_text,
        score_record,
    )
except ImportError:
    from build_rlff_dataset import (  # type: ignore[no-redef]
        IssueLog,
        MessageUnit,
        ParsedSftRecord,
        PlotText,
        _match_unit_positions,
        _normalize_with_positions,
        choose_plot,
        load_sft_records,
        normalize_text,
        score_record,
    )


@dataclass(frozen=True)
class InputSpec:
    input_index: int
    plot_index: int
    source_chunk_start: int
    source_chunk_end: int
    source_token_count: int
    volume: str | None
    chapter: str | None
    text: str


@dataclass(frozen=True)
class LocatedUnit:
    position: int
    sft_line: int
    unit_index: int
    unit: MessageUnit


def _load_rebuilt(
    path: Path,
    chunks_per_input: int,
) -> tuple[str, list[PlotText], dict[tuple[int, int], InputSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("title"), str)
        or not isinstance(raw.get("plots"), list)
    ):
        raise ValueError("Rebuilt plot JSON must contain title and plots.")

    plots: list[PlotText] = []
    inputs: dict[tuple[int, int], InputSpec] = {}
    input_index = 0
    for plot_index, raw_plot in enumerate(raw["plots"]):
        if not isinstance(raw_plot, dict) or not isinstance(
            raw_plot.get("chunks"), list
        ):
            raise ValueError(f"plots[{plot_index}] must contain chunks.")
        chunks: list[dict[str, Any]] = []
        for chunk_index, raw_chunk in enumerate(raw_plot["chunks"]):
            if (
                not isinstance(raw_chunk, dict)
                or not isinstance(raw_chunk.get("text"), str)
                or not raw_chunk["text"]
                or isinstance(raw_chunk.get("token"), bool)
                or not isinstance(raw_chunk.get("token"), int)
                or raw_chunk["token"] < 0
            ):
                raise ValueError(
                    f"plots[{plot_index}].chunks[{chunk_index}] is invalid."
                )
            chunks.append(raw_chunk)

        text = "".join(chunk["text"] for chunk in chunks)
        normalized, positions = _normalize_with_positions(text)
        plots.append(
            PlotText(
                plot_index=plot_index,
                volume=raw_plot.get("volume"),
                chapter=raw_plot.get("chapter"),
                text=text,
                normalized=normalized,
                normalized_to_original=positions,
                state_chunks=(),
                rebuilt_chunks=tuple(chunks),
            )
        )
        for source_chunk_start in range(0, len(chunks), chunks_per_input):
            source_chunk_end = min(
                source_chunk_start + chunks_per_input,
                len(chunks),
            ) - 1
            selected = chunks[source_chunk_start : source_chunk_end + 1]
            inputs[(plot_index, source_chunk_start)] = InputSpec(
                input_index=input_index,
                plot_index=plot_index,
                source_chunk_start=source_chunk_start,
                source_chunk_end=source_chunk_end,
                source_token_count=sum(chunk["token"] for chunk in selected),
                volume=raw_plot.get("volume"),
                chapter=raw_plot.get("chapter"),
                text="".join(chunk["text"] for chunk in selected),
            )
            input_index += 1
    return raw["title"], plots, inputs


def _chunk_starts(plot: PlotText) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for chunk in plot.rebuilt_chunks:
        starts.append(cursor)
        cursor += len(chunk["text"])
    return starts


def _infer_positions(
    positions: tuple[int | None, ...],
) -> tuple[int, ...]:
    resolved: list[int] = []
    for unit_index, position in enumerate(positions):
        if position is not None:
            resolved.append(position)
            continue
        following = next(
            (
                candidate
                for candidate in positions[unit_index + 1 :]
                if candidate is not None
            ),
            None,
        )
        if following is not None:
            resolved.append(following)
            continue
        preceding = next(
            (
                candidate
                for candidate in reversed(positions[:unit_index])
                if candidate is not None
            ),
            0,
        )
        resolved.append(preceding)
    return tuple(resolved)


def _record_utterance(unit: MessageUnit) -> dict[str, str]:
    if unit.environment:
        return {
            "speaker": "Environment",
            "content": f"*({unit.content})*",
        }
    return {
        "speaker": unit.character,
        "content": unit.content,
    }


def _map_sft_records(
    records: list[ParsedSftRecord],
    plots: list[PlotText],
    inputs: dict[tuple[int, int], InputSpec],
    chunks_per_input: int,
    min_coverage: float,
) -> tuple[dict[int, list[LocatedUnit]], IssueLog]:
    issues = IssueLog()
    by_input: dict[int, list[LocatedUnit]] = {}
    for record in records:
        plot = choose_plot(record, plots, issues, min_coverage)
        if plot is None:
            matches = sorted(
                (score_record(record, candidate) for candidate in plots),
                key=lambda item: (item.score, item.coverage),
                reverse=True,
            )
            best = matches[0]
            raise ValueError(
                f"SFT line {record.line_number} could not be mapped; "
                f"best plot={best.plot_index}, score={best.score}, "
                f"coverage={best.coverage:.3f}, anchors={best.matched_anchors}."
            )
        positions = _match_unit_positions(record, plot, issues)
        resolved = _infer_positions(positions)
        starts = _chunk_starts(plot)
        for unit_index, (unit, position) in enumerate(
            zip(record.units, resolved, strict=True)
        ):
            chunk_index = max(0, bisect.bisect_right(starts, position) - 1)
            source_chunk_start = (chunk_index // chunks_per_input) * chunks_per_input
            spec = inputs[(plot.plot_index, source_chunk_start)]
            by_input.setdefault(spec.input_index, []).append(
                LocatedUnit(
                    position=position,
                    sft_line=record.line_number,
                    unit_index=unit_index,
                    unit=unit,
                )
            )
    return by_input, issues


def _migrated_rows(
    title: str,
    located: dict[int, list[LocatedUnit]],
    inputs: dict[tuple[int, int], InputSpec],
) -> dict[int, dict[str, Any]]:
    specs_by_index = {spec.input_index: spec for spec in inputs.values()}
    rows: dict[int, dict[str, Any]] = {}
    for input_index, units in located.items():
        spec = specs_by_index[input_index]
        ordered = sorted(
            units,
            key=lambda item: (item.position, item.sft_line, item.unit_index),
        )
        utterances: list[dict[str, str]] = []
        seen: set[tuple[int, str, str]] = set()
        for item in ordered:
            utterance = _record_utterance(item.unit)
            key = (
                item.position,
                utterance["speaker"],
                normalize_text(utterance["content"]),
            )
            if key in seen:
                continue
            seen.add(key)
            utterances.append(utterance)
        rows[input_index] = {
            "model": "sft-v2-migration",
            "source_title": title,
            "input_index": input_index,
            "plot_index": spec.plot_index,
            "source_chunk_start": spec.source_chunk_start,
            "source_chunk_end": spec.source_chunk_end,
            "source_token_count": spec.source_token_count,
            "volume": spec.volume,
            "chapter": spec.chapter,
            "result": {
                "characters": [],
                "utterances": utterances,
            },
        }
    return rows


def _load_existing(path: Path, title: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        input_index = row.get("input_index") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("source_title") != title
            or isinstance(input_index, bool)
            or not isinstance(input_index, int)
            or input_index < 0
            or input_index in rows
        ):
            raise ValueError(f"Existing conversation line {line_number} is invalid.")
        rows[input_index] = row
    return rows


def merge_sft_conversations(
    rebuilt_path: Path,
    sft_path: Path,
    conversation_path: Path,
    backup_path: Path,
    *,
    target_character: str,
    chunks_per_input: int,
    min_coverage: float,
) -> dict[str, Any]:
    title, plots, inputs = _load_rebuilt(rebuilt_path, chunks_per_input)
    records = load_sft_records(sft_path, target_character, "Environment")
    located, issues = _map_sft_records(
        records,
        plots,
        inputs,
        chunks_per_input,
        min_coverage,
    )
    migrated = _migrated_rows(title, located, inputs)
    existing = _load_existing(conversation_path, title)
    overlap = sorted(set(existing) & set(migrated))
    if overlap:
        raise ValueError(
            "Existing extracted records overlap migrated SFT input indexes: "
            + ", ".join(map(str, overlap))
        )

    combined = {**existing, **migrated}
    conversation_path.parent.mkdir(parents=True, exist_ok=True)
    if conversation_path.exists():
        if backup_path.exists():
            raise ValueError(f"Backup already exists: {backup_path}")
        shutil.copy2(conversation_path, backup_path)
    content = "".join(
        f"{json.dumps(combined[index], ensure_ascii=False)}\n"
        for index in sorted(combined)
    )
    conversation_path.write_text(content, encoding="utf-8")
    return {
        "title": title,
        "sft_records": len(records),
        "existing_records": len(existing),
        "migrated_input_records": len(migrated),
        "combined_records": len(combined),
        "input_indexes": sorted(combined),
        "issues": len(issues.entries),
        "backup": str(backup_path) if existing else None,
        "output": str(conversation_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SFT dialogue messages into conversation extraction JSONL."
    )
    parser.add_argument("rebuilt_plots", type=Path)
    parser.add_argument("sft_jsonl", type=Path)
    parser.add_argument("conversation_jsonl", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--character", default="逢坂大河")
    parser.add_argument("--chunks-per-input", type=int, default=4)
    parser.add_argument("--min-match-coverage", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunks_per_input < 1:
        raise SystemExit("--chunks-per-input must be positive.")
    if not 0.0 <= args.min_match_coverage <= 1.0:
        raise SystemExit("--min-match-coverage must be between 0 and 1.")
    backup = args.backup or args.conversation_jsonl.with_name(
        f"{args.conversation_jsonl.stem}.extracted.jsonl"
    )
    try:
        summary = merge_sft_conversations(
            args.rebuilt_plots,
            args.sft_jsonl,
            args.conversation_jsonl,
            backup,
            target_character=args.character,
            chunks_per_input=args.chunks_per_input,
            min_coverage=args.min_match_coverage,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
