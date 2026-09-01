#!/usr/bin/env python3
"""Build a reproducible, bounded episode JSONL subset for cloud experiments.

The source ``dataset/episodes/<book>/episodes.jsonl`` files are treated as
immutable JSONL inputs. Every book other than the two explicitly named large
sources is retained in full. If the requested target is smaller than the
combined input, only ``狼与香辛料`` and ``化物语`` may be sampled down. The
sampling seed and per-source keep counts are recorded in the console output.
Records are serialized through the current ``EpisodeRecord`` protocol so stale
derived fingerprints are repaired without changing semantic content.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlff.contracts import EpisodeRecord  # noqa: E402, I001


SPECIAL_SOURCE_MARKERS = {
    "狼与香辛料": "狼与香辛料",
    "化物语": "化物语",
}


@dataclass(frozen=True)
class SourceRecords:
    """Validated source lines and their source-book identifier."""

    source: str
    path: Path
    lines: tuple[str, ...]
    episode_ids: tuple[str, ...]
    normalization_fields: tuple[str, ...]


def _source_kind(book_name: str) -> str | None:
    """Return the special source kind, or ``None`` for an always-kept book."""

    for marker, kind in SPECIAL_SOURCE_MARKERS.items():
        if marker in book_name:
            return kind
    return None


def _read_source(path: Path) -> SourceRecords:
    """Read, normalize, and validate one source JSONL.

    ``EpisodeRecord`` is the protocol authority for canonical serialization.
    The source fingerprint is removed before validation so an old/stale
    fingerprint can be repaired, while all semantic fields still pass the
    normal strict contract validation.  The resulting fingerprint is then
    emitted from ``model_dump``.
    """

    lines: list[str] = []
    episode_ids: list[str] = []
    normalization_fields: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            raw_record = dict(record)
            raw_record.pop("fingerprint", None)
            try:
                canonical_record = EpisodeRecord.model_validate(raw_record).model_dump(
                    mode="json", exclude_none=True
                )
            except Exception as exc:
                raise ValueError(
                    f"{path}:{line_number} is not a valid rlff.episode.v1 record: {exc}"
                ) from exc
            normalization_fields.update(
                key
                for key in set(record) | set(canonical_record)
                if record.get(key) != canonical_record.get(key)
            )
            episode_id = canonical_record.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id.strip():
                raise ValueError(
                    f"{path}:{line_number} must contain a non-empty string episode_id"
                )
            lines.append(
                json.dumps(canonical_record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            episode_ids.append(episode_id)
    if not lines:
        raise ValueError(f"{path} contains no non-empty JSONL records")
    return SourceRecords(
        path.parent.name,
        path,
        tuple(lines),
        tuple(episode_ids),
        tuple(sorted(normalization_fields)),
    )


def load_sources(
    episodes_root: Path, *, exclude_paths: Iterable[Path] = ()
) -> list[SourceRecords]:
    """Load all book episode files in stable path order."""

    excluded = {path.resolve() for path in exclude_paths}
    paths = [
        path
        for path in sorted(episodes_root.glob("*/episodes.jsonl"))
        if path.resolve() not in excluded
    ]
    if not paths:
        raise ValueError(f"No */episodes.jsonl files found under {episodes_root}")
    sources = [_read_source(path) for path in paths]
    all_ids = [episode_id for source in sources for episode_id in source.episode_ids]
    duplicates = sorted(
        episode_id for episode_id, count in Counter(all_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate episode_id values found: {duplicates}")
    return sources


def _sample_indices(count: int, keep: int, rng: random.Random) -> list[int]:
    """Sample records while retaining original source order in the output."""

    return sorted(rng.sample(range(count), keep))


def _special_keep_counts(
    sources: Iterable[SourceRecords], target_size: int
) -> dict[str, int]:
    """Compute balanced deletion counts for the two allowed deletion sources."""

    special = {
        kind: source
        for source in sources
        if (kind := _source_kind(source.source)) is not None
    }
    missing = set(SPECIAL_SOURCE_MARKERS.values()) - set(special)
    if missing:
        raise ValueError(
            "Cannot reach the requested target under the deletion rule: "
            f"missing special source(s) {sorted(missing)}"
        )
    fixed_count = sum(
        len(source.lines)
        for source in sources
        if _source_kind(source.source) is None
    )
    special_total = sum(len(source.lines) for source in special.values())
    required_special = target_size - fixed_count
    if required_special < 0 or required_special > special_total:
        raise ValueError(
            "Cannot reach target-size by deleting only from 狼与香辛料 and 化物语: "
            f"fixed={fixed_count}, special={special_total}, target={target_size}"
        )
    delete_count = special_total - required_special
    if delete_count == 0:
        return {kind: len(source.lines) for kind, source in special.items()}
    if delete_count < 2:
        raise ValueError(
            "Both special sources must participate in deletion, but fewer than "
            "two records need to be deleted."
        )
    # Split deletions as evenly as possible.  This produces 21 deletions from
    # each source for the current 70+90 -> 118 special-record target.
    left = delete_count // 2
    right = delete_count - left
    counts = {"狼与香辛料": left, "化物语": right}
    for kind, deleted in counts.items():
        available = len(special[kind].lines)
        if deleted <= 0 or deleted >= available:
            raise ValueError(
                f"Deletion allocation for {kind} is invalid: delete={deleted}, "
                f"available={available}"
            )
    return {
        kind: len(special[kind].lines) - deleted
        for kind, deleted in counts.items()
    }


def build_subset(
    sources: list[SourceRecords], *, target_size: int, seed: int
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Select records and return lines, before counts, and after counts."""

    before = {source.source: len(source.lines) for source in sources}
    total = sum(before.values())
    if target_size <= 0:
        raise ValueError("target-size must be positive")
    if total < target_size:
        raise ValueError(
            f"Only {total} episodes are available, fewer than target-size={target_size}"
        )
    if total == target_size:
        return [line for source in sources for line in source.lines], before, before

    keep_special = _special_keep_counts(sources, target_size)
    rng = random.Random(seed)
    output: list[str] = []
    after: dict[str, int] = {}
    for source in sources:
        kind = _source_kind(source.source)
        if kind is None:
            selected = list(source.lines)
        else:
            selected = [source.lines[index] for index in _sample_indices(
                len(source.lines), keep_special[kind], rng
            )]
        output.extend(selected)
        after[source.source] = len(selected)
    return output, before, after


def write_and_validate(lines: list[str], output: Path, expected_size: int) -> None:
    """Write selected raw JSONL and validate count, JSON, and ID uniqueness."""

    if len(lines) != expected_size:
        raise ValueError(f"Internal selection count is {len(lines)}, expected {expected_size}")
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Selected output line {line_number} is invalid JSON") from exc
        episode_id = record.get("episode_id") if isinstance(record, dict) else None
        if not isinstance(episode_id, str) or episode_id in seen:
            raise ValueError(
                "Selected output has invalid or duplicate episode_id "
                f"at line {line_number}"
            )
        seen.add(episode_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8", newline="\n")
    if sum(1 for line in output.open("r", encoding="utf-8") if line.strip()) != expected_size:
        raise ValueError("Written output does not contain the expected JSONL count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=Path("dataset/episodes"),
        help="Root containing one <book>/episodes.jsonl per source book.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/episodes/rlff_200/episodes.jsonl"),
    )
    parser.add_argument("--target-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = load_sources(args.episodes_root, exclude_paths=(args.output,))
    lines, before, after = build_subset(
        sources, target_size=args.target_size, seed=args.seed
    )
    write_and_validate(lines, args.output, args.target_size)
    print(f"Wrote {len(lines)} episodes to {args.output}")
    print(f"seed={args.seed} target-size={args.target_size}")
    print("source counts (before -> after):")
    for source in sources:
        print(f"  {source.source}: {before[source.source]} -> {after[source.source]}")
        if source.normalization_fields:
            print(
                "    normalized fields: "
                + ", ".join(source.normalization_fields)
            )
    print("validation: JSONL parseable, episode_id values unique, exact target count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
