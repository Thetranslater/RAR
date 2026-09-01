#!/usr/bin/env python3
"""Build canonical RLFF episodes for one book.

This is an offline data-preparation boundary.  It joins one rebuilt-plot JSON
file, the matching task-extraction JSONL file, and that book's profile files
into ``rlff.episode.v1`` JSONL.  Rollout history deliberately starts empty.

Character identity is conservative: aliases are merged from every plot in the
book, but a merged group becomes rollout-eligible only when one profile file
stem exactly matches one of its aliases.  That profile stem is the character's
formal name.  Source plot/task/profile files are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlff.contracts import CharacterSpec, EpisodeRecord  # noqa: E402
from rlff.episodes import load_episode_jsonl  # noqa: E402
from script.build.build_role_sft_dataset import (  # noqa: E402
    CharacterNameGroup,
    build_character_name_groups,
    load_json,
    load_jsonl,
    load_profiles,
    resolve_profile_dir,
)


@dataclass(frozen=True)
class ProfileCharacter:
    """One profile-backed character resolved from a merged alias group."""

    formal_name: str
    profile: str
    aliases: tuple[str, ...]
    group_key: str


@dataclass(frozen=True)
class TaskContext:
    """Plot summary and profile-backed private tasks for one source plot."""

    plot: str
    tasks_by_group: dict[str, tuple[str, ...]]
    model: str
    source_metadata: dict[str, Any]
    ignored_task_characters: tuple[str, ...]


def _unique_strings(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _alias_groups(
    groups: list[CharacterNameGroup],
) -> tuple[dict[str, str], dict[str, CharacterNameGroup]]:
    alias_to_group: dict[str, str] = {}
    groups_by_key: dict[str, CharacterNameGroup] = {}
    for group in groups:
        groups_by_key[group.canonical] = group
        for alias in group.names:
            previous = alias_to_group.get(alias)
            if previous is not None and previous != group.canonical:
                raise ValueError(
                    f"Character alias {alias!r} belongs to multiple merged name groups"
                )
            alias_to_group[alias] = group.canonical
    return alias_to_group, groups_by_key


def resolve_profile_characters(
    groups: list[CharacterNameGroup],
    profiles: dict[str, str],
) -> tuple[dict[str, ProfileCharacter], tuple[str, ...]]:
    """Resolve formal names strictly through exact profile-stem matches."""

    resolved: dict[str, ProfileCharacter] = {}
    matched_profile_names: set[str] = set()
    for group in groups:
        matches = [profile_name for profile_name in profiles if profile_name in group.names]
        if not matches:
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Merged character aliases {group.names!r} match multiple profile files: "
                f"{matches!r}"
            )
        formal_name = matches[0]
        matched_profile_names.add(formal_name)
        resolved[group.canonical] = ProfileCharacter(
            formal_name=formal_name,
            profile=profiles[formal_name],
            aliases=tuple(alias for alias in group.names if alias != formal_name),
            group_key=group.canonical,
        )
    unmatched_profiles = tuple(sorted(set(profiles) - matched_profile_names))
    return resolved, unmatched_profiles


def build_task_contexts(
    task_records: list[dict[str, Any]],
    plots: list[dict[str, Any]],
    alias_to_group: dict[str, str],
    eligible_groups: set[str],
) -> dict[int, TaskContext]:
    """Index task records by plot and retain only profile-backed role tasks."""

    contexts: dict[int, TaskContext] = {}
    for line_number, record in enumerate(task_records, start=1):
        plot_index = record.get("plot_index")
        if not isinstance(plot_index, int) or not 0 <= plot_index < len(plots):
            raise ValueError(f"tasks line {line_number} has an invalid plot_index")
        if plot_index in contexts:
            raise ValueError(f"tasks contains duplicate plot_index {plot_index}")

        result = record.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"tasks line {line_number} has an invalid result")
        plot_summary = result.get("plot")
        if not isinstance(plot_summary, str) or not plot_summary.strip():
            raise ValueError(f"tasks line {line_number} has an invalid plot summary")
        raw_tasks = result.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError(f"tasks line {line_number} has an invalid tasks list")

        tasks_by_group: dict[str, list[str]] = {}
        ignored_names: list[str] = []
        for task_index, raw_task in enumerate(raw_tasks):
            if not isinstance(raw_task, dict):
                raise ValueError(
                    f"tasks line {line_number} task {task_index} must be an object"
                )
            raw_name = raw_task.get("name")
            raw_values = raw_task.get("values")
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(
                    f"tasks line {line_number} task {task_index} has an invalid name"
                )
            if not isinstance(raw_values, list) or not all(
                isinstance(value, str) and value.strip() for value in raw_values
            ):
                raise ValueError(
                    f"tasks line {line_number} task {task_index} has invalid values"
                )
            name = raw_name.strip()
            group_key = alias_to_group.get(name)
            if group_key is None or group_key not in eligible_groups:
                ignored_names.append(name)
                continue
            tasks_by_group.setdefault(group_key, []).extend(raw_values)

        source_metadata: dict[str, Any] = {}
        for key in (
                "model",
                "source_chunk_start",
                "source_chunk_end",
                "source_chunk_count",
                "source_token_count",
        ):
            if key in record:
                source_metadata[key] = record[key]
        raw_model = record.get("model", "")
        contexts[plot_index] = TaskContext(
            plot=plot_summary.strip(),
            tasks_by_group={
                group_key: _unique_strings(values)
                for group_key, values in tasks_by_group.items()
            },
            model=raw_model if isinstance(raw_model, str) else "",
            source_metadata=source_metadata,
            ignored_task_characters=tuple(dict.fromkeys(ignored_names)),
        )
    return contexts


def _plot_character_groups(
    plot: dict[str, Any],
    *,
    plot_index: int,
    alias_to_group: dict[str, str],
) -> tuple[str, ...]:
    raw_characters = plot.get("characters")
    if not isinstance(raw_characters, list):
        raise ValueError(f"plot {plot_index} contains an invalid characters list")

    ordered_groups: list[str] = []
    for character_index, raw_character in enumerate(raw_characters):
        if not isinstance(raw_character, dict):
            raise ValueError(
                f"plot {plot_index} character {character_index} must be an object"
            )
        raw_names = raw_character.get("name")
        if not isinstance(raw_names, list) or not raw_names or not all(
            isinstance(name, str) and name.strip() for name in raw_names
        ):
            raise ValueError(
                f"plot {plot_index} character {character_index} has invalid names"
            )
        matched_groups = {
            alias_to_group[name.strip()]
            for name in raw_names
            if name.strip() in alias_to_group
        }
        # A polluted local name array may bridge multiple independently merged
        # groups.  It is ignored rather than guessing a character identity.
        if len(matched_groups) != 1:
            continue
        group_key = next(iter(matched_groups))
        if group_key not in ordered_groups:
            ordered_groups.append(group_key)
    return tuple(ordered_groups)


def build_episode_records(
    *,
    plot_document: dict[str, Any],
    task_records: list[dict[str, Any]],
    profiles: dict[str, str],
) -> tuple[list[EpisodeRecord], dict[str, Any]]:
    """Build and validate one canonical episode per task-covered plot."""

    plots = plot_document.get("plots")
    if not isinstance(plots, list):
        raise ValueError("The rebuilt plot document must contain a plots array")
    if not plots:
        raise ValueError("The rebuilt plot document contains no plots")
    if not profiles:
        raise ValueError("No non-empty profile files were found for this book")

    groups = build_character_name_groups(plots)
    alias_to_group, _ = _alias_groups(groups)
    profile_characters, unmatched_profiles = resolve_profile_characters(groups, profiles)
    if not profile_characters:
        raise ValueError("No profile filename exactly matches any merged character name array")
    task_contexts = build_task_contexts(
        task_records,
        plots,
        alias_to_group,
        set(profile_characters),
    )

    raw_title = plot_document.get("title", "")
    title = raw_title.strip() if isinstance(raw_title, str) else ""
    records: list[EpisodeRecord] = []
    skipped_without_tasks: list[int] = []
    skipped_without_profiles: list[int] = []
    taskless_profile_characters: dict[int, list[str]] = {}
    ignored_task_characters: dict[int, list[str]] = {}

    for plot_index, plot in enumerate(plots):
        context = task_contexts.get(plot_index)
        if context is None:
            skipped_without_tasks.append(plot_index)
            continue
        if not isinstance(plot, dict):
            raise ValueError(f"plot {plot_index} must be an object")

        local_groups = _plot_character_groups(
            plot,
            plot_index=plot_index,
            alias_to_group=alias_to_group,
        )
        characters: list[CharacterSpec] = []
        for group_key in local_groups:
            resolved = profile_characters.get(group_key)
            if resolved is None:
                continue
            characters.append(
                CharacterSpec(
                    name=resolved.formal_name,
                    profile=resolved.profile,
                    private_tasks=context.tasks_by_group.get(group_key, ()),
                    aliases=resolved.aliases,
                )
            )
        if not characters:
            skipped_without_profiles.append(plot_index)
            continue

        no_task_names = [character.name for character in characters if not character.private_tasks]
        if no_task_names:
            taskless_profile_characters[plot_index] = no_task_names
        if context.ignored_task_characters:
            ignored_task_characters[plot_index] = list(context.ignored_task_characters)

        metadata: dict[str, Any] = {
            "book": title,
            "plot_index": plot_index,
            "volume": plot.get("volume", ""),
            "chapter": plot.get("chapter", ""),
            "task_model": context.model,
            "source": context.source_metadata,
        }
        records.append(
            EpisodeRecord(
                title=title,
                plot=context.plot,
                shared_tasks=(),
                characters=tuple(characters),
                dialogue=(),
                metadata=metadata,
            )
        )

    manifest = {
        "schema_version": "rlff.episode-dataset.v1",
        "title": title,
        "source_plot_count": len(plots),
        "source_task_count": len(task_records),
        "episode_count": len(records),
        "profile_count": len(profiles),
        "matched_profile_count": len(profile_characters),
        "matched_characters": sorted(
            character.formal_name for character in profile_characters.values()
        ),
        "unmatched_profiles": list(unmatched_profiles),
        "skipped_plot_indices_without_tasks": skipped_without_tasks,
        "skipped_plot_indices_without_profile_characters": skipped_without_profiles,
        "profile_characters_without_private_tasks": taskless_profile_characters,
        "ignored_task_characters_without_profiles": ignored_task_characters,
        "dialogue_policy": "empty",
        "formal_name_policy": "exact_profile_filename_match",
    }
    return records, manifest


def write_episode_jsonl(path: Path, records: list[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )


def default_output_path(rebuilt_plots: Path) -> Path:
    return PROJECT_ROOT / "dataset" / "episodes" / rebuilt_plots.stem / "episodes.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rlff.episode.v1 JSONL for one book.",
    )
    parser.add_argument("tasks_jsonl", type=Path, help="One book's extracted tasks JSONL")
    parser.add_argument("rebuilt_plots", type=Path, help="The same book's rebuilt plot JSON")
    parser.add_argument(
        "output_jsonl",
        type=Path,
        nargs="?",
        help=(
            "Output episode JSONL. Defaults to "
            "dataset/episodes/<rebuilt-plot-stem>/episodes.jsonl"
        ),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="The matching book profile directory; inferred from the plot title by default",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing episode JSONL and manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_jsonl or default_output_path(args.rebuilt_plots)
    manifest_path = output_path.with_suffix(".manifest.json")
    for path in (output_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists (use --overwrite): {path}")

    plot_document = load_json(args.rebuilt_plots)
    if not isinstance(plot_document, dict):
        raise ValueError("The rebuilt plot document must be a JSON object")
    profile_dir = args.profile_dir or resolve_profile_dir(plot_document, args.rebuilt_plots)
    records, manifest = build_episode_records(
        plot_document=plot_document,
        task_records=load_jsonl(args.tasks_jsonl),
        profiles=load_profiles(profile_dir),
    )
    if not records:
        raise ValueError("No episodes were generated")

    write_episode_jsonl(output_path, records)
    # Reload through the production loader before publishing the manifest.
    loaded = load_episode_jsonl(output_path)
    manifest.update(
        {
            "output": str(output_path),
            "profile_dir": str(profile_dir),
            "dataset_fingerprint": loaded.fingerprint,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "manifest": str(manifest_path),
                "episodes": len(records),
                "characters": manifest["matched_characters"],
                "dataset_fingerprint": loaded.fingerprint,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
