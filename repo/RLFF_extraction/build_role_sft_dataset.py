#!/usr/bin/env python3
"""Build per-character ShareGPT conversations with plot-specific tasks."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from script.system_prompt_renderer import render_system_prompt  # noqa: E402

ENVIRONMENT_CHARACTER = "Environment"
INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class Utterance:
    speaker: str
    content: str
    known_character: bool


@dataclass(frozen=True)
class SystemTemplate:
    name: str
    content: str


@dataclass(frozen=True)
class CharacterNameGroup:
    canonical: str
    names: tuple[str, ...]
    frequencies: tuple[tuple[str, int], ...]
    description: str


@dataclass(frozen=True)
class PlotTaskContext:
    plot: str
    tasks_by_character: dict[str, tuple[str, ...]]
    model: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_system_templates(paths: list[Path]) -> list[SystemTemplate]:
    if not paths:
        raise ValueError("At least one --system-template is required")
    templates: list[SystemTemplate] = []
    names: set[str] = set()
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ValueError(f"Cannot read system template {path}: {exc}") from exc
        if not content.strip():
            raise ValueError(f"System template is empty: {path}")
        name = path.stem
        if name in names:
            raise ValueError(f"Duplicate system template name: {name}")
        names.add(name)
        templates.append(SystemTemplate(name=name, content=content))
    return templates


def load_profiles(profile_dir: Path) -> dict[str, str]:
    if not profile_dir.is_dir():
        return {}
    profiles: dict[str, str] = {}
    for path in sorted(profile_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8-sig").strip()
        if not content:
            raise ValueError(f"Profile is empty: {path}")
        if path.stem in profiles:
            raise ValueError(f"Duplicate character profile: {path.stem}")
        profiles[path.stem] = content
    return profiles


def build_character_name_groups(
    plots: list[dict[str, Any]],
) -> list[CharacterNameGroup]:
    grouped_names: list[set[str]] = []
    grouped_descriptions: list[str] = []
    first_seen: dict[str, int] = {}
    frequencies: Counter[str] = Counter()

    for plot_index, plot in enumerate(plots):
        characters = plot.get("characters", [])
        if not isinstance(characters, list):
            raise ValueError(f"plot {plot_index} contains an invalid character list")
        for character in characters:
            if not isinstance(character, dict):
                raise ValueError(f"plot {plot_index} contains an invalid character")
            names = character.get("name")
            if not isinstance(names, list) or not names or not all(
                isinstance(name, str) and name for name in names
            ):
                raise ValueError(
                    f"plot {plot_index} contains an invalid character name list"
                )
            raw_description = character.get("description", "")
            if raw_description is None:
                raw_description = ""
            if not isinstance(raw_description, str):
                raise ValueError(
                    f"plot {plot_index} contains an invalid character description"
                )
            description = raw_description.strip()
            unique_names = list(dict.fromkeys(names))
            incoming_names = set(unique_names)
            overlapping_groups = [
                group_index
                for group_index, existing_names in enumerate(grouped_names)
                if incoming_names & existing_names
            ]
            if len(overlapping_groups) > 1:
                continue
            if overlapping_groups:
                group_index = overlapping_groups[0]
            else:
                group_index = len(grouped_names)
                grouped_names.append(set())
                grouped_descriptions.append("")

            for name in unique_names:
                if name not in first_seen:
                    first_seen[name] = len(first_seen)
                grouped_names[group_index].add(name)
                frequencies[name] += 1
            if len(description) > len(grouped_descriptions[group_index]):
                grouped_descriptions[group_index] = description

    groups: list[CharacterNameGroup] = []
    for names, description in zip(
        grouped_names,
        grouped_descriptions,
        strict=True,
    ):
        ordered_names = sorted(
            names,
            key=lambda name: (-frequencies[name], first_seen[name]),
        )
        groups.append(
            CharacterNameGroup(
                canonical=ordered_names[0],
                names=tuple(ordered_names),
                frequencies=tuple(
                    (name, frequencies[name]) for name in ordered_names
                ),
                description=description,
            )
        )
    return groups


def character_aliases(groups: list[CharacterNameGroup]) -> dict[str, str]:
    return {
        name: group.canonical
        for group in groups
        for name in group.names
    }


def build_plot_task_contexts(
    records: list[dict[str, Any]],
    plots: list[dict[str, Any]],
    aliases: dict[str, str],
) -> tuple[dict[int, PlotTaskContext], set[str]]:
    contexts: dict[int, PlotTaskContext] = {}
    unknown_characters: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        plot_index = record.get("plot_index")
        if not isinstance(plot_index, int) or not 0 <= plot_index < len(plots):
            raise ValueError(f"tasks line {line_number} has an invalid plot_index")
        if plot_index in contexts:
            raise ValueError(f"tasks contains duplicate plot_index {plot_index}")

        result = record.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"tasks line {line_number} has an invalid result")
        plot_text = result.get("plot")
        if not isinstance(plot_text, str) or not plot_text.strip():
            raise ValueError(f"tasks line {line_number} has an invalid plot text")
        task_groups = result.get("tasks")
        if not isinstance(task_groups, list):
            raise ValueError(f"tasks line {line_number} has an invalid tasks list")

        tasks_by_character: dict[str, list[str]] = defaultdict(list)
        for group_index, group in enumerate(task_groups):
            if not isinstance(group, dict):
                raise ValueError(
                    f"tasks line {line_number} group {group_index} must be an object"
                )
            name = group.get("name")
            values = group.get("values")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"tasks line {line_number} group {group_index} has an invalid name"
                )
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(
                    f"tasks line {line_number} group {group_index} has invalid values"
                )
            canonical = aliases.get(name.strip())
            if canonical is None:
                unknown_characters.add(name.strip())
                continue
            tasks_by_character[canonical].extend(value.strip() for value in values)

        raw_model = record.get("model", "")
        model = raw_model if isinstance(raw_model, str) else ""
        contexts[plot_index] = PlotTaskContext(
            plot=plot_text.strip(),
            tasks_by_character={
                character: tuple(values)
                for character, values in tasks_by_character.items()
            },
            model=model,
        )
    return contexts, unknown_characters


def match_group_profiles(
    groups: list[CharacterNameGroup],
    profiles: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    matched_profiles: dict[str, str] = {}
    profile_names: dict[str, str] = {}
    profile_sources: dict[str, str] = {}
    for group in groups:
        matched_names = [name for name in group.names if name in profiles]
        if not matched_names:
            matched_profiles[group.canonical] = group.description
            profile_names[group.canonical] = ""
            profile_sources[group.canonical] = (
                "character_description" if group.description else "empty"
            )
            continue
        unique_contents = {profiles[name] for name in matched_names}
        if len(unique_contents) > 1:
            raise ValueError(
                f"Character aliases {group.names!r} match multiple different profiles: "
                f"{matched_names!r}"
            )
        profile_name = matched_names[0]
        matched_profiles[group.canonical] = profiles[profile_name]
        profile_names[group.canonical] = profile_name
        profile_sources[group.canonical] = "profile_file"
    return matched_profiles, profile_names, profile_sources


def resolve_character_names(
    groups: list[CharacterNameGroup],
    profile_names: dict[str, str],
) -> dict[str, str]:
    return {
        group.canonical: profile_names.get(group.canonical) or group.canonical
        for group in groups
    }


def resolve_profile_dir(plot_document: dict[str, Any], rebuilt_plots: Path) -> Path:
    profile_root = Path(__file__).resolve().parents[3] / "dataset" / "profile"
    raw_names = [plot_document.get("title"), rebuilt_plots.stem]
    candidates: list[str] = []
    for raw_name in raw_names:
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        name = raw_name.strip()
        for candidate in (name, re.sub(r"^\d+\s*", "", name)):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        profile_dir = profile_root / candidate
        if profile_dir.is_dir():
            return profile_dir
    fallback = candidates[-1] if candidates else rebuilt_plots.stem
    return profile_root / fallback


def validate_record_range(
    record: dict[str, Any],
    plots: list[dict[str, Any]],
    line_number: int,
) -> None:
    plot_index = record.get("plot_index")
    start = record.get("source_chunk_start")
    end = record.get("source_chunk_end")
    if (
        not isinstance(plot_index, int)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise ValueError(f"conversation line {line_number} has invalid source indexes")
    if not 0 <= plot_index < len(plots):
        raise ValueError(f"conversation line {line_number} has an unknown plot_index")
    if not 0 <= start <= end < len(plots[plot_index]["chunks"]):
        raise ValueError(f"conversation line {line_number} has an invalid chunk range")


def group_records(
    records: list[dict[str, Any]],
    plots: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line_number, record in enumerate(records, start=1):
        validate_record_range(record, plots, line_number)
        grouped[record["plot_index"]].append(record)

    for plot_index, plot_records in grouped.items():
        plot_records.sort(
            key=lambda record: (
                record["source_chunk_start"],
                record["source_chunk_end"],
                record["input_index"],
            )
        )
        previous_end = -1
        for record in plot_records:
            if record["source_chunk_start"] <= previous_end:
                raise ValueError(f"plot {plot_index} contains overlapping conversation ranges")
            previous_end = record["source_chunk_end"]
    return grouped


def normalize_utterances(
    plot_index: int,
    plot_records: list[dict[str, Any]],
    aliases: dict[str, str],
) -> list[Utterance]:
    utterances: list[Utterance] = []
    for record in plot_records:
        result = record.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("utterances"), list):
            raise ValueError(f"plot {plot_index} contains an invalid extraction result")
        for utterance in result["utterances"]:
            speaker = utterance.get("speaker")
            content = utterance.get("content")
            if not isinstance(speaker, str) or not isinstance(content, str) or not content:
                raise ValueError(f"plot {plot_index} contains an invalid utterance")
            content = content.replace("\r", "").replace("\n", "")
            if not content:
                raise ValueError(
                    f"plot {plot_index} contains an utterance empty after newline cleanup"
                )
            canonical = aliases.get(speaker)
            utterances.append(
                Utterance(
                    speaker=canonical or speaker,
                    content=content,
                    known_character=canonical is not None,
                )
            )
    return utterances


def merge_consecutive_utterances(utterances: list[Utterance]) -> list[Utterance]:
    merged: list[Utterance] = []
    for utterance in utterances:
        if merged and merged[-1].speaker == utterance.speaker:
            previous = merged[-1]
            merged[-1] = Utterance(
                speaker=previous.speaker,
                content=previous.content + utterance.content,
                known_character=previous.known_character,
            )
        else:
            merged.append(utterance)
    return merged


def merge_messages(
    utterances: list[Utterance],
    target_character: str,
    resolved_names: dict[str, str],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for utterance in merge_consecutive_utterances(utterances):
        is_assistant = (
            utterance.known_character and utterance.speaker == target_character
        )
        role = "assistant" if is_assistant else "user"
        if is_assistant or not utterance.speaker:
            content = utterance.content
        else:
            content = (
                f"{resolved_names.get(utterance.speaker, utterance.speaker)}:"
                f"{utterance.content}"
            )
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += f"\n{content}"
        else:
            message: dict[str, Any] = {"role": role, "content": content}
            if is_assistant:
                message["loss"] = True
            merged.append(message)
    while merged and merged[-1]["role"] == "user":
        merged.pop()
    return merged


def safe_filename(character: str) -> str:
    filename = INVALID_FILENAME.sub("_", character).strip(" .")
    if not filename:
        raise ValueError(f"Character name {character!r} cannot form a file name")
    return filename


def covered_chunks(records: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return {
        (record["plot_index"], chunk_index)
        for record in records
        for chunk_index in range(
            record["source_chunk_start"],
            record["source_chunk_end"] + 1,
        )
    }


def build_rows(
    plots: list[dict[str, Any]],
    grouped_records: dict[int, list[dict[str, Any]]],
    task_contexts: dict[int, PlotTaskContext],
    aliases: dict[str, str],
    templates: list[SystemTemplate],
    profiles: dict[str, str],
    profile_names: dict[str, str],
    profile_sources: dict[str, str],
    resolved_names: dict[str, str],
    excluded_characters: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plot_index, plot in enumerate(plots):
        plot_records = grouped_records.get(plot_index, [])
        task_context = task_contexts.get(plot_index)
        if not plot_records or task_context is None:
            continue
        utterances = normalize_utterances(
            plot_index,
            plot_records,
            aliases,
        )
        target_characters = list(
            dict.fromkeys(
                utterance.speaker
                for utterance in utterances
                if utterance.known_character
                and utterance.speaker not in excluded_characters
            )
        )
        covered_plot_chunks = [
            chunk_index
            for record in plot_records
            for chunk_index in range(
                record["source_chunk_start"],
                record["source_chunk_end"] + 1,
            )
        ]
        plot_text = task_context.plot
        for character in target_characters:
            resolved_character = resolved_names.get(character, character)
            dialogue = merge_messages(utterances, character, resolved_names)
            profile = profiles.get(character, "")
            assistant_count = sum(
                message["role"] == "assistant" for message in dialogue
            )
            template = rng.choice(templates)
            character_tasks = task_context.tasks_by_character.get(character, ())
            tasks_text = ";".join(character_tasks)
            messages = [
                {
                    "role": "system",
                    "content": render_system_prompt(
                        template.content,
                        {
                            "character": resolved_character,
                            "profile": profile,
                            "plot": plot_text,
                            "tasks": tasks_text,
                        },
                    )
                    + "\n",
                },
                *dialogue,
            ]
            rows.append(
                {
                    "details": {
                        "plot": plot_index,
                        "character": resolved_character,
                        "volume": plot.get("volume", ""),
                        "chapter": plot.get("chapter", ""),
                        "source_chunk_ranges": [
                            [
                                record["source_chunk_start"],
                                record["source_chunk_end"],
                            ]
                            for record in plot_records
                        ],
                        "source_input_indices": [
                            record["input_index"] for record in plot_records
                        ],
                        "source_models": list(
                            dict.fromkeys(record["model"] for record in plot_records)
                        ),
                        "source_plot_chunk_count": len(plot["chunks"]),
                        "covered_source_chunks": covered_plot_chunks,
                        "source_plot_complete": len(covered_plot_chunks)
                        == len(plot["chunks"]),
                        "source_utterance_count": len(utterances),
                        "dialogue_message_count": len(dialogue),
                        "assistant_message_count": assistant_count,
                        "starts_with_assistant": bool(
                            dialogue and dialogue[0]["role"] == "assistant"
                        ),
                        "ends_with_assistant": bool(
                            dialogue and dialogue[-1]["role"] == "assistant"
                        ),
                        "profile": "available" if profile else "empty",
                        "profile_name": profile_names.get(character, ""),
                        "profile_source": profile_sources.get(character, "empty"),
                        "system_prompt_variant": template.name,
                        "plot_context_source": "tasks",
                        "task_count": len(character_tasks),
                        "task_model": task_context.model,
                    },
                    "data": {"messages": messages},
                },
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build plot-level, per-character ShareGPT datasets.",
    )
    parser.add_argument("conversation_jsonl", type=Path)
    parser.add_argument("rebuilt_plots", type=Path)
    parser.add_argument("tasks_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--system-template",
        action="append",
        type=Path,
        required=True,
        help="System prompt template; repeat for each equally weighted version.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed controlling template selection, random blocks, and rearrange blocks.",
    )
    parser.add_argument(
        "--exclude-character",
        action="append",
        default=[],
        help="Character omitted as an assistant target; repeat as needed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files in an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.conversation_jsonl)
    task_records = load_jsonl(args.tasks_jsonl)
    plot_document = load_json(args.rebuilt_plots)
    plots = plot_document.get("plots")
    if not isinstance(plots, list):
        raise ValueError("The rebuilt plot document must contain a plots array")
    templates = load_system_templates(args.system_template)
    profile_dir = resolve_profile_dir(plot_document, args.rebuilt_plots)
    name_groups = build_character_name_groups(plots)
    aliases = character_aliases(name_groups)
    task_contexts, unknown_task_characters = build_plot_task_contexts(
        task_records,
        plots,
        aliases,
    )
    profiles, profile_names, profile_sources = match_group_profiles(
        name_groups,
        load_profiles(profile_dir),
    )
    resolved_names = resolve_character_names(name_groups, profile_names)
    excluded_characters = {
        aliases.get(character, character)
        for character in args.exclude_character
    }

    grouped_records = group_records(records, plots)
    random.seed(args.seed)
    rows = build_rows(
        plots,
        grouped_records,
        task_contexts,
        aliases,
        templates,
        profiles,
        profile_names,
        profile_sources,
        resolved_names,
        excluded_characters,
        random.Random(args.seed),
    )
    if not rows:
        raise ValueError("No per-character records were generated")

    if (
        not args.overwrite
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    character_dir = args.output_dir / "characters"
    character_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale_path in character_dir.glob("*.jsonl"):
            stale_path.unlink()

    rows_by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_character[row["details"]["character"]].append(row)

    write_jsonl(args.output_dir / "all.jsonl", rows)
    character_files: dict[str, str] = {}
    for character, character_rows in sorted(rows_by_character.items()):
        output_path = character_dir / f"{safe_filename(character)}.jsonl"
        write_jsonl(output_path, character_rows)
        character_files[character] = output_path.relative_to(args.output_dir).as_posix()

    all_chunks = {
        (plot_index, chunk_index)
        for plot_index, plot in enumerate(plots)
        for chunk_index in range(len(plot["chunks"]))
    }
    covered = covered_chunks(records)
    manifest = {
        "title": plot_document.get("title", ""),
        "format": "nested_sharegpt_messages",
        "system_prompt_variants": [template.name for template in templates],
        "variants_per_dialogue": 1,
        "seed": args.seed,
        "profile_dir": str(profile_dir),
        "character_name_groups": [
            {
                "canonical": group.canonical,
                "names": [
                    {"name": name, "frequency": frequency}
                    for name, frequency in group.frequencies
                ],
                "resolved_name": resolved_names[group.canonical],
                "profile_name": profile_names.get(group.canonical, ""),
                "profile_source": profile_sources.get(group.canonical, "empty"),
            }
            for group in name_groups
        ],
        "excluded_characters": sorted(excluded_characters),
        "environment_character": ENVIRONMENT_CHARACTER,
        "record_count": len(rows),
        "character_count": len(rows_by_character),
        "character_record_counts": {
            character: len(character_rows)
            for character, character_rows in sorted(rows_by_character.items())
        },
        "character_files": character_files,
        "source_conversation_records": len(records),
        "source_task_records": len(task_records),
        "task_covered_plot_count": len(task_contexts),
        "task_missing_conversation_plots": sorted(
            set(grouped_records) - set(task_contexts)
        ),
        "unknown_task_characters": sorted(unknown_task_characters),
        "source_plot_count": len(plots),
        "covered_plot_count": len(grouped_records),
        "source_chunk_count": len(all_chunks),
        "covered_chunk_count": len(covered),
        "chunk_coverage": len(covered) / len(all_chunks) if all_chunks else 1.0,
        "missing_chunks": [
            {"plot": plot_index, "chunk": chunk_index}
            for plot_index, chunk_index in sorted(all_chunks - covered)
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "records": len(rows),
                "characters": len(rows_by_character),
                "covered_chunks": len(covered),
                "total_chunks": len(all_chunks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
