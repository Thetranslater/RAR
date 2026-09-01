"""Build an RLFF dialogue dataset from rebuilt plots, states, and conversations."""

from __future__ import annotations

import argparse
import bisect
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

StateValue = str | int | float | bool | None
_DIALOGUE_RE = re.compile(r"^([^：:\n]{1,40})[：:](.*)$")
_NARRATION_RE = re.compile(r"^\*\((.*)\)\*$")
_STRONG_ANCHOR_LENGTH = 4
_MIN_MATCHED_CHARS = 20
_MIN_MATCHED_ANCHORS = 2
_DEFAULT_MIN_COVERAGE = 0.45


@dataclass(frozen=True)
class StateNode:
    name: str
    before: StateValue
    after: StateValue
    type: str
    source: str
    description: str


@dataclass(frozen=True)
class LocatedNode:
    node: StateNode
    position: int
    chunk_index: int
    node_index: int


@dataclass(frozen=True)
class ValidationState:
    value: StateValue
    description: str


@dataclass(frozen=True)
class MessageUnit:
    character: str
    content: str
    environment: bool
    anchors: tuple[str, ...]
    message_index: int
    source: str | None = None


@dataclass(frozen=True)
class ParsedSftRecord:
    line_number: int
    units: tuple[MessageUnit, ...]


@dataclass(frozen=True)
class PlotText:
    plot_index: int
    volume: str | None
    chapter: str | None
    text: str
    normalized: str
    normalized_to_original: tuple[int, ...]
    state_chunks: tuple[dict[str, Any], ...]
    rebuilt_chunks: tuple[dict[str, Any], ...]
    chunk_starts: tuple[int, ...]
    chunk_ends: tuple[int, ...]


@dataclass(frozen=True)
class RecordMatch:
    plot_index: int
    score: int
    coverage: float
    matched_anchors: int


@dataclass(frozen=True)
class PositionedRecord:
    record: ParsedSftRecord
    plot_index: int
    unit_positions: tuple[int | None, ...]
    fallback_start: int = 0

    @property
    def start(self) -> int:
        return min(
            (value for value in self.unit_positions if value is not None),
            default=self.fallback_start,
        )


class IssueLog:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def add(self, code: str, message: str, **context: Any) -> None:
        self.entries.append(
            {
                "level": "warning",
                "code": code,
                "message": message,
                **{key: value for key, value in context.items() if value is not None},
            }
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            f"{json.dumps(entry, ensure_ascii=False)}\n" for entry in self.entries
        )
        path.write_text(content, encoding="utf-8")


class StateTimeline:
    def __init__(
        self,
        events: list[LocatedNode],
        issues: IssueLog,
        plot_index: int,
    ) -> None:
        current: dict[str, ValidationState] = {}
        self.positions = [-1]
        self.versions = [dict(current)]
        event_offset = 0
        while event_offset < len(events):
            position = events[event_offset].position
            while event_offset < len(events) and events[event_offset].position == position:
                located = events[event_offset]
                self._apply(current, located, issues, plot_index)
                event_offset += 1
            self.positions.append(position)
            self.versions.append(dict(current))

    @staticmethod
    def _apply(
        current: dict[str, ValidationState],
        located: LocatedNode,
        issues: IssueLog,
        plot_index: int,
    ) -> None:
        node = located.node
        existing = current.get(node.name)
        context = {
            "plot_index": plot_index,
            "chunk_index": located.chunk_index,
            "node_index": located.node_index,
            "state": node.name,
            "source": node.source,
        }
        if existing is None:
            if node.before is not None:
                issues.add(
                    "missing_current_state",
                    "State transition has a non-null before value before the state exists.",
                    expected_before=node.before,
                    **context,
                )
            description = node.description
        else:
            if node.before is not None and existing.value != node.before:
                issues.add(
                    "before_mismatch",
                    "State transition before value does not match the current state value.",
                    current_value=existing.value,
                    expected_before=node.before,
                    **context,
                )
            description = existing.description
            if existing.description != node.description:
                issues.add(
                    "description_conflict",
                    "State descriptions differ; the first description is retained.",
                    retained_description=existing.description,
                    ignored_description=node.description,
                    **context,
                )
        current[node.name] = ValidationState(node.after, description)

    def snapshot_at(self, position: int) -> list[dict[str, Any]]:
        version_index = bisect.bisect_right(self.positions, position) - 1
        return [
            {
                "name": name,
                "description": state.description,
                "value": state.value,
            }
            for name, state in self.versions[version_index].items()
        ]

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _state_value(value: Any) -> StateValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("State values must be JSON scalars or null.")


def _parse_node(
    raw: Any,
    issues: IssueLog,
    *,
    plot_index: int,
    chunk_index: int,
    node_index: int,
) -> StateNode | None:
    context = {
        "plot_index": plot_index,
        "chunk_index": chunk_index,
        "node_index": node_index,
    }
    required = {"name", "before", "after", "type", "source", "description"}
    try:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(f"Node must contain exactly {sorted(required)}.")
        if (
            not isinstance(raw["name"], str)
            or raw["name"].count(".") != 1
            or not all(raw["name"].split("."))
        ):
            raise ValueError("name must use the Entity.attribute format.")
        if raw["type"] not in {"initialization", "instant"}:
            raise ValueError("type must be initialization or instant.")
        if not isinstance(raw["source"], str) or not raw["source"]:
            raise ValueError("source must be a non-empty string.")
        if not isinstance(raw["description"], str) or not raw["description"]:
            raise ValueError("description must be a non-empty string.")
        before = _state_value(raw["before"])
        after = _state_value(raw["after"])
        if before == after:
            return None
    except ValueError as exc:
        issues.add("invalid_state_node", str(exc), **context)
        return None
    return StateNode(
        name=raw["name"],
        before=before,
        after=after,
        type=raw["type"],
        source=raw["source"],
        description=raw["description"],
    )


def _normalize_with_positions(text: str) -> tuple[str, tuple[int, ...]]:
    normalized: list[str] = []
    positions: list[int] = []
    for original_index, character in enumerate(text):
        for expanded in unicodedata.normalize("NFKC", character):
            if expanded.isalnum():
                normalized.append(expanded.casefold())
                positions.append(original_index)
    return "".join(normalized), tuple(positions)


def normalize_text(text: str) -> str:
    return _normalize_with_positions(text)[0]


def _all_occurrences(text: str, needle: str) -> list[int]:
    starts: list[int] = []
    offset = 0
    while True:
        start = text.find(needle, offset)
        if start < 0:
            return starts
        starts.append(start)
        offset = start + 1


def _non_negative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _load_state_chunks(
    path: Path,
    expected_title: str,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, int]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("result"), dict):
            raise ValueError(f"Invalid state record at line {line_number}.")
        if raw.get("source_title") != expected_title:
            raise ValueError(
                f"State record line {line_number} belongs to "
                f"{raw.get('source_title')!r}, expected {expected_title!r}."
            )
        plot_index = _non_negative_integer(
            raw.get("plot_index"),
            f"line {line_number}.plot_index",
        )
        chunk_index = _non_negative_integer(
            raw.get("chunk_index"),
            f"line {line_number}.chunk_index",
        )
        key = (plot_index, chunk_index)
        if key in seen:
            raise ValueError(
                f"Duplicate state record for plot {plot_index}, chunk {chunk_index}."
            )
        seen.add(key)
        nodes = raw["result"].get("nodes")
        if not isinstance(nodes, list):
            raise ValueError(f"line {line_number}.result.nodes must be an array.")
        grouped.setdefault(plot_index, []).append(
            {
                "chunk_index": chunk_index,
                "volume": raw.get("volume"),
                "chapter": raw.get("chapter"),
                "nodes": nodes,
            }
        )
    if not grouped:
        raise ValueError("State result JSONL contains no records.")
    return grouped


def load_plot_texts(
    rebuilt_path: Path,
    states_path: Path,
    issues: IssueLog,
) -> tuple[str, list[PlotText]]:
    rebuilt = _load_json(rebuilt_path)
    if (
        not isinstance(rebuilt, dict)
        or not isinstance(rebuilt.get("title"), str)
        or not isinstance(rebuilt.get("plots"), list)
    ):
        raise ValueError("Rebuilt plot JSON must contain title and plots.")
    grouped_states = _load_state_chunks(states_path, rebuilt["title"])

    output: list[PlotText] = []
    for plot_index in sorted(grouped_states):
        if plot_index >= len(rebuilt["plots"]):
            raise ValueError(f"Unknown plot_index {plot_index}.")
        rebuilt_plot = rebuilt["plots"][plot_index]
        rebuilt_chunks = rebuilt_plot.get("chunks")
        if not isinstance(rebuilt_chunks, list):
            raise ValueError(f"Rebuilt plot {plot_index} has no chunks array.")
        state_chunks = sorted(
            grouped_states[plot_index],
            key=lambda item: item["chunk_index"],
        )
        chunk_indexes = [chunk["chunk_index"] for chunk in state_chunks]
        if chunk_indexes != list(range(len(state_chunks))):
            raise ValueError(
                f"State plot {plot_index} chunks must be a contiguous prefix starting at zero."
            )
        selected_rebuilt: list[dict[str, Any]] = []
        for state_chunk in state_chunks:
            chunk_index = state_chunk["chunk_index"]
            if chunk_index >= len(rebuilt_chunks):
                raise ValueError(f"Unknown rebuilt chunk {plot_index}:{chunk_index}.")
            chunk = rebuilt_chunks[chunk_index]
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                raise ValueError(f"Rebuilt chunk {plot_index}:{chunk_index} has no text.")
            if (
                state_chunk["volume"] != rebuilt_plot.get("volume")
                or state_chunk["chapter"] != rebuilt_plot.get("chapter")
            ):
                raise ValueError(
                    f"Plot {plot_index}, chunk {chunk_index} state metadata does not "
                    "match the rebuilt plot."
                )
            selected_rebuilt.append(chunk)
        chunk_starts: list[int] = []
        chunk_ends: list[int] = []
        text_parts: list[str] = []
        text_length = 0
        for chunk in selected_rebuilt:
            chunk_starts.append(text_length)
            chunk_text = chunk["text"]
            text_parts.append(chunk_text)
            text_length += len(chunk_text)
            chunk_ends.append(text_length)
        text = "".join(text_parts)
        normalized, position_map = _normalize_with_positions(text)
        output.append(
            PlotText(
                plot_index=plot_index,
                volume=rebuilt_plot.get("volume"),
                chapter=rebuilt_plot.get("chapter"),
                text=text,
                normalized=normalized,
                normalized_to_original=position_map,
                state_chunks=tuple(state_chunks),
                rebuilt_chunks=tuple(selected_rebuilt),
                chunk_starts=tuple(chunk_starts),
                chunk_ends=tuple(chunk_ends),
            )
        )
        if len(state_chunks) < len(rebuilt_chunks):
            issues.add(
                "partial_plot_coverage",
                "Only the state-covered chunk prefix is used for this plot.",
                plot_index=plot_index,
                covered_chunks=len(state_chunks),
                total_chunks=len(rebuilt_chunks),
            )
    return rebuilt["title"], sorted(output, key=lambda item: item.plot_index)


def build_timeline(plot: PlotText, issues: IssueLog) -> StateTimeline:
    events: list[LocatedNode] = []
    for state_chunk in plot.state_chunks:
        chunk_index = state_chunk["chunk_index"]
        raw_nodes = state_chunk.get("nodes")
        if not isinstance(raw_nodes, list):
            issues.add(
                "invalid_state_chunk",
                "State chunk nodes must be an array.",
                plot_index=plot.plot_index,
                chunk_index=chunk_index,
            )
            continue
        chunk_text = plot.rebuilt_chunks[chunk_index]["text"]
        for node_index, raw_node in enumerate(raw_nodes):
            node = _parse_node(
                raw_node,
                issues,
                plot_index=plot.plot_index,
                chunk_index=chunk_index,
                node_index=node_index,
            )
            if node is None:
                continue

            context = {
                "plot_index": plot.plot_index,
                "chunk_index": chunk_index,
                "node_index": node_index,
                "state": node.name,
                "source": node.source,
            }
            if node.type == "initialization":
                if not _all_occurrences(chunk_text, node.source):
                    issues.add(
                        "source_not_found",
                        "Initialization source is not an exact substring of "
                        "its source chunk.",
                        **context,
                    )
                events.append(
                    LocatedNode(
                        node=node,
                        position=plot.chunk_starts[chunk_index],
                        chunk_index=chunk_index,
                        node_index=node_index,
                    )
                )
                continue

            occurrences = _all_occurrences(chunk_text, node.source)
            if not occurrences:
                issues.add(
                    "source_not_found",
                    "State source is not an exact substring of its source chunk.",
                    plot_index=plot.plot_index,
                    chunk_index=chunk_index,
                    node_index=node_index,
                    state=node.name,
                    source=node.source,
                )
                continue
            source_start = occurrences[0]
            if len(occurrences) > 1:
                issues.add(
                    "source_ambiguous",
                    "State source occurs more than once; the first occurrence is used.",
                    plot_index=plot.plot_index,
                    chunk_index=chunk_index,
                    node_index=node_index,
                    state=node.name,
                    source=node.source,
                    occurrences=len(occurrences),
                )
            located = LocatedNode(
                node=node,
                position=(
                    plot.chunk_starts[chunk_index]
                    + source_start
                    + len(node.source)
                ),
                chunk_index=chunk_index,
                node_index=node_index,
            )
            events.append(located)

    events.sort(
        key=lambda item: (
            item.position,
            0 if item.node.type == "initialization" else 1,
            item.chunk_index,
            item.node_index,
        )
    )
    return StateTimeline(events, issues, plot.plot_index)


def _parse_user_message(
    content: str,
    message_index: int,
    environment_character: str,
) -> list[MessageUnit]:
    units: list[MessageUnit] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        narration = _NARRATION_RE.fullmatch(stripped)
        if narration:
            units.append(
                MessageUnit(
                    character=environment_character,
                    content=narration.group(1).strip(),
                    environment=True,
                    anchors=(),
                    message_index=message_index,
                )
            )
            continue
        dialogue = _DIALOGUE_RE.fullmatch(stripped)
        if dialogue:
            character = dialogue.group(1).strip()
            if re.fullmatch(r"未知角色\d*", character):
                character = ""
            spoken = dialogue.group(2).strip()
            units.append(
                MessageUnit(
                    character=character,
                    content=spoken,
                    environment=False,
                    anchors=(spoken,),
                    message_index=message_index,
                )
            )
            continue
        units.append(
            MessageUnit(
                character=environment_character,
                content=stripped,
                environment=True,
                anchors=(),
                message_index=message_index,
            )
        )
    return units


def load_sft_records(
    path: Path,
    target_character: str,
    environment_character: str,
) -> list[ParsedSftRecord]:
    records: list[ParsedSftRecord] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
            raise ValueError(f"SFT line {line_number} must contain messages.")
        units: list[MessageUnit] = []
        for message_index, message in enumerate(raw["messages"]):
            if not isinstance(message, dict):
                raise ValueError(f"SFT line {line_number} message {message_index} is invalid.")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"SFT line {line_number} message {message_index} has no content."
                )
            if role == "system":
                continue
            if role == "assistant":
                anchors = tuple(part.strip() for part in content.splitlines() if part.strip())
                units.append(
                    MessageUnit(
                        character=target_character,
                        content=content.strip(),
                        environment=False,
                        anchors=anchors,
                        message_index=message_index,
                    )
                )
            elif role == "user":
                units.extend(
                    _parse_user_message(content, message_index, environment_character)
                )
            else:
                raise ValueError(
                    f"SFT line {line_number} message {message_index} has role {role!r}."
                )
        records.append(ParsedSftRecord(line_number, tuple(units)))
    return records


def load_conversation_records(
    path: Path,
    title: str,
    plots: list[PlotText],
    issues: IssueLog,
    environment_character: str,
) -> dict[int, list[PositionedRecord]]:
    """Load explicit plot-indexed records from conversation extraction JSONL."""

    plots_by_index = {plot.plot_index: plot for plot in plots}
    positioned: dict[int, list[PositionedRecord]] = {
        plot.plot_index: [] for plot in plots
    }
    seen_inputs: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"Conversation line {line_number} must be an object.")
        if raw.get("source_title") != title:
            raise ValueError(
                f"Conversation line {line_number} source_title does not match {title!r}."
            )
        input_index = raw.get("input_index")
        plot_index = raw.get("plot_index")
        if (
            isinstance(input_index, bool)
            or not isinstance(input_index, int)
            or input_index < 0
            or input_index in seen_inputs
        ):
            raise ValueError(
                f"Conversation line {line_number} has invalid or duplicate "
                f"input_index {input_index!r}."
            )
        if isinstance(plot_index, bool) or not isinstance(plot_index, int):
            raise ValueError(
                f"Conversation line {line_number} has invalid plot_index "
                f"{plot_index!r}."
            )
        seen_inputs.add(input_index)
        if plot_index not in plots_by_index:
            issues.add(
                "conversation_plot_not_covered",
                "Conversation record belongs to a plot without extracted states.",
                conversation_line=line_number,
                input_index=input_index,
                plot_index=plot_index,
            )
            continue

        result = raw.get("result")
        if not isinstance(result, dict) or not isinstance(
            result.get("utterances"), list
        ):
            raise ValueError(
                f"Conversation line {line_number} result must contain utterances."
            )
        units: list[MessageUnit] = []
        for utterance_index, utterance in enumerate(result["utterances"]):
            utterance_fields = set(utterance) if isinstance(utterance, dict) else set()
            if (
                not isinstance(utterance, dict)
                or utterance_fields
                not in ({"speaker", "content"}, {"speaker", "content", "source"})
                or not isinstance(utterance["speaker"], str)
                or not isinstance(utterance["content"], str)
                or not utterance["content"].strip()
                or (
                    "source" in utterance
                    and (
                        not isinstance(utterance["source"], str)
                        or not utterance["source"]
                    )
                )
            ):
                raise ValueError(
                    f"Conversation line {line_number} utterance "
                    f"{utterance_index} must contain speaker/content strings and an "
                    "optional non-empty source string."
                )
            speaker = utterance["speaker"]
            content = utterance["content"].strip()
            explicit_source = utterance.get("source")
            source = explicit_source or utterance["content"]
            environment = speaker == environment_character
            if environment:
                narration = _NARRATION_RE.fullmatch(content)
                if narration is None:
                    raise ValueError(
                        f"Conversation line {line_number} utterance "
                        f"{utterance_index} Environment content must use *(...)*."
                )
                content = narration.group(1).strip()
            anchors = () if environment else (source,)
            units.append(
                MessageUnit(
                    character=speaker,
                    content=content,
                    environment=environment,
                    anchors=anchors,
                    message_index=utterance_index,
                    source=source,
                )
            )

        record = ParsedSftRecord(line_number, tuple(units))
        plot = plots_by_index[plot_index]
        source_chunk_start = _non_negative_integer(
            raw.get("source_chunk_start"),
            f"conversation line {line_number}.source_chunk_start",
        )
        source_chunk_end = _non_negative_integer(
            raw.get("source_chunk_end"),
            f"conversation line {line_number}.source_chunk_end",
        )
        if source_chunk_end < source_chunk_start:
            raise ValueError(
                f"Conversation line {line_number} source chunk range is reversed."
            )
        if source_chunk_end >= len(plot.rebuilt_chunks):
            issues.add(
                "conversation_chunk_range_not_covered",
                "Conversation record extends beyond the state-covered chunk prefix.",
                conversation_line=line_number,
                input_index=input_index,
                plot_index=plot_index,
                source_chunk_start=source_chunk_start,
                source_chunk_end=source_chunk_end,
                covered_chunks=len(plot.rebuilt_chunks),
            )
            continue
        lower_bound = plot.chunk_starts[source_chunk_start]
        upper_bound = plot.chunk_ends[source_chunk_end]
        positioned[plot_index].append(
            PositionedRecord(
                record=record,
                plot_index=plot_index,
                unit_positions=_match_unit_positions(
                    record,
                    plot,
                    issues,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                ),
                fallback_start=lower_bound,
            )
        )

    for records in positioned.values():
        records.sort(key=lambda item: (item.start, item.record.line_number))
    return positioned


def load_extracted_dialogues(
    path: Path,
    title: str,
    plots: list[PlotText],
    issues: IssueLog,
) -> dict[int, list[PositionedRecord]]:
    raw = _load_json(path)
    if (
        not isinstance(raw, dict)
        or raw.get("title") != title
        or not isinstance(raw.get("plots"), list)
    ):
        raise ValueError("Extracted dialogue JSON must contain the matching title and plots.")

    plots_by_index = {plot.plot_index: plot for plot in plots}
    positioned: dict[int, list[PositionedRecord]] = {
        plot.plot_index: [] for plot in plots
    }
    seen_plots: set[int] = set()
    for raw_plot in raw["plots"]:
        if not isinstance(raw_plot, dict):
            raise ValueError("Each extracted dialogue plot must be an object.")
        plot_index = raw_plot.get("plot_index")
        dialogues = raw_plot.get("dialogues")
        if (
            isinstance(plot_index, bool)
            or not isinstance(plot_index, int)
            or plot_index not in plots_by_index
            or plot_index in seen_plots
        ):
            raise ValueError(f"Invalid or duplicate dialogue plot_index {plot_index!r}.")
        if not isinstance(dialogues, list):
            raise ValueError(f"Dialogue plot {plot_index} must contain a dialogues array.")
        seen_plots.add(plot_index)

        units: list[MessageUnit] = []
        for dialogue_index, dialogue in enumerate(dialogues):
            if (
                not isinstance(dialogue, dict)
                or set(dialogue) != {"character", "content"}
                or not isinstance(dialogue["character"], str)
                or not isinstance(dialogue["content"], str)
                or not dialogue["content"]
            ):
                raise ValueError(
                    f"Dialogue {plot_index}:{dialogue_index} must contain only "
                    "non-empty string character and content fields."
                )
            units.append(
                MessageUnit(
                    character=dialogue["character"],
                    content=dialogue["content"],
                    environment=False,
                    anchors=(dialogue["content"],),
                    message_index=dialogue_index,
                )
            )

        record = ParsedSftRecord(-(plot_index + 1), tuple(units))
        plot = plots_by_index[plot_index]
        positioned[plot_index].append(
            PositionedRecord(
                record=record,
                plot_index=plot_index,
                unit_positions=_match_unit_positions(record, plot, issues),
            )
        )
    return positioned


def _strong_anchors(record: ParsedSftRecord) -> list[tuple[int, str, bool]]:
    anchors: list[tuple[int, str, bool]] = []
    for unit_index, unit in enumerate(record.units):
        if unit.environment:
            continue
        for anchor in unit.anchors:
            normalized = normalize_text(anchor)
            if len(normalized) >= _STRONG_ANCHOR_LENGTH:
                anchors.append((unit_index, normalized, bool(unit.character)))
    return anchors


def score_record(record: ParsedSftRecord, plot: PlotText) -> RecordMatch:
    anchors = _strong_anchors(record)
    total_chars = sum(len(anchor) for _, anchor, _ in anchors)
    cursor = 0
    matched_chars = 0
    matched_anchors = 0
    target_bonus = 0
    for _, anchor, target in anchors:
        start = plot.normalized.find(anchor, cursor)
        if start < 0:
            continue
        cursor = start + len(anchor)
        matched_chars += len(anchor)
        matched_anchors += 1
        if target:
            target_bonus += 10
    coverage = matched_chars / total_chars if total_chars else 0.0
    return RecordMatch(
        plot_index=plot.plot_index,
        score=matched_chars + target_bonus,
        coverage=coverage,
        matched_anchors=matched_anchors,
    )


def choose_plot(
    record: ParsedSftRecord,
    plots: list[PlotText],
    issues: IssueLog,
    min_coverage: float,
) -> PlotText | None:
    matches = sorted(
        (score_record(record, plot) for plot in plots),
        key=lambda item: (item.score, item.coverage),
        reverse=True,
    )
    if not matches:
        return None
    best = matches[0]
    if (
        best.score < _MIN_MATCHED_CHARS
        or best.matched_anchors < _MIN_MATCHED_ANCHORS
        or best.coverage < min_coverage
    ):
        return None
    if len(matches) > 1 and matches[1].score >= best.score - 5:
        issues.add(
            "sft_plot_ambiguous",
            "SFT record matches more than one covered plot and is skipped.",
            sft_line=record.line_number,
            candidates=[matches[0].plot_index, matches[1].plot_index],
            scores=[matches[0].score, matches[1].score],
        )
        return None
    return next(plot for plot in plots if plot.plot_index == best.plot_index)


def _match_unit_positions(
    record: ParsedSftRecord,
    plot: PlotText,
    issues: IssueLog,
    *,
    lower_bound: int = 0,
    upper_bound: int | None = None,
) -> tuple[int | None, ...]:
    if upper_bound is None:
        upper_bound = len(plot.text)
    if not 0 <= lower_bound <= upper_bound <= len(plot.text):
        raise ValueError(
            f"Invalid message search bounds [{lower_bound}, {upper_bound}) "
            f"for plot {plot.plot_index}."
        )
    normalized_lower = bisect.bisect_left(
        plot.normalized_to_original,
        lower_bound,
    )
    normalized_upper = bisect.bisect_left(
        plot.normalized_to_original,
        upper_bound,
    )
    positions: list[int | None] = [None] * len(record.units)
    normalized_ends: list[int | None] = [None] * len(record.units)
    cursor = normalized_lower
    for unit_index, unit in enumerate(record.units):
        if unit.environment:
            continue
        for raw_anchor in unit.anchors:
            anchor = normalize_text(raw_anchor)
            if len(anchor) < _STRONG_ANCHOR_LENGTH:
                continue
            start = plot.normalized.find(anchor, cursor, normalized_upper)
            if start < 0:
                continue
            if positions[unit_index] is None:
                positions[unit_index] = plot.normalized_to_original[start]
            cursor = start + len(anchor)
            normalized_ends[unit_index] = cursor

    for unit_index, unit in enumerate(record.units):
        if unit.environment or positions[unit_index] is not None:
            continue
        previous_end = normalized_lower
        for index in range(unit_index - 1, -1, -1):
            if normalized_ends[index] is not None:
                previous_end = normalized_ends[index] or 0
                break
        next_start = normalized_upper
        for index in range(unit_index + 1, len(record.units)):
            original_start = positions[index]
            if original_start is not None:
                next_start = bisect.bisect_left(plot.normalized_to_original, original_start)
                break
        candidates: list[tuple[int, int]] = []
        for raw_anchor in unit.anchors:
            anchor = normalize_text(raw_anchor)
            if not anchor:
                continue
            search = previous_end
            while True:
                start = plot.normalized.find(anchor, search, next_start)
                if start < 0:
                    break
                candidates.append((start, start + len(anchor)))
                search = start + 1
        if candidates:
            start, end = min(candidates)
            positions[unit_index] = plot.normalized_to_original[start]
            normalized_ends[unit_index] = end
            if len(candidates) > 1:
                issues.add(
                    "message_source_ambiguous",
                    "Message anchor occurs more than once in its bounded source interval.",
                    plot_index=plot.plot_index,
                    sft_line=record.line_number,
                    message_index=unit.message_index,
                    content=unit.content,
                    occurrences=len(candidates),
                )
        elif any(unit.anchors):
            previous_original = lower_bound
            for index in range(unit_index - 1, -1, -1):
                if positions[index] is not None:
                    previous_original = (positions[index] or 0) + len(
                        record.units[index].source or record.units[index].content
                    )
                    break
            next_original = upper_bound
            for index in range(unit_index + 1, len(record.units)):
                if positions[index] is not None:
                    next_original = positions[index] or len(plot.text)
                    break
            raw_candidates: list[int] = []
            for raw_anchor in unit.anchors:
                if normalize_text(raw_anchor):
                    continue
                search = previous_original
                while True:
                    start = plot.text.find(raw_anchor, search, next_original)
                    if start < 0:
                        break
                    raw_candidates.append(start)
                    search = start + 1
            if raw_candidates:
                positions[unit_index] = min(raw_candidates)
                if len(raw_candidates) > 1:
                    issues.add(
                        "message_source_ambiguous",
                        "Punctuation-only message occurs more than once in its "
                        "bounded source interval.",
                        plot_index=plot.plot_index,
                        sft_line=record.line_number,
                        message_index=unit.message_index,
                        content=unit.content,
                        occurrences=len(raw_candidates),
                    )
                continue
            issues.add(
                "message_source_not_found",
                "Dialogue message could not be located; validation is left empty.",
                plot_index=plot.plot_index,
                sft_line=record.line_number,
                message_index=unit.message_index,
                content=unit.content,
            )
        else:
            issues.add(
                "message_source_not_found",
                "Dialogue message could not be located; validation is left empty.",
                plot_index=plot.plot_index,
                sft_line=record.line_number,
                message_index=unit.message_index,
                content=unit.content,
            )
    return tuple(positions)


def match_sft_records(
    records: list[ParsedSftRecord],
    plots: list[PlotText],
    issues: IssueLog,
    min_coverage: float,
) -> dict[int, list[PositionedRecord]]:
    matched: dict[int, list[PositionedRecord]] = {plot.plot_index: [] for plot in plots}
    for record in records:
        plot = choose_plot(record, plots, issues, min_coverage)
        if plot is None:
            continue
        positioned = PositionedRecord(
            record=record,
            plot_index=plot.plot_index,
            unit_positions=_match_unit_positions(record, plot, issues),
        )
        matched[plot.plot_index].append(positioned)
    for positioned_records in matched.values():
        positioned_records.sort(key=lambda item: (item.start, item.record.line_number))
    return matched


def merge_positioned_records(
    base: dict[int, list[PositionedRecord]],
    added: dict[int, list[PositionedRecord]],
) -> None:
    for plot_index, records in added.items():
        base[plot_index].extend(records)
        base[plot_index].sort(key=lambda item: (item.start, item.record.line_number))


def _build_plot_messages(
    plot: PlotText,
    timeline: StateTimeline,
    records: list[PositionedRecord],
    environment_character: str,
    issues: IssueLog,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for positioned in records:
        for unit_index, unit in enumerate(positioned.record.units):
            position = positioned.unit_positions[unit_index]
            if unit.environment:
                validation: list[dict[str, Any]] = []
            elif position is None:
                validation = []
            else:
                validation = timeline.snapshot_at(position)
            if position is not None:
                key = (position, unit.character, normalize_text(unit.content))
                if key in seen:
                    issues.add(
                        "duplicate_sft_message",
                        "Duplicate positioned SFT message is omitted.",
                        plot_index=plot.plot_index,
                        sft_line=positioned.record.line_number,
                        message_index=unit.message_index,
                        content=unit.content,
                    )
                    continue
                seen.add(key)
            message = {
                "character": unit.character,
                "validation": validation,
                "content": unit.content,
            }
            if unit.character != environment_character:
                message["system"] = ""
            messages.append(message)
    return messages


def build_output(
    title: str,
    plots: list[PlotText],
    matched: dict[int, list[PositionedRecord]],
    environment_character: str,
    issues: IssueLog,
) -> dict[str, Any]:
    output_plots: list[dict[str, Any]] = []
    for plot in plots:
        timeline = build_timeline(plot, issues)
        output_plots.append(
            {
                "plot_index": plot.plot_index,
                "messages": _build_plot_messages(
                    plot,
                    timeline,
                    matched[plot.plot_index],
                    environment_character,
                    issues,
                ),
            }
        )
    return {"title": title, "plots": output_plots}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an RLFF dialogue dataset with state validation snapshots."
    )
    parser.add_argument("state_results", type=Path, help="Extracted state JSONL.")
    parser.add_argument("rebuilt_plots", type=Path, help="Rebuilt plot JSON.")
    parser.add_argument(
        "conversation_jsonl",
        type=Path,
        help="Plot-indexed conversation extraction JSONL.",
    )
    parser.add_argument("output_json", type=Path, help="RLFF dataset JSON to write.")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--environment-character", default="Environment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issues = IssueLog()
    try:
        title, plots = load_plot_texts(
            args.rebuilt_plots,
            args.state_results,
            issues,
        )
        matched = load_conversation_records(
            args.conversation_jsonl,
            title,
            plots,
            issues,
            args.environment_character,
        )
        output = build_output(
            title,
            plots,
            matched,
            args.environment_character,
            issues,
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    log_path = args.log or Path("debug") / "rlff" / f"{title}.jsonl"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        f"{json.dumps(output, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    issues.write(log_path)
    matched_records = sum(len(value) for value in matched.values())
    message_count = sum(len(plot["messages"]) for plot in output["plots"])
    print(
        json.dumps(
            {
                "output": str(args.output_json),
                "log": str(log_path),
                "plots": len(output["plots"]),
                "matched_conversation_records": matched_records,
                "messages": message_count,
                "issues": len(issues.entries),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
