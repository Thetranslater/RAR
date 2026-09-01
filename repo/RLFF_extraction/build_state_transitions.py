"""Merge extracted state nodes into plot-level, position-ordered transitions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

StateValue = str | int | float | bool | None


@dataclass(frozen=True)
class StateNode:
    name: str
    before: StateValue
    after: StateValue
    type: str
    source: str
    description: str

    def without_source(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "before": self.before,
            "after": self.after,
            "type": self.type,
            "description": self.description,
        }


@dataclass(frozen=True)
class StateRecord:
    plot_index: int
    chunk_index: int
    volume: str | None
    chapter: str | None
    nodes: tuple[StateNode, ...]


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _state_value(value: Any, label: str) -> StateValue:
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{label} must be a JSON scalar or null.")
    return value


def load_rebuilt_plots(path: Path) -> dict[str, Any]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(root, dict)
        or not isinstance(root.get("title"), str)
        or not isinstance(root.get("plots"), list)
    ):
        raise ValueError("Rebuilt plot JSON must contain title and plots.")
    for plot_index, plot in enumerate(root["plots"]):
        if not isinstance(plot, dict) or not isinstance(plot.get("chunks"), list):
            raise ValueError(f"plots[{plot_index}] must contain a chunks array.")
        for chunk_index, chunk in enumerate(plot["chunks"]):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                raise ValueError(
                    f"plots[{plot_index}].chunks[{chunk_index}].text must be a string."
                )
    return root


def _parse_node(raw: Any, label: str) -> StateNode:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object.")
    required = {"name", "before", "after", "type", "source", "description"}
    if set(raw) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}.")
    if (
        not isinstance(raw["name"], str)
        or not raw["name"]
        or not isinstance(raw["source"], str)
        or not raw["source"]
        or not isinstance(raw["description"], str)
        or raw["type"] not in {"initialization", "instant"}
    ):
        raise ValueError(f"{label} has invalid string or type fields.")
    before = _state_value(raw["before"], f"{label}.before")
    after = _state_value(raw["after"], f"{label}.after")
    if raw["type"] == "initialization" and before is not None:
        raise ValueError(f"{label} initialization must use before=null.")
    return StateNode(
        name=raw["name"],
        before=before,
        after=after,
        type=raw["type"],
        source=raw["source"],
        description=raw["description"],
    )


def load_state_records(path: Path, expected_title: str) -> list[StateRecord]:
    records: list[StateRecord] = []
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
        plot_index = _integer(raw.get("plot_index"), f"line {line_number}.plot_index")
        chunk_index = _integer(raw.get("chunk_index"), f"line {line_number}.chunk_index")
        key = (plot_index, chunk_index)
        if key in seen:
            raise ValueError(
                f"Duplicate state record for plot {plot_index}, chunk {chunk_index}."
            )
        seen.add(key)
        raw_nodes = raw["result"].get("nodes")
        if not isinstance(raw_nodes, list):
            raise ValueError(f"line {line_number}.result.nodes must be an array.")
        records.append(
            StateRecord(
                plot_index=plot_index,
                chunk_index=chunk_index,
                volume=raw.get("volume"),
                chapter=raw.get("chapter"),
                nodes=tuple(
                    _parse_node(node, f"line {line_number}.nodes[{node_index}]")
                    for node_index, node in enumerate(raw_nodes)
                ),
            )
        )
    if not records:
        raise ValueError("State result JSONL contains no records.")
    return records


def _append_unique(
    nodes: list[StateNode],
    indexes: dict[str, int],
    node: StateNode,
    label: str,
) -> None:
    existing_index = indexes.get(node.name)
    if existing_index is None:
        indexes[node.name] = len(nodes)
        nodes.append(node)
        return
    existing = nodes[existing_index]
    if existing.before != node.before or existing.after != node.after:
        raise ValueError(
            f"{label} has conflicting values for {node.name}: "
            f"{existing.before!r}->{existing.after!r} and "
            f"{node.before!r}->{node.after!r}."
        )


def _locate_source(chunk_text: str, source: str, label: str) -> int:
    start = chunk_text.find(source)
    if start < 0:
        raise ValueError(f"{label} source is not an exact chunk substring: {source!r}")
    if chunk_text.find(source, start + 1) >= 0:
        raise ValueError(f"{label} source occurs more than once in its chunk: {source!r}")
    return start


def build_plot_transition(
    plot_index: int,
    plot: dict[str, Any],
    records: list[StateRecord],
) -> dict[str, Any]:
    chunks = plot["chunks"]
    records = sorted(records, key=lambda record: record.chunk_index)
    actual_chunks = {record.chunk_index for record in records}
    expected_chunks = set(range(len(chunks)))
    if actual_chunks != expected_chunks:
        missing = sorted(expected_chunks - actual_chunks)
        unknown = sorted(actual_chunks - expected_chunks)
        details = []
        if missing:
            details.append(f"missing chunks {missing}")
        if unknown:
            details.append(f"unknown chunks {unknown}")
        raise ValueError(f"Plot {plot_index} state results are incomplete: {', '.join(details)}.")

    for record in records:
        if record.volume != plot.get("volume") or record.chapter != plot.get("chapter"):
            raise ValueError(
                f"Plot {plot_index}, chunk {record.chunk_index} metadata does not "
                "match the rebuilt plot."
            )

    chunk_starts: list[int] = []
    position = 0
    for chunk in chunks:
        chunk_starts.append(position)
        position += len(chunk["text"])

    initial_nodes: list[StateNode] = []
    initial_names: dict[str, int] = {}
    instant_groups: dict[int, dict[str, Any]] = {}

    for record in records:
        chunk_text = chunks[record.chunk_index]["text"]
        for node_index, node in enumerate(record.nodes):
            label = f"plot {plot_index}, chunk {record.chunk_index}, node {node_index}"
            local_start = _locate_source(chunk_text, node.source, label)
            if node.type == "initialization":
                _append_unique(initial_nodes, initial_names, node, "Initialization")
                continue

            end_position = (
                chunk_starts[record.chunk_index] + local_start + len(node.source)
            )
            group = instant_groups.setdefault(
                end_position,
                {"source": node.source, "nodes": [], "names": {}},
            )
            if len(node.source) > len(group["source"]):
                if not node.source.endswith(group["source"]):
                    raise ValueError(
                        f"Sources ending at position {end_position} are not suffix-related."
                    )
                group["source"] = node.source
            elif not group["source"].endswith(node.source):
                raise ValueError(
                    f"Sources ending at position {end_position} are not suffix-related."
                )
            _append_unique(group["nodes"], group["names"], node, f"Transition {end_position}")

    transition = [
        {
            "source": "",
            "nodes": [node.without_source() for node in initial_nodes],
        }
    ]
    for end_position in sorted(instant_groups):
        group = instant_groups[end_position]
        transition.append(
            {
                "source": group["source"],
                "nodes": [node.without_source() for node in group["nodes"]],
            }
        )
    return {
        "plot_index": plot_index,
        "volume": plot.get("volume"),
        "chapter": plot.get("chapter"),
        "transition": transition,
    }


def build_output(
    rebuilt: dict[str, Any],
    records: list[StateRecord],
) -> dict[str, Any]:
    grouped: dict[int, list[StateRecord]] = {}
    for record in records:
        if record.plot_index >= len(rebuilt["plots"]):
            raise ValueError(f"Unknown plot_index {record.plot_index}.")
        grouped.setdefault(record.plot_index, []).append(record)
    return {
        "title": rebuilt["title"],
        "plots": [
            build_plot_transition(plot_index, rebuilt["plots"][plot_index], grouped[plot_index])
            for plot_index in sorted(grouped)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge extracted state nodes into ordered plot transitions."
    )
    parser.add_argument("rebuilt_plots", type=Path, help="Rebuilt plot JSON.")
    parser.add_argument("state_results", type=Path, help="State extraction JSONL.")
    parser.add_argument("output_json", type=Path, help="Transition JSON to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rebuilt = load_rebuilt_plots(args.rebuilt_plots)
        records = load_state_records(args.state_results, rebuilt["title"])
        output = build_output(rebuilt, records)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    transition_count = sum(len(plot["transition"]) for plot in output["plots"])
    print(
        f"Wrote {len(output['plots'])} plots and {transition_count} transitions "
        f"to {args.output_json}"
    )


if __name__ == "__main__":
    main()
