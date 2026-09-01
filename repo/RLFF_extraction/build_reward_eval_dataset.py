"""Build a reproducible LLaMA-Factory evaluation set linked to RLFF facts.

Each output example uses canonical ShareGPT ``conversations`` with ``from`` and
``value`` fields and ends at one GPT response. LLaMA-Factory can therefore use
all preceding turns as context and generate that final response, while the
sidecar mapping keeps the RLFF validation facts needed by the reward model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "evaluation" / "roleplay_reward_model"
ENVIRONMENT_CHARACTER = "Environment"


@dataclass(frozen=True)
class BookConfig:
    key: str
    title: str
    rlff_path: Path
    sft_path: Path
    metadata_path: Path | None = None
    character_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class SftRecord:
    line_number: int
    plot_index: int
    character: str
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Candidate:
    book_key: str
    title: str
    sft_line: int
    sft_message_index: int
    plot_index: int
    rlff_message_index: int
    character: str
    messages: tuple[dict[str, Any], ...]
    history: str
    validations: tuple[dict[str, Any], ...]
    reference_response: str

    @property
    def priority(self) -> int:
        after_plot_start = self.rlff_message_index >= 3
        has_validation = bool(self.validations)
        if after_plot_start and has_validation:
            return 0
        if after_plot_start:
            return 1
        if has_validation:
            return 2
        return 3


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _messages(row: dict[str, Any], path: Path, line_number: int) -> list[dict[str, Any]]:
    value = row.get("messages")
    if value is None and isinstance(row.get("data"), dict):
        value = row["data"].get("messages")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} line {line_number} has no messages array")
    if any(not isinstance(message, dict) for message in value):
        raise ValueError(f"{path} line {line_number} contains an invalid message")
    return value


def _message_fingerprint(messages: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(
        list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assistant_fingerprint(messages: Iterable[dict[str, Any]]) -> str:
    """Identify a dialogue independent of prompt variant and synthetic lead-in."""
    responses = [
        message.get("content")
        for message in messages
        if message.get("role") == "assistant"
    ]
    payload = json.dumps(responses, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", value)
        if character.isalnum()
    )


def _character_matches(config: BookConfig, target: str, source: Any) -> bool:
    if not isinstance(source, str):
        return False
    if source == target:
        return True
    aliases = dict(config.character_aliases).get(target, ())
    return source in aliases


def _load_sft_records(config: BookConfig) -> list[SftRecord]:
    raw_rows = _read_jsonl(config.sft_path)
    metadata_by_messages: dict[str, list[dict[str, Any]]] = {}
    if config.metadata_path is not None:
        for line_number, row in enumerate(_read_jsonl(config.metadata_path), 1):
            fingerprint = _assistant_fingerprint(
                _messages(row, config.metadata_path, line_number)
            )
            metadata_by_messages.setdefault(fingerprint, []).append(row)

    records: list[SftRecord] = []
    for line_number, row in enumerate(raw_rows, 1):
        messages = _messages(row, config.sft_path, line_number)
        metadata_row = row
        if config.metadata_path is not None:
            fingerprint = _assistant_fingerprint(messages)
            metadata_matches = metadata_by_messages.get(fingerprint, [])
            if not metadata_matches:
                raise ValueError(
                    f"Cannot recover metadata for {config.sft_path} line {line_number}"
                )
            identities = {
                (match.get("details", {}).get("plot"), match.get("details", {}).get("character"))
                for match in metadata_matches
            }
            if len(identities) != 1:
                raise ValueError(
                    f"Ambiguous metadata for {config.sft_path} line {line_number}: "
                    f"{sorted(identities)}"
                )
            metadata_row = metadata_matches[0]
        details = metadata_row.get("details")
        if not isinstance(details, dict):
            raise ValueError(f"{config.sft_path} line {line_number} has no details metadata")
        plot_index = details.get("plot")
        character = details.get("character")
        if isinstance(plot_index, bool) or not isinstance(plot_index, int):
            raise ValueError(f"{config.sft_path} line {line_number} has invalid plot index")
        if not isinstance(character, str) or not character:
            raise ValueError(f"{config.sft_path} line {line_number} has invalid character")
        records.append(
            SftRecord(
                line_number=line_number,
                plot_index=plot_index,
                character=character,
                messages=tuple(messages),
            )
        )
    return records


def _load_rlff(config: BookConfig) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    root = _read_json(config.rlff_path)
    if not isinstance(root, dict) or not isinstance(root.get("plots"), list):
        raise ValueError(f"{config.rlff_path} must contain a plots array")
    by_plot: dict[int, list[dict[str, Any]]] = {}
    for position, plot in enumerate(root["plots"]):
        if not isinstance(plot, dict) or not isinstance(plot.get("messages"), list):
            raise ValueError(f"{config.rlff_path} plots[{position}] is invalid")
        plot_index = plot.get("plot_index")
        if isinstance(plot_index, bool) or not isinstance(plot_index, int):
            raise ValueError(f"{config.rlff_path} plots[{position}] has invalid plot_index")
        if plot_index in by_plot:
            raise ValueError(f"{config.rlff_path} contains duplicate plot {plot_index}")
        by_plot[plot_index] = plot["messages"]
    return root, by_plot


def _match_candidates(config: BookConfig) -> tuple[list[Candidate], dict[str, Any]]:
    records = _load_sft_records(config)
    rlff_root, rlff_by_plot = _load_rlff(config)
    candidates: list[Candidate] = []
    unmatched: list[dict[str, Any]] = []

    for record in records:
        plot_messages = rlff_by_plot.get(record.plot_index)
        if plot_messages is None:
            unmatched.append(
                {
                    "sft_line": record.line_number,
                    "reason": "plot_missing_from_rlff",
                    "plot_index": record.plot_index,
                }
            )
            continue
        cursor = -1
        for sft_message_index, message in enumerate(record.messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"{config.sft_path} line {record.line_number} contains an empty assistant"
                )
            normalized = _normalize_text(content)
            matches = [
                rlff_message_index
                for rlff_message_index in range(cursor + 1, len(plot_messages))
                if isinstance(plot_messages[rlff_message_index], dict)
                and _character_matches(
                    config,
                    record.character,
                    plot_messages[rlff_message_index].get("character"),
                )
                and _normalize_text(str(plot_messages[rlff_message_index].get("content", "")))
                == normalized
            ]
            if not matches:
                unmatched.append(
                    {
                        "sft_line": record.line_number,
                        "sft_message_index": sft_message_index,
                        "reason": "assistant_not_found_in_rlff",
                        "plot_index": record.plot_index,
                        "character": record.character,
                        "content": content,
                    }
                )
                continue
            rlff_message_index = matches[0]
            cursor = rlff_message_index
            rlff_message = plot_messages[rlff_message_index]
            validations = rlff_message.get("validation", [])
            if not isinstance(validations, list):
                raise ValueError(
                    f"{config.rlff_path} plot {record.plot_index} message "
                    f"{rlff_message_index} has invalid validation"
                )
            candidates.append(
                Candidate(
                    book_key=config.key,
                    title=config.title,
                    sft_line=record.line_number,
                    sft_message_index=sft_message_index,
                    plot_index=record.plot_index,
                    rlff_message_index=rlff_message_index,
                    character=record.character,
                    messages=record.messages[: sft_message_index + 1],
                    history="\n".join(
                        f"{previous.get('character') or 'Unknown'}:{previous.get('content', '')}"
                        for previous in plot_messages[:rlff_message_index]
                    ),
                    validations=tuple(validations),
                    reference_response=content,
                )
            )

    non_environment_count = sum(
        message.get("character") != ENVIRONMENT_CHARACTER
        for messages in rlff_by_plot.values()
        for message in messages
        if isinstance(message, dict)
    )
    report = {
        "title": rlff_root.get("title", config.title),
        "rlff_plot_count": len(rlff_by_plot),
        "rlff_message_count": sum(len(messages) for messages in rlff_by_plot.values()),
        "rlff_non_environment_count": non_environment_count,
        "sft_record_count": len(records),
        "sft_assistant_count": sum(
            message.get("role") == "assistant"
            for record in records
            for message in record.messages
        ),
        "matched_candidate_count": len(candidates),
        "preferred_candidate_count": sum(candidate.priority == 0 for candidate in candidates),
        "unmatched_assistant_count": len(unmatched),
        "unmatched": unmatched,
    }
    return candidates, report


def _target_count(population: int, fraction: float) -> int:
    return max(1, math.floor(population * fraction + 0.5))


def _select(
    candidates: list[Candidate], target: int, rng: random.Random
) -> tuple[list[Candidate], Counter[int]]:
    selected: list[Candidate] = []
    priorities: Counter[int] = Counter()
    for priority in range(4):
        tier = [candidate for candidate in candidates if candidate.priority == priority]
        rng.shuffle(tier)
        take = min(target - len(selected), len(tier))
        selected.extend(tier[:take])
        priorities[priority] += take
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError(
            f"Only {len(selected)} matched SFT/RLFF candidates are available for target {target}"
        )
    return selected, priorities


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _default_books() -> list[BookConfig]:
    roleplay = PROJECT_ROOT / "dataset" / "sft" / "roleplay"
    rlff = PROJECT_ROOT / "dataset" / "rlff"
    return [
        BookConfig(
            key="toradora",
            title="2龙与虎",
            rlff_path=rlff / "2龙与虎" / "2龙与虎.json",
            sft_path=roleplay / "2龙与虎" / "half.jsonl",
            metadata_path=roleplay / "2龙与虎" / "all.jsonl",
            character_aliases=(
                ("高须龙儿", ("龙儿",)),
                ("高须泰子", ("泰子",)),
                ("北村佑作", ("北村",)),
                ("栉枝实乃梨", ("实乃梨",)),
                ("能登久光", ("能登",)),
            ),
        ),
        BookConfig(
            key="index",
            title="3魔法禁书目录",
            rlff_path=rlff / "魔法禁书目录" / "3魔法禁书目录.json",
            sft_path=roleplay / "3魔法禁书目录" / "all.jsonl",
            character_aliases=(
                ("上条当麻", ("上条",)),
                ("神裂", ("神裂火织",)),
                ("史提尔", ("史提尔·马格努斯",)),
                ("小萌老师", ("月咏小萌",)),
            ),
        ),
        BookConfig(
            key="seitokai",
            title="7碧阳学园学生会议事录(学生会的一己之见)",
            rlff_path=(
                rlff
                / "7碧阳学园学生会议事录(学生会的一己之见)"
                / "7碧阳学园学生会议事录(学生会的一己之见).json"
            ),
            sft_path=(
                roleplay
                / "7碧阳学园学生会议事录(学生会的一己之见)"
                / "all.jsonl"
            ),
            character_aliases=(
                ("会长", ("樱野玖璃梦", "玖璃梦")),
                ("小真冬", ("椎名真冬", "真冬")),
                ("杉崎键", ("杉崎",)),
                ("深夏", ("椎名深夏",)),
                ("知弦学姐", ("红叶知弦", "知弦")),
            ),
        ),
    ]


def build(output_dir: Path, seed: int, fraction: float) -> dict[str, Any]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected_all: list[Candidate] = []
    book_reports: list[dict[str, Any]] = []

    unmatched_audit: list[dict[str, Any]] = []
    for config in _default_books():
        candidates, report = _match_candidates(config)
        for item in report.pop("unmatched"):
            unmatched_audit.append({"book_key": config.key, "title": config.title, **item})
        target = _target_count(report["rlff_non_environment_count"], fraction)
        selected, priorities = _select(candidates, target, rng)
        selected_all.extend(selected)
        report.update(
            {
                "sample_target": target,
                "sampled_count": len(selected),
                "sampled_fraction_of_rlff_non_environment": (
                    len(selected) / report["rlff_non_environment_count"]
                ),
                "sampled_priority_counts": {
                    "message_index_ge_3_with_validation": priorities[0],
                    "message_index_ge_3_without_validation": priorities[1],
                    "message_index_lt_3_with_validation": priorities[2],
                    "message_index_lt_3_without_validation": priorities[3],
                },
                "sampled_plot_count": len({candidate.plot_index for candidate in selected}),
                "sampled_character_count": len({candidate.character for candidate in selected}),
                "sampled_validation_count": sum(
                    len(candidate.validations) for candidate in selected
                ),
            }
        )
        book_reports.append(report)

    rng.shuffle(selected_all)
    evaluation_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(selected_all):
        sample_id = f"reward-eval-{position:04d}"
        raw_messages = list(candidate.messages)
        system = ""
        if raw_messages and raw_messages[0].get("role") == "system":
            raw_system = raw_messages.pop(0).get("content")
            if not isinstance(raw_system, str):
                raise ValueError(f"Sample {sample_id} has an invalid system prompt")
            system = raw_system
        role_names = {"user": "human", "assistant": "gpt"}
        conversations: list[dict[str, str]] = []
        for message in raw_messages:
            role = message.get("role")
            content = message.get("content")
            if role not in role_names or not isinstance(content, str):
                raise ValueError(f"Sample {sample_id} has an invalid conversation message")
            conversations.append({"from": role_names[role], "value": content})
        evaluation_rows.append(
            {"sample_id": sample_id, "system": system, "conversations": conversations}
        )
        mapping_rows.append(
            {
                "sample_id": sample_id,
                "evaluation_line": position + 1,
                "book_key": candidate.book_key,
                "title": candidate.title,
                "plot_index": candidate.plot_index,
                "message_index": candidate.rlff_message_index,
                "character": candidate.character,
                "sft_line": candidate.sft_line,
                "sft_message_index": candidate.sft_message_index,
                "reference_response": candidate.reference_response,
                "history": candidate.history,
                "validation": list(candidate.validations),
            }
        )

    evaluation_path = output_dir / "evaluation.jsonl"
    mapping_path = output_dir / "rlff_mapping.jsonl"
    audit_path = output_dir / "unmatched_audit.jsonl"
    _write_jsonl(evaluation_path, evaluation_rows)
    _write_jsonl(mapping_path, mapping_rows)
    _write_jsonl(audit_path, unmatched_audit)
    dataset_info = {
        "share_gpt_all": {
            "file_name": "share_gpt_all.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "roleplay_reward_eval": {
            "file_name": evaluation_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "seed": seed,
        "fraction": fraction,
        "sampling_scope": "one quarter of non-Environment RLFF messages per book",
        "priority_order": [
            "message_index >= 3 and validation is non-empty",
            "message_index >= 3 and validation is empty",
            "message_index < 3 and validation is non-empty",
            "message_index < 3 and validation is empty",
        ],
        "evaluation_count": len(evaluation_rows),
        "mapping_count": len(mapping_rows),
        "books": book_reports,
        "outputs": {
            "evaluation": str(evaluation_path),
            "mapping": str(mapping_path),
            "dataset_info": str(output_dir / "dataset_info.json"),
            "unmatched_audit": str(audit_path),
        },
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args.output_dir.resolve(), args.seed, args.fraction)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
