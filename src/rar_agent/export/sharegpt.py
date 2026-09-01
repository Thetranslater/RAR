"""Convert a DatasetBundle into role-specific ShareGPT training samples."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rar_agent.domain.models import CharacterProfile, DatasetBundle, Plot, PlotsDocument

JsonObject = dict[str, Any]
_UNSAFE_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


@dataclass(frozen=True, slots=True)
class ShareGPTExportReport:
    """Summary of the deterministic export operation."""

    output_dir: Path
    sample_count: int
    character_counts: dict[str, int]


class ShareGPTExporter:
    """Build one training sample per Conversation and target character."""

    def export(
        self,
        bundle: DatasetBundle,
        plots: PlotsDocument,
        output_dir: Path,
        *,
        system_template: str,
    ) -> ShareGPTExportReport:
        output_dir.mkdir(parents=True, exist_ok=True)
        character_dir = output_dir / "characters"
        character_dir.mkdir(parents=True, exist_ok=True)

        plot_by_index = {plot.index: plot for plot in plots.plots}
        samples: list[JsonObject] = []
        per_character: dict[str, list[JsonObject]] = defaultdict(list)

        for conversation in bundle.conversations:
            plot = plot_by_index.get(conversation.plot_ref.index)
            if plot is None:
                raise ValueError(
                    f"Conversation references missing Plot {conversation.plot_ref.index}"
                )
            normalized = self._merge_same_speaker(
                [(item.speaker, item.content) for item in conversation.utterances]
            )
            speakers = {speaker for speaker, _ in normalized}
            for character in bundle.characters:
                if character.name not in speakers:
                    continue
                sample = self._build_sample(
                    character,
                    normalized,
                    plot,
                    system_template=system_template,
                )
                if sample is None:
                    continue
                samples.append(sample)
                per_character[character.name].append(sample)

        self._write_jsonl(output_dir / "all.jsonl", samples)
        character_counts: dict[str, int] = {}
        for index, character in enumerate(bundle.characters):
            rows = per_character.get(character.name, [])
            character_counts[character.name] = len(rows)
            self._write_jsonl(
                character_dir / f"{index:04d}-{self._safe_filename(character.name)}.jsonl",
                rows,
            )

        return ShareGPTExportReport(
            output_dir=output_dir,
            sample_count=len(samples),
            character_counts=character_counts,
        )

    @staticmethod
    def _build_sample(
        character: CharacterProfile,
        utterances: list[tuple[str, str]],
        plot: Plot,
        *,
        system_template: str,
    ) -> JsonObject | None:
        plot_text = "\n".join(chunk.text for chunk in plot.chunks)
        messages: list[JsonObject] = [
            {
                "role": "system",
                "content": system_template.format(
                    character=character.name,
                    profile=character.profile,
                    plot=plot_text,
                ),
            }
        ]

        for speaker, content in utterances:
            role = "assistant" if speaker == character.name else "user"
            # The full-width colon is part of RAR's stable training contract.
            rendered = f"{speaker}：{content}"  # noqa: RUF001
            if messages[-1]["role"] == role:
                messages[-1]["content"] += f"\n{rendered}"
            else:
                message: JsonObject = {"role": role, "content": rendered}
                if role == "assistant":
                    message["loss"] = True
                messages.append(message)

        while len(messages) > 1 and messages[-1]["role"] == "user":
            messages.pop()
        if not any(message["role"] == "assistant" for message in messages):
            return None
        return {"messages": messages}

    @staticmethod
    def _merge_same_speaker(utterances: list[tuple[str, str]]) -> list[tuple[str, str]]:
        merged: list[tuple[str, str]] = []
        for speaker, content in utterances:
            if merged and merged[-1][0] == speaker:
                previous_speaker, previous_content = merged[-1]
                merged[-1] = (previous_speaker, f"{previous_content}\n{content}")
            else:
                merged.append((speaker, content))
        return merged

    @staticmethod
    def _safe_filename(name: str) -> str:
        return _UNSAFE_FILE_CHARS.sub("-", name).strip(" .-") or "character"

    @staticmethod
    def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
        content = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )
        path.write_text(content, encoding="utf-8", newline="\n")
