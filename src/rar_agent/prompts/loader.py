"""Load default or user-selected prompt files without making prompts stateful."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPT_STAGES = (
    "plot_extraction",
    "character_filter",
    "dialogue_extraction",
    "character_profile",
)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    filename: str
    text: str


class PromptCatalog:
    def __init__(self, overrides: dict[str, Path] | None = None) -> None:
        unknown = set(overrides or {}) - set(PROMPT_STAGES)
        if unknown:
            raise ValueError(f"unknown prompt stages: {sorted(unknown)}")
        self._overrides = overrides or {}
        self._defaults = Path(__file__).parent / "defaults"

    def load(self, stage: str) -> PromptTemplate:
        if stage not in PROMPT_STAGES:
            raise ValueError(f"unknown prompt stage: {stage}")
        path = self._overrides.get(stage, self._defaults / f"{stage}.txt")
        return PromptTemplate(filename=path.name, text=path.read_text(encoding="utf-8"))
