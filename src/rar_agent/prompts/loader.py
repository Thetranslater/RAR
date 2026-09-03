"""Load default or user-selected prompt files without making prompts stateful."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROMPT_STAGES = (
    "plot_extraction",
    "character_profile",
    "dialogue_extraction",
)
PROMPT_SEPARATOR_RE = re.compile(r"^----------$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    system: str
    user_prefix: str


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    filename: str
    text: str

    def render(self, replacements: dict[str, str] | None = None) -> RenderedPrompt:
        """Render placeholders and split an optional RLFF-style user prefix."""

        normalized = self.text.replace("\r\n", "\n").replace("\r", "\n")
        delimiters = list(PROMPT_SEPARATOR_RE.finditer(normalized))
        if delimiters:
            delimiter = delimiters[-1]
            system = normalized[: delimiter.start()].strip()
            user_prefix = normalized[delimiter.end() :].strip()
        else:
            system = normalized.strip()
            user_prefix = ""
        if not system:
            raise ValueError(f"prompt system section is empty: {self.filename}")
        for key, value in (replacements or {}).items():
            placeholder = "{" + key + "}"
            system = system.replace(placeholder, value)
            user_prefix = user_prefix.replace(placeholder, value)
        return RenderedPrompt(system=system, user_prefix=user_prefix)


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
