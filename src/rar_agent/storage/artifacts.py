"""Stable Dataset directory layout and atomic JSON persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    root: Path
    input_manifest: Path
    work: Path
    text_chunks: Path
    plot_extractions: Path
    plots: Path
    dialogue_extractions: Path
    character_profiles: Path
    dataset: Path
    characters: Path
    exports: Path
    report: Path

    @classmethod
    def from_root(cls, root: Path) -> DatasetPaths:
        work = root / "work"
        return cls(
            root=root,
            input_manifest=root / "input_manifest.json",
            work=work,
            text_chunks=work / "text_chunks.jsonl",
            plot_extractions=work / "plot_extractions.jsonl",
            plots=work / "plots.json",
            dialogue_extractions=work / "dialogue_extractions.jsonl",
            character_profiles=work / "character_profiles.jsonl",
            dataset=root / "dataset.json",
            characters=root / "characters",
            exports=root / "exports",
            report=root / "report",
        )


def _slugify(name: str) -> str:
    ascii_name = name.strip().lower().replace("_", "-").replace(" ", "-")
    slug = _SLUG_UNSAFE.sub("-", ascii_name).strip("-")
    if slug:
        return slug
    # Non-Latin dataset names remain readable and are safe after separator replacement.
    unicode_slug = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", name.strip()).strip(" .-")
    return unicode_slug or "dataset"


class DatasetArtifactStore:
    """Own one Dataset's stable, user-inspectable artifact layout."""

    def __init__(self, project_root: Path, dataset_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.root = dataset_root.resolve()
        self.paths = DatasetPaths.from_root(self.root)
        self._ensure_inside_project(self.root)

    @classmethod
    def create(cls, project_root: Path, dataset_name: str) -> DatasetArtifactStore:
        project = project_root.resolve()
        datasets = project / "datasets"
        datasets.mkdir(parents=True, exist_ok=True)
        stem = _slugify(dataset_name)
        candidate = datasets / stem
        suffix = 2
        while candidate.exists():
            candidate = datasets / f"{stem}-{suffix}"
            suffix += 1
        store = cls(project, candidate)
        for directory in (
            store.paths.work,
            store.paths.characters,
            store.paths.exports,
            store.paths.report,
        ):
            directory.mkdir(parents=True, exist_ok=False)
        return store

    @classmethod
    def open(cls, project_root: Path, dataset_root: Path) -> DatasetArtifactStore:
        store = cls(project_root, dataset_root)
        if not store.root.is_dir():
            raise FileNotFoundError(store.root)
        return store

    def _ensure_inside_project(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.project_root)
        except ValueError as error:
            raise ValueError(f"artifact path escapes project: {path}") from error

    def _checked_path(self, path: Path) -> Path:
        resolved = path.resolve()
        self._ensure_inside_project(resolved)
        return resolved

    def write_json(self, path: Path, value: Any) -> None:
        self._atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def read_json(self, path: Path) -> Any:
        return json.loads(self._checked_path(path).read_text(encoding="utf-8"))

    def write_jsonl(self, path: Path, values: Iterable[JsonObject]) -> None:
        content = "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in values
        )
        self._atomic_write(path, content)

    def read_jsonl(self, path: Path) -> list[JsonObject]:
        checked = self._checked_path(path)
        if not checked.exists():
            return []
        rows: list[JsonObject] = []
        for line_number, line in enumerate(checked.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{checked.name}:{line_number} must contain an object")
            rows.append(value)
        return rows

    def pending_unit_indexes(
        self, path: Path, expected_inputs: list[JsonObject]
    ) -> list[int]:
        rows = self.read_jsonl(path)
        if len(rows) > len(expected_inputs):
            raise ValueError(f"{path.name} contains more rows than expected inputs")
        pending: list[int] = []
        for index, expected in enumerate(expected_inputs):
            if index >= len(rows):
                pending.append(index)
                continue
            row = rows[index]
            if row.get("input") != expected:
                raise ValueError(f"{path.name} input mismatch at index {index}")
            if row.get("result") == {}:
                pending.append(index)
        return pending

    def _atomic_write(self, path: Path, content: str) -> None:
        checked = self._checked_path(path)
        checked.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{checked.name}.", suffix=".tmp", dir=checked.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, checked)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
