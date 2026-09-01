"""Composable deterministic tools with Project path containment."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rar_agent.domain.models import DatasetBundle
from rar_agent.models.base import JsonObject, ToolCall, ToolDefinition
from rar_agent.storage.artifacts import DatasetPaths

ToolEffect = Literal["read", "write", "delete", "network", "shell"]
ToolHandler = Callable[[JsonObject], JsonObject | Awaitable[JsonObject]]
ApprovalCallback = Callable[[ToolCall, ToolEffect], bool | Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    definition: ToolDefinition
    effect: ToolEffect
    handler: ToolHandler


@dataclass(frozen=True, slots=True)
class ToolObservation:
    success: bool
    content: JsonObject
    effect: ToolEffect
    uncertain: bool = False

    def model_payload(self) -> JsonObject:
        return {
            "success": self.success,
            "content": self.content,
            "uncertain": self.uncertain,
        }


class ToolDispatcher:
    def __init__(
        self,
        project_root: Path,
        tools: list[RegisteredTool],
        *,
        approval: ApprovalCallback | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self._tools = {tool.definition.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("tool names must be unique")
        self._approval = approval

    def definitions(self, names: set[str] | None = None) -> list[ToolDefinition]:
        return [
            tool.definition
            for name, tool in self._tools.items()
            if names is None or name in names
        ]

    async def dispatch(self, call: ToolCall) -> ToolObservation:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolObservation(False, {"error": f"unknown tool: {call.name}"}, "read")
        if self._approval is not None:
            approved = self._approval(call, tool.effect)
            if inspect.isawaitable(approved):
                approved = await approved
            if not approved:
                return ToolObservation(False, {"error": "operation was not approved"}, tool.effect)
        try:
            value = tool.handler(call.arguments)
            if inspect.isawaitable(value):
                value = await value
            return ToolObservation(True, value, tool.effect)
        except Exception as error:
            return ToolObservation(False, {"error": str(error)}, tool.effect)


def build_workspace_tools(project_root: Path) -> list[RegisteredTool]:
    root = project_root.resolve()

    def checked(path_value: object) -> Path:
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("path must be a non-empty string")
        path = (root / path_value).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("path escapes the Project") from error
        return path

    def inspect_project(_arguments: JsonObject) -> JsonObject:
        datasets_root = root / "datasets"
        datasets: list[JsonObject] = []
        complete = 0
        in_progress = 0
        intermediate_count = 0
        if datasets_root.is_dir():
            for dataset_root in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
                paths = DatasetPaths.from_root(dataset_root)
                name = dataset_root.name
                if paths.input_manifest.is_file():
                    try:
                        manifest = json.loads(paths.input_manifest.read_text(encoding="utf-8"))
                        if isinstance(manifest, dict) and isinstance(manifest.get("name"), str):
                            name = manifest["name"]
                    except (OSError, ValueError):
                        pass

                final_dataset: JsonObject | None = None
                status = "in_progress"
                if paths.dataset.is_file():
                    try:
                        bundle = DatasetBundle.model_validate_json(
                            paths.dataset.read_text(encoding="utf-8")
                        )
                        name = bundle.name
                        final_dataset = {
                            "path": paths.dataset.relative_to(root).as_posix(),
                            "valid": True,
                            "characters": len(bundle.characters),
                            "conversations": len(bundle.conversations),
                        }
                        status = "complete"
                    except (OSError, ValueError) as error:
                        final_dataset = {
                            "path": paths.dataset.relative_to(root).as_posix(),
                            "valid": False,
                            "error": str(error),
                        }
                        status = "invalid"

                known_intermediates = (
                    ("input_manifest", paths.input_manifest),
                    ("text_chunks", paths.text_chunks),
                    ("plot_extractions", paths.plot_extractions),
                    ("character_filter", paths.character_filter),
                    ("plots", paths.plots),
                    ("dialogue_extractions", paths.dialogue_extractions),
                    ("character_profiles", paths.character_profiles),
                )
                intermediates = [
                    {
                        "kind": kind,
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                    }
                    for kind, path in known_intermediates
                    if path.is_file()
                ]
                intermediate_count += len(intermediates)
                if status == "complete":
                    complete += 1
                else:
                    in_progress += 1

                def files_under(directory: Path) -> list[str]:
                    if not directory.is_dir():
                        return []
                    return [
                        path.relative_to(root).as_posix()
                        for path in sorted(directory.rglob("*"))
                        if path.is_file()
                    ]

                datasets.append(
                    {
                        "directory": dataset_root.name,
                        "path": dataset_root.relative_to(root).as_posix(),
                        "name": name,
                        "status": status,
                        "final_dataset": final_dataset,
                        "intermediate_artifacts": intermediates,
                        "exports": files_under(paths.exports),
                        "reports": files_under(paths.report),
                    }
                )
        return {
            "summary": {
                "datasets": len(datasets),
                "complete": complete,
                "in_progress": in_progress,
                "intermediate_artifacts": intermediate_count,
            },
            "datasets": datasets,
        }

    def read_file(arguments: JsonObject) -> JsonObject:
        path = checked(arguments.get("path"))
        return {"path": path.relative_to(root).as_posix(), "content": path.read_text("utf-8")}

    def list_files(arguments: JsonObject) -> JsonObject:
        directory = checked(arguments.get("path", "."))
        if not directory.is_dir():
            raise ValueError("path is not a directory")
        limit = int(arguments.get("limit", 500))
        files = [
            path.relative_to(root).as_posix()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and ".rar" not in path.relative_to(root).parts
        ][:limit]
        return {"files": files, "truncated": len(files) == limit}

    def write_file(arguments: JsonObject) -> JsonObject:
        path = checked(arguments.get("path"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return {"path": path.relative_to(root).as_posix(), "bytes": len(content.encode())}

    def replace_text(arguments: JsonObject) -> JsonObject:
        path = checked(arguments.get("path"))
        old = arguments.get("old")
        new = arguments.get("new")
        if not isinstance(old, str) or not old:
            raise ValueError("old must be a non-empty string")
        if not isinstance(new, str):
            raise ValueError("new must be a string")
        content = path.read_text(encoding="utf-8")
        occurrences = content.count(old)
        if occurrences == 0:
            raise ValueError("old text was not found")
        replace_all = bool(arguments.get("replace_all", False))
        updated = content.replace(old, new, -1 if replace_all else 1)
        path.write_text(updated, encoding="utf-8", newline="\n")
        return {
            "path": path.relative_to(root).as_posix(),
            "replacements": occurrences if replace_all else 1,
        }

    def validate_dataset(arguments: JsonObject) -> JsonObject:
        path = checked(arguments.get("path"))
        bundle = DatasetBundle.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return {
            "valid": True,
            "characters": len(bundle.characters),
            "conversations": len(bundle.conversations),
        }

    def delete_file(arguments: JsonObject) -> JsonObject:
        path = checked(arguments.get("path"))
        if not path.is_file():
            raise ValueError("path is not a file")
        path.unlink()
        return {"path": path.relative_to(root).as_posix(), "deleted": True}

    def tool(
        name: str,
        description: str,
        effect: ToolEffect,
        properties: JsonObject,
        required: list[str],
        handler: ToolHandler,
    ) -> RegisteredTool:
        return RegisteredTool(
            ToolDefinition(
                name=name,
                description=description,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            ),
            effect,
            handler,
        )

    path_property = {"type": "string", "description": "Project-relative path"}
    return [
        tool(
            "inspect_project",
            (
                "Inspect existing RAR Datasets, fixed workflow intermediate artifacts, "
                "exports and reports. Use this first for Project or Dataset status questions; "
                "do not inspect source code to infer the artifact layout."
            ),
            "read",
            {},
            [],
            inspect_project,
        ),
        tool(
            "read_file",
            "Read one UTF-8 file.",
            "read",
            {"path": path_property},
            ["path"],
            read_file,
        ),
        tool(
            "list_files",
            "List files recursively under a Project directory.",
            "read",
            {"path": path_property, "limit": {"type": "integer", "minimum": 1}},
            [],
            list_files,
        ),
        tool(
            "write_file",
            "Create or overwrite one UTF-8 file.",
            "write",
            {"path": path_property, "content": {"type": "string"}},
            ["path", "content"],
            write_file,
        ),
        tool(
            "replace_text",
            "Replace exact text in one UTF-8 file.",
            "write",
            {
                "path": path_property,
                "old": {"type": "string"},
                "new": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            ["path", "old", "new"],
            replace_text,
        ),
        tool(
            "validate_dataset",
            "Validate a dataset.json against the RAR schema.",
            "read",
            {"path": path_property},
            ["path"],
            validate_dataset,
        ),
        tool(
            "delete_file",
            "Delete one file inside the Project.",
            "delete",
            {"path": path_property},
            ["path"],
            delete_file,
        ),
    ]
