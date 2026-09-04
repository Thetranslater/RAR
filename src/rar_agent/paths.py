"""Shared lexical validation for paths stored in model and artifact contracts."""

from pathlib import PurePath, PureWindowsPath


def workspace_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if (
        not normalized
        or PurePath(normalized).is_absolute()
        or PureWindowsPath(normalized).is_absolute()
        or ".." in PurePath(normalized).parts
    ):
        raise ValueError("path must be workspace-relative")
    return normalized
