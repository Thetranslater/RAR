"""Stable extension seams reserved beyond the V1 text core."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rar_agent.agent.tools import RegisteredTool
from rar_agent.domain.models import DatasetBundle, InputResource, PlotsDocument
from rar_agent.export.sharegpt import ShareGPTExportReport


@runtime_checkable
class ResourceAdapter(Protocol):
    media_type: str

    async def prepare(self, project_root: Path, resource: InputResource) -> Any: ...


@runtime_checkable
class ToolProvider(Protocol):
    def tools(self, project_root: Path) -> list[RegisteredTool]: ...


@runtime_checkable
class Exporter(Protocol):
    def export(
        self,
        bundle: DatasetBundle,
        plots: PlotsDocument,
        output_dir: Path,
        *,
        system_template: str,
    ) -> ShareGPTExportReport: ...


@runtime_checkable
class ReportGenerator(Protocol):
    def generate(self, bundle: DatasetBundle, output_dir: Path) -> Path: ...


@runtime_checkable
class WorkflowExtension(Protocol):
    name: str

    async def run(self, project_root: Path, configuration: dict[str, Any]) -> Any: ...
