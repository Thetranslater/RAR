"""RAR local-service and extraction commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from rar_agent.api.app import AppRuntime, create_app
from rar_agent.domain.models import InputManifest
from rar_agent.models.base import ModelClient
from rar_agent.models.scheduler import DEFAULT_MODEL_CONCURRENCY
from rar_agent.settings import create_model_client
from rar_agent.text.tokenizer import TikTokenTokenizer
from rar_agent.workflow.dataset_build import DatasetBuildWorkflow, WorkflowConfig
from rar_agent.workflow.manga.dataset_build import (
    MangaDatasetBuildWorkflow,
    MangaWorkflowConfig,
)
from rar_agent.workflow.manga.ocr_runtime import (
    PaddleOcrWorkerManager,
    probe_paddleocr,
)

app = typer.Typer(help="RAR: Read And Retrieve role-play dialogue datasets.")
DEFAULT_PROJECT = Path.cwd()


@app.command()
def serve(
    project: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_PROJECT,
    provider: str = "deepseek",
    model: str = "deepseek-chat",
    base_url: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_model_concurrency: int = DEFAULT_MODEL_CONCURRENCY,
    max_active_chat_turns: int = 4,
    max_context_tokens: int = 32_768,
    reserved_output_tokens: int = 4_096,
    vision_provider: str = "qwen",
    vision_model: str = "qwen3.7-flash",
) -> None:
    """Start the local Web service for one Project directory."""
    client: ModelClient | None
    try:
        client = create_model_client(
            provider, base_url=base_url, project_root=project
        )
    except ValueError:
        client = None
    try:
        vision_client = create_model_client(
            vision_provider,
            project_root=project,
            timeout_seconds=180.0,
        )
    except ValueError:
        vision_client = None
    ocr_capability = probe_paddleocr(project)
    runtime = AppRuntime(
        project,
        client,
        model,
        TikTokenTokenizer(),
        vision_model_client=vision_client,
        vision_model=vision_model,
        ocr_capability=ocr_capability,
        ocr_manager=(
            PaddleOcrWorkerManager(ocr_capability)
            if ocr_capability.available
            else None
        ),
        max_model_concurrency=max_model_concurrency,
        max_active_chat_turns=max_active_chat_turns,
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    uvicorn.run(create_app(runtime), host=host, port=port)


@app.command()
def extract(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    project: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_PROJECT,
    provider: str = "deepseek",
    model: str = "deepseek-chat",
    base_url: str | None = None,
    resume: Annotated[Path | None, typer.Option()] = None,
    debug: Annotated[
        bool,
        typer.Option(help="Limit plot and dialogue extraction to five units."),
    ] = False,
) -> None:
    """Run or resume the fixed text extraction Workflow."""
    manifest_value = InputManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    client = create_model_client(
        provider, base_url=base_url, project_root=project
    )
    workflow = DatasetBuildWorkflow(
        model_client=client,
        tokenizer=TikTokenTokenizer(),
        config=WorkflowConfig(model=model, debug=debug),
    )
    result = asyncio.run(
        workflow.run(
            project,
            manifest_value,
            dataset_root=(project / resume if resume is not None else None),
        )
    )
    typer.echo(result.dataset_root)


@app.command("extract-manga")
def extract_manga(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    project: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_PROJECT,
    text_provider: str = "deepseek",
    text_model: str = "deepseek-chat",
    vision_provider: str = "qwen",
    vision_model: str = "qwen3.7-flash",
    resume: Annotated[Path | None, typer.Option()] = None,
    debug: bool = False,
    image_batch_size: int = 5,
    hard_directory_boundaries: bool = False,
    ocr: bool = False,
    ocr_batch_size: int = 4,
    ocr_threshold: int = 70,
) -> None:
    """Run or resume the fixed local manga extraction Workflow."""
    manifest_value = InputManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    text_client = create_model_client(text_provider, project_root=project)
    vision_client = create_model_client(
        vision_provider,
        project_root=project,
        timeout_seconds=180.0,
    )
    capability = probe_paddleocr(project) if ocr else None
    if capability is not None and not capability.available:
        raise typer.BadParameter(capability.reason or "OCR is unavailable", param_hint="--ocr")
    manager = PaddleOcrWorkerManager(capability) if capability is not None else None

    async def run() -> Path:
        workflow = MangaDatasetBuildWorkflow(
            vision_model_client=vision_client,
            text_model_client=text_client,
            tokenizer=TikTokenTokenizer(),
            config=MangaWorkflowConfig(
                vision_model=vision_model,
                text_model=text_model,
                debug=debug,
                image_batch_size=image_batch_size,
                hard_directory_boundaries=hard_directory_boundaries,
                ocr_enabled=ocr,
                ocr_batch_size=ocr_batch_size,
                ocr_threshold=ocr_threshold,
            ),
            ocr_runner=manager,
        )
        try:
            result = await workflow.run(
                project,
                manifest_value,
                dataset_root=(project / resume if resume is not None else None),
            )
            return result.dataset_root
        finally:
            if manager is not None:
                await manager.close()

    typer.echo(asyncio.run(run()))
