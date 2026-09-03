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
) -> None:
    """Start the local Web service for one Project directory."""
    client: ModelClient | None
    try:
        client = create_model_client(provider, base_url=base_url)
    except ValueError:
        client = None
    runtime = AppRuntime(
        project,
        client,
        model,
        TikTokenTokenizer(),
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
    client = create_model_client(provider, base_url=base_url)
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
