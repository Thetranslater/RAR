"""HTTP boundary for the local RAR Web interface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from rar_agent.agent.harness import AgentHarness
from rar_agent.agent.tools import RegisteredTool, ToolDispatcher, build_workspace_tools
from rar_agent.domain.models import DatasetBundle, InputManifest
from rar_agent.models.base import ModelClient, ToolDefinition
from rar_agent.models.scheduler import ModelScheduler
from rar_agent.storage.database import ProjectDatabase
from rar_agent.text.tokenizer import Tokenizer
from rar_agent.workflow.dataset_build import (
    DatasetBuildWorkflow,
    IncompleteStageError,
    WorkflowConfig,
)


@dataclass(slots=True)
class AppRuntime:
    project_root: Path
    model_client: ModelClient | None
    model: str
    tokenizer: Tokenizer
    database: ProjectDatabase = field(init=False)
    scheduler: ModelScheduler = field(init=False)

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.database = ProjectDatabase(self.project_root)
        self.scheduler = ModelScheduler()


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(ApiModel):
    message: str = Field(min_length=1)
    chat_session: int | None = None
    enabled_tools: list[str] | None = None


class ChatResponse(ApiModel):
    chat_session: int
    content: str
    tool_calls: int


class ExtractionRequest(ApiModel):
    manifest: InputManifest
    dataset_root: str | None = None
    text_chunk_tokens: int = Field(default=5120, ge=1)
    plot_chunk_tokens: int = Field(default=5120, ge=1)
    max_concurrency: int = Field(default=4, ge=1)
    mode: Literal["automatic", "staged"] = "automatic"


class ExtractionAccepted(ApiModel):
    task: str
    status: str


@dataclass(slots=True)
class _TaskState:
    status: str = "queued"
    stage: str | None = None
    dataset_root: str | None = None
    error: str | None = None
    confirmation: asyncio.Event | None = None

    def value(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage": self.stage,
            "dataset_root": self.dataset_root,
            "error": self.error,
        }


def create_app(runtime: AppRuntime) -> FastAPI:
    app = FastAPI(title="RAR Agent", version="0.1.0")
    task_states: dict[str, _TaskState] = {}
    active_tasks: set[asyncio.Task[None]] = set()

    def submit_extraction(request: ExtractionRequest) -> ExtractionAccepted:
        model_client = runtime.model_client
        if model_client is None:
            raise HTTPException(503, "No model provider is configured")
        task_name = str(uuid4())
        state = _TaskState()
        task_states[task_name] = state

        async def stage_callback(stage: str, _artifact: Path) -> None:
            state.stage = stage
            if request.mode == "staged":
                state.status = "awaiting_confirmation"
                state.confirmation = asyncio.Event()
                await state.confirmation.wait()
                state.confirmation = None
                state.status = "running"

        async def execute() -> None:
            state.status = "running"
            workflow = DatasetBuildWorkflow(
                model_client=model_client,
                tokenizer=runtime.tokenizer,
                config=WorkflowConfig(
                    model=runtime.model,
                    text_chunk_tokens=request.text_chunk_tokens,
                    plot_chunk_tokens=request.plot_chunk_tokens,
                    max_concurrency=request.max_concurrency,
                ),
                stage_callback=stage_callback,
                scheduler=runtime.scheduler,
            )
            try:
                result = await workflow.run(
                    runtime.project_root,
                    request.manifest,
                    dataset_root=(
                        runtime.project_root / request.dataset_root
                        if request.dataset_root
                        else None
                    ),
                )
                state.status = "completed"
                state.dataset_root = result.dataset_root.relative_to(
                    runtime.project_root
                ).as_posix()
            except IncompleteStageError as error:
                state.status = "incomplete"
                state.stage = error.stage
                state.dataset_root = error.dataset_root.relative_to(
                    runtime.project_root
                ).as_posix()
                state.error = str(error)
            except Exception as error:
                state.status = "failed"
                state.error = str(error)

        task = asyncio.create_task(execute(), name=f"rar-extraction-{task_name}")
        active_tasks.add(task)
        task.add_done_callback(active_tasks.discard)
        return ExtractionAccepted(task=task_name, status=state.status)

    async def start_text_extraction(arguments: dict[str, Any]) -> dict[str, Any]:
        accepted = submit_extraction(ExtractionRequest.model_validate(arguments))
        return accepted.model_dump(mode="json")

    extraction_tool = RegisteredTool(
        definition=ToolDefinition(
            name="start_text_extraction",
            description=(
                "Start or resume RAR's fixed text Dataset workflow after resolving a local "
                "InputManifest. Use Project-relative resource paths."
            ),
            parameters=ExtractionRequest.model_json_schema(),
        ),
        effect="write",
        handler=start_text_extraction,
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/project")
    async def project() -> dict[str, Any]:
        return {
            "name": runtime.project_root.name,
            "path": str(runtime.project_root),
            "datasets": _list_datasets(runtime.project_root),
            "model_ready": runtime.model_client is not None,
        }

    @app.get("/api/datasets")
    async def datasets() -> list[dict[str, Any]]:
        return _list_datasets(runtime.project_root)

    @app.get("/api/datasets/{directory}/dataset")
    async def dataset(directory: str) -> dict[str, Any]:
        path = _dataset_file(runtime.project_root, directory)
        if not path.is_file():
            raise HTTPException(404, "Dataset not found")
        return DatasetBundle.model_validate_json(path.read_text(encoding="utf-8")).model_dump(
            mode="json"
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if runtime.model_client is None:
            raise HTTPException(503, "No model provider is configured")
        harness = AgentHarness(
            model_client=runtime.model_client,
            model=runtime.model,
            dispatcher=ToolDispatcher(
                runtime.project_root,
                [*build_workspace_tools(runtime.project_root), extraction_tool],
            ),
            database=runtime.database,
            scheduler=runtime.scheduler,
        )
        result = await harness.run(
            request.message,
            chat_session=request.chat_session,
            enabled_tools=(set(request.enabled_tools) if request.enabled_tools else None),
        )
        return ChatResponse(
            chat_session=result.chat_session,
            content=result.content,
            tool_calls=result.tool_calls,
        )

    @app.get("/api/chats/{chat_session}/messages")
    async def messages(chat_session: int) -> list[dict[str, Any]]:
        return [
            {
                "role": message.role,
                "content": message.content,
                "tool_calls": [
                    call.model_dump(mode="json") for call in message.tool_calls
                ],
            }
            for message in runtime.database.list_messages(chat_session)
        ]

    @app.post("/api/extractions", response_model=ExtractionAccepted, status_code=202)
    async def extract(request: ExtractionRequest) -> ExtractionAccepted:
        return submit_extraction(request)

    @app.get("/api/extractions/{task_name}")
    async def extraction_status(task_name: str) -> dict[str, Any]:
        state = task_states.get(task_name)
        if state is None:
            raise HTTPException(404, "Extraction task not found")
        return {"task": task_name, **state.value()}

    @app.post("/api/extractions/{task_name}/confirm")
    async def confirm_extraction(task_name: str) -> dict[str, str]:
        state = task_states.get(task_name)
        if state is None:
            raise HTTPException(404, "Extraction task not found")
        if state.status != "awaiting_confirmation" or state.confirmation is None:
            raise HTTPException(409, "Extraction is not waiting for confirmation")
        state.confirmation.set()
        return {"status": "confirmed"}

    packaged_frontend = Path(__file__).parents[1] / "web_dist"
    development_frontend = Path(__file__).parents[3] / "web" / "dist"
    frontend = (
        packaged_frontend if packaged_frontend.is_dir() else development_frontend
    )
    if frontend.is_dir():
        assets = frontend / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def frontend_fallback(path: str) -> FileResponse:
            candidate = (frontend / path).resolve()
            try:
                candidate.relative_to(frontend.resolve())
            except ValueError:
                candidate = frontend / "index.html"
            if not candidate.is_file():
                candidate = frontend / "index.html"
            return FileResponse(candidate)

    return app


def _list_datasets(project_root: Path) -> list[dict[str, Any]]:
    datasets_root = project_root / "datasets"
    if not datasets_root.is_dir():
        return []
    values: list[dict[str, Any]] = []
    for path in sorted(datasets_root.glob("*/dataset.json")):
        try:
            bundle = DatasetBundle.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            values.append({"directory": path.parent.name, "name": path.parent.name, "valid": False})
            continue
        values.append(
            {
                "directory": path.parent.name,
                "name": bundle.name,
                "valid": True,
                "characters": len(bundle.characters),
                "conversations": len(bundle.conversations),
            }
        )
    return values


def _dataset_file(project_root: Path, directory: str) -> Path:
    path = (project_root / "datasets" / directory / "dataset.json").resolve()
    datasets_root = (project_root / "datasets").resolve()
    try:
        path.relative_to(datasets_root)
    except ValueError as error:
        raise HTTPException(400, "Invalid Dataset path") from error
    return path
