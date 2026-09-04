"""HTTP boundary for the local RAR Web interface."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from rar_agent.agent.harness import AgentHarness, AgentRunResult, PendingAgentRun
from rar_agent.agent.tools import (
    RegisteredTool,
    ToolAuthorization,
    ToolDispatcher,
    build_workspace_tools,
)
from rar_agent.domain.models import DatasetBundle, InputManifest
from rar_agent.models.base import ModelClient, ToolDefinition
from rar_agent.models.scheduler import DEFAULT_MODEL_CONCURRENCY, ModelScheduler
from rar_agent.storage.database import (
    ChatArchivedError,
    ChatNotFoundError,
    ProjectDatabase,
    StoredChat,
)
from rar_agent.text.tokenizer import Tokenizer
from rar_agent.workflow.dataset_build import (
    DatasetBuildWorkflow,
    IncompleteStageError,
    WorkflowConfig,
)
from rar_agent.workflow.manga.dataset_build import (
    MangaDatasetBuildWorkflow,
    MangaWorkflowConfig,
)
from rar_agent.workflow.manga.ocr_runtime import (
    OcrCapability,
    PaddleOcrWorkerManager,
)
from rar_agent.workflow.manga.scanning import MangaScanError, scan_manga_folder

ChatRunStatus = Literal[
    "queued",
    "running",
    "awaiting_approval",
    "completed",
    "failed",
    "cancelled",
]
TERMINAL_CHAT_RUNS = {"completed", "failed", "cancelled"}


@dataclass(slots=True)
class AppRuntime:
    project_root: Path
    model_client: ModelClient | None
    model: str
    tokenizer: Tokenizer
    vision_model_client: ModelClient | None = None
    vision_model: str = "qwen3.7-flash"
    ocr_capability: OcrCapability | None = None
    ocr_manager: PaddleOcrWorkerManager | None = None
    max_model_concurrency: int = DEFAULT_MODEL_CONCURRENCY
    max_active_chat_turns: int = 4
    max_context_tokens: int = 32_768
    reserved_output_tokens: int = 4_096
    database: ProjectDatabase = field(init=False)
    scheduler: ModelScheduler = field(init=False)

    def __post_init__(self) -> None:
        if self.max_active_chat_turns < 1:
            raise ValueError("max_active_chat_turns must be positive")
        if self.max_context_tokens <= self.reserved_output_tokens:
            raise ValueError("max_context_tokens must exceed reserved_output_tokens")
        self.project_root = self.project_root.resolve()
        self.database = ProjectDatabase(self.project_root)
        self.scheduler = ModelScheduler(default_limit=self.max_model_concurrency)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(ApiModel):
    message: str = Field(min_length=1)
    chat_session: int | None = None
    enabled_tools: list[str] | None = None


class ChatCreateRequest(ApiModel):
    title: str = Field(default="New chat", max_length=200)


class ChatUpdateRequest(ApiModel):
    title: str | None = Field(default=None, max_length=200)
    archived: bool | None = None


class ApprovalToolResponse(ApiModel):
    name: str
    description: str
    arguments: dict[str, Any]
    effect: str
    risk: str
    reason: str


class ApprovalResponse(ApiModel):
    token: str
    expires_in_seconds: int
    tools: list[ApprovalToolResponse]


class ApprovalDecisionRequest(ApiModel):
    allowed: bool


class ChatRunAccepted(ApiModel):
    chat_session: int
    run_id: str
    status: ChatRunStatus


class ChatRunResponse(ChatRunAccepted):
    content: str = ""
    tool_calls: int = 0
    error: str | None = None
    approval: ApprovalResponse | None = None


class ChatSummaryResponse(ApiModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None
    run_id: str | None = None
    run_status: ChatRunStatus | None = None
    extraction_task: str | None = None
    extraction_status: str | None = None


class ChatMessageResponse(ApiModel):
    id: int
    role: str
    content: str
    kind: Literal["message", "event"]
    created_at: datetime
    tool_calls: list[dict[str, Any]]


class ChatMessagePage(ApiModel):
    items: list[ChatMessageResponse]
    next_before: int | None = None


class ExtractionToolRequest(ApiModel):
    workflow_type: Literal["text", "manga"] = "text"
    manifest: InputManifest
    dataset_root: str | None = None
    debug: bool = False
    text_chunk_tokens: int = Field(default=5120, ge=1)
    plot_chunk_tokens: int = Field(default=5120, ge=1)
    volume_splitters: list[str] | None = None
    chapter_splitters: list[str] | None = None
    include_headings: bool = False
    include_front_matter: bool = False
    mode: Literal["automatic", "staged"] = "automatic"
    vision_model: str = "qwen3.7-flash"
    text_model: str | None = None
    image_batch_size: int = Field(default=5, ge=1, le=10)
    hard_directory_boundaries: bool = False
    ocr_enabled: bool = False
    ocr_batch_size: int = Field(default=4, ge=1)
    ocr_threshold: int = Field(default=70, ge=0, le=100)


class ExtractionRequest(ExtractionToolRequest):
    chat_session: int | None = None


class ExtractionAccepted(ApiModel):
    task: str
    status: str
    chat_session: int


class ModelOptionResponse(ApiModel):
    provider: str
    model: str
    capabilities: list[Literal["text", "vision"]]
    available: bool
    environment_variables: list[str]
    reason: str | None = None


@dataclass(slots=True)
class _TaskState:
    chat_session: int
    workflow_run: int
    workflow_type: Literal["text", "manga"] = "text"
    status: str = "queued"
    stage: str | None = None
    dataset_root: str | None = None
    error: str | None = None
    confirmation: asyncio.Event | None = None

    def value(self) -> dict[str, Any]:
        return {
            "chat_session": self.chat_session,
            "workflow_type": self.workflow_type,
            "status": self.status,
            "stage": self.stage,
            "dataset_root": self.dataset_root,
            "error": self.error,
        }


@dataclass(slots=True)
class _ChatRunState:
    run_id: str
    chat_session: int
    status: ChatRunStatus = "queued"
    content: str = ""
    tool_calls: int = 0
    error: str | None = None
    approval: ApprovalResponse | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    def response(self) -> ChatRunResponse:
        return ChatRunResponse(
            chat_session=self.chat_session,
            run_id=self.run_id,
            status=self.status,
            content=self.content,
            tool_calls=self.tool_calls,
            error=self.error,
            approval=self.approval,
        )


@dataclass(frozen=True, slots=True)
class _PendingApprovalState:
    run: PendingAgentRun
    run_id: str
    expires_at: float


def create_app(runtime: AppRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if runtime.ocr_manager is not None:
            await runtime.ocr_manager.close()

    app = FastAPI(title="RAR Agent", version="0.1.0", lifespan=lifespan)
    task_states: dict[str, _TaskState] = {}
    active_tasks: set[asyncio.Task[None]] = set()
    chat_runs: dict[str, _ChatRunState] = {}
    active_run_by_chat: dict[int, str] = {}
    chat_tasks: dict[str, asyncio.Task[None]] = {}
    chat_slots = asyncio.Semaphore(runtime.max_active_chat_turns)
    pending_approvals: dict[str, _PendingApprovalState] = {}
    approval_timeout_seconds = 300
    terminal_run_ttl_seconds = 3600

    def _not_found(error: Exception) -> HTTPException:
        return HTTPException(404, str(error))

    def _active_extraction(
        chat_session: int,
    ) -> tuple[str, _TaskState] | None:
        return next(
            (
                (task_name, state)
                for task_name, state in reversed(task_states.items())
                if state.chat_session == chat_session
                and state.status
                not in {"completed", "failed", "incomplete", "cancelled"}
            ),
            None,
        )

    def _chat_summary(value: StoredChat) -> ChatSummaryResponse:
        run_id = active_run_by_chat.get(value.id)
        run = chat_runs.get(run_id) if run_id else None
        extraction = _active_extraction(value.id)
        return ChatSummaryResponse(
            id=value.id,
            title=value.title,
            created_at=value.created_at,
            updated_at=value.updated_at,
            archived_at=value.archived_at,
            run_id=run_id,
            run_status=run.status if run else None,
            extraction_task=extraction[0] if extraction else None,
            extraction_status=extraction[1].status if extraction else None,
        )

    def _require_chat(chat_session: int, *, active: bool = False) -> StoredChat:
        try:
            return (
                runtime.database.require_active_chat(chat_session)
                if active
                else runtime.database.get_chat(chat_session)
            )
        except ChatNotFoundError as error:
            raise _not_found(error) from error
        except ChatArchivedError as error:
            raise HTTPException(409, str(error)) from error

    def _approval_response(result: AgentRunResult, token: str) -> ApprovalResponse:
        return ApprovalResponse(
            token=token,
            expires_in_seconds=approval_timeout_seconds,
            tools=[
                ApprovalToolResponse(
                    name=approval.tool_name,
                    description=approval.description,
                    arguments=approval.arguments,
                    effect=approval.effect,
                    risk=approval.risk,
                    reason=approval.reason,
                )
                for approval in result.approvals
            ],
        )

    def _finish_run(
        state: _ChatRunState,
        status: Literal["completed", "failed", "cancelled"],
        *,
        content: str = "",
        tool_calls: int = 0,
        error: str | None = None,
        event_message: str | None = None,
    ) -> None:
        state.status = status
        state.content = content
        state.tool_calls = tool_calls
        state.error = error
        state.approval = None
        state.finished_at = time.monotonic()
        active_run_by_chat.pop(state.chat_session, None)
        if event_message:
            runtime.database.append_event(state.chat_session, event_message)

    def _apply_agent_result(state: _ChatRunState, result: AgentRunResult) -> None:
        if result.status == "completed":
            _finish_run(
                state,
                "completed",
                content=result.content,
                tool_calls=result.tool_calls,
            )
            return
        if result.pending is None:
            raise RuntimeError("awaiting approval result has no pending run")
        token = str(uuid4())
        pending_approvals[token] = _PendingApprovalState(
            result.pending,
            state.run_id,
            time.monotonic() + approval_timeout_seconds,
        )
        state.status = "awaiting_approval"
        state.content = result.content
        state.tool_calls = result.tool_calls
        state.approval = _approval_response(result, token)

    def _cleanup_runtime_state() -> None:
        now = time.monotonic()
        for token, pending in list(pending_approvals.items()):
            if pending.expires_at > now:
                continue
            pending_approvals.pop(token, None)
            state = chat_runs.get(pending.run_id)
            if state is not None and state.status == "awaiting_approval":
                _finish_run(
                    state,
                    "failed",
                    error="Tool approval expired",
                    event_message="授权请求已过期, 本次任务已停止。",
                )
        for run_id, state in list(chat_runs.items()):
            if (
                state.finished_at is not None
                and now - state.finished_at > terminal_run_ttl_seconds
            ):
                chat_runs.pop(run_id, None)
                chat_tasks.pop(run_id, None)

    def _extraction_tool(chat_session: int) -> RegisteredTool:
        async def start_text_extraction(arguments: dict[str, Any]) -> dict[str, Any]:
            value = ExtractionToolRequest.model_validate(arguments)
            payload = value.model_dump(mode="python")
            payload["workflow_type"] = "text"
            accepted = submit_extraction(
                ExtractionRequest(
                    **payload,
                    chat_session=chat_session,
                )
            )
            return accepted.model_dump(mode="json")

        return RegisteredTool(
            definition=ToolDefinition(
                name="start_text_extraction",
                description=(
                    "Start or resume RAR's fixed text Dataset workflow after resolving a "
                    "local InputManifest. Use Project-relative resource paths."
                ),
                parameters=ExtractionToolRequest.model_json_schema(),
            ),
            effect="write",
            handler=start_text_extraction,
            authorization=ToolAuthorization(
                mode="always",
                risk="medium",
                reason="该操作会启动 Workflow 并在指定 Dataset 目录中生成文件。",
            ),
        )

    def _manga_extraction_tool(chat_session: int) -> RegisteredTool:
        async def start_manga_extraction(arguments: dict[str, Any]) -> dict[str, Any]:
            value = ExtractionToolRequest.model_validate(arguments)
            payload = value.model_dump(mode="python")
            payload["workflow_type"] = "manga"
            accepted = submit_extraction(
                ExtractionRequest(
                    **payload,
                    chat_session=chat_session,
                )
            )
            return accepted.model_dump(mode="json")

        return RegisteredTool(
            definition=ToolDefinition(
                name="start_manga_extraction",
                description=(
                    "Start or resume RAR's fixed manga Dataset workflow for one "
                    "Project-relative image folder."
                ),
                parameters=ExtractionToolRequest.model_json_schema(),
            ),
            effect="write",
            handler=start_manga_extraction,
            authorization=ToolAuthorization(
                mode="always",
                risk="medium",
                reason="This starts a Workflow that writes manga Dataset artifacts.",
            ),
        )

    def create_harness(chat_session: int) -> AgentHarness:
        model_client = runtime.model_client
        if model_client is None:
            raise HTTPException(503, "No model provider is configured")
        return AgentHarness(
            model_client=model_client,
            model=runtime.model,
            dispatcher=ToolDispatcher(
                runtime.project_root,
                [
                    *build_workspace_tools(runtime.project_root),
                    _extraction_tool(chat_session),
                    _manga_extraction_tool(chat_session),
                ],
            ),
            database=runtime.database,
            scheduler=runtime.scheduler,
            tokenizer=runtime.tokenizer,
            max_context_tokens=runtime.max_context_tokens,
            reserved_output_tokens=runtime.reserved_output_tokens,
        )

    def _schedule_chat_run(
        state: _ChatRunState,
        operation: Callable[[], Awaitable[AgentRunResult]],
    ) -> None:
        async def execute() -> None:
            try:
                state.status = "queued"
                async with chat_slots:
                    state.status = "running"
                    result = await operation()
                    _apply_agent_result(state, result)
            except asyncio.CancelledError:
                _finish_run(
                    state,
                    "cancelled",
                    event_message="本次任务已由用户停止。",
                )
            except Exception as error:
                _finish_run(
                    state,
                    "failed",
                    error=str(error),
                    event_message=f"任务失败: {error}",
                )

        task = asyncio.create_task(execute(), name=f"rar-chat-{state.run_id}")
        chat_tasks[state.run_id] = task
        task.add_done_callback(lambda _task: chat_tasks.pop(state.run_id, None))

    def submit_extraction(request: ExtractionRequest) -> ExtractionAccepted:
        model_client = runtime.model_client
        if model_client is None:
            raise HTTPException(503, "No model provider is configured")
        if request.workflow_type == "manga" and runtime.vision_model_client is None:
            raise HTTPException(503, "No vision model provider is configured")
        if request.ocr_enabled and (
            runtime.ocr_capability is None or not runtime.ocr_capability.available
        ):
            reason = (
                runtime.ocr_capability.reason
                if runtime.ocr_capability is not None
                else "OCR capability was not probed"
            )
            raise HTTPException(422, reason)
        if request.chat_session is None:
            chat_session = runtime.database.create_chat(f"提取: {request.manifest.name}")
        else:
            _require_chat(request.chat_session, active=True)
            chat_session = request.chat_session
        if _active_extraction(chat_session) is not None:
            raise HTTPException(409, "This chat already has an active extraction")
        task_name = str(uuid4())
        dataset_hint = request.dataset_root or request.manifest.name
        workflow_run = runtime.database.create_workflow_run(
            dataset_hint, chat_session=chat_session
        )
        state = _TaskState(
            chat_session=chat_session,
            workflow_run=workflow_run,
            workflow_type=request.workflow_type,
        )
        task_states[task_name] = state
        runtime.database.append_event(
            chat_session, f"已开始提取数据集: {request.manifest.name}"
        )

        async def stage_callback(
            stage: str, phase: Literal["started", "completed"], _artifact: Path
        ) -> None:
            state.stage = stage
            if phase == "started":
                state.status = "running"
                runtime.database.update_workflow_run(
                    workflow_run, status="running", stage=stage
                )
                return
            if request.mode == "staged":
                state.status = "awaiting_confirmation"
            runtime.database.update_workflow_run(
                workflow_run, status=state.status, stage=stage
            )
            if request.mode == "staged":
                state.confirmation = asyncio.Event()
                await state.confirmation.wait()
                state.confirmation = None
                state.status = "running"
                runtime.database.update_workflow_run(
                    workflow_run, status="running", stage=stage
                )

        async def execute() -> None:
            state.status = "running"
            runtime.database.update_workflow_run(workflow_run, status="running")
            try:
                target_root = (
                    runtime.project_root / request.dataset_root
                    if request.dataset_root
                    else None
                )
                workflow: MangaDatasetBuildWorkflow | DatasetBuildWorkflow
                if request.workflow_type == "manga":
                    vision_client = runtime.vision_model_client
                    if vision_client is None:  # guarded before scheduling
                        raise RuntimeError("No vision model provider is configured")
                    workflow = MangaDatasetBuildWorkflow(
                        vision_model_client=vision_client,
                        text_model_client=model_client,
                        tokenizer=runtime.tokenizer,
                        config=MangaWorkflowConfig(
                            vision_model=request.vision_model,
                            text_model=request.text_model or runtime.model,
                            debug=request.debug,
                            image_batch_size=request.image_batch_size,
                            hard_directory_boundaries=(
                                request.hard_directory_boundaries
                            ),
                            ocr_enabled=request.ocr_enabled,
                            ocr_batch_size=request.ocr_batch_size,
                            ocr_threshold=request.ocr_threshold,
                        ),
                        stage_callback=stage_callback,
                        scheduler=runtime.scheduler,
                        ocr_runner=runtime.ocr_manager,
                    )
                else:
                    workflow = DatasetBuildWorkflow(
                        model_client=model_client,
                        tokenizer=runtime.tokenizer,
                        config=WorkflowConfig(
                            model=runtime.model,
                            debug=request.debug,
                            text_chunk_tokens=request.text_chunk_tokens,
                            plot_chunk_tokens=request.plot_chunk_tokens,
                            volume_splitters=(
                                tuple(request.volume_splitters)
                                if request.volume_splitters is not None
                                else None
                            ),
                            chapter_splitters=(
                                tuple(request.chapter_splitters)
                                if request.chapter_splitters is not None
                                else None
                            ),
                            include_headings=request.include_headings,
                            include_front_matter=request.include_front_matter,
                        ),
                        stage_callback=stage_callback,
                        scheduler=runtime.scheduler,
                    )
                result = await workflow.run(
                    runtime.project_root,
                    request.manifest,
                    dataset_root=target_root,
                )
                state.status = "completed"
                state.dataset_root = result.dataset_root.relative_to(
                    runtime.project_root
                ).as_posix()
                runtime.database.update_workflow_run(
                    workflow_run,
                    status="completed",
                    stage=state.stage,
                    dataset_path=state.dataset_root,
                )
                runtime.database.append_event(
                    chat_session, f"数据集提取完成: {state.dataset_root}"
                )
            except IncompleteStageError as error:
                state.status = "incomplete"
                state.stage = error.stage
                state.dataset_root = error.dataset_root.relative_to(
                    runtime.project_root
                ).as_posix()
                state.error = str(error)
                runtime.database.update_workflow_run(
                    workflow_run,
                    status="incomplete",
                    stage=error.stage,
                    dataset_path=state.dataset_root,
                    error=state.error,
                )
                runtime.database.append_event(
                    chat_session, f"数据集提取不完整: {error}"
                )
            except Exception as error:
                state.status = "failed"
                state.error = str(error)
                runtime.database.update_workflow_run(
                    workflow_run,
                    status="failed",
                    stage=state.stage,
                    error=state.error,
                )
                runtime.database.append_event(
                    chat_session, f"数据集提取失败: {error}"
                )

        task = asyncio.create_task(execute(), name=f"rar-extraction-{task_name}")
        active_tasks.add(task)
        task.add_done_callback(active_tasks.discard)
        return ExtractionAccepted(
            task=task_name, status=state.status, chat_session=chat_session
        )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/project")
    async def project() -> dict[str, Any]:
        ocr = runtime.ocr_capability
        return {
            "name": runtime.project_root.name,
            "path": str(runtime.project_root),
            "model_ready": runtime.model_client is not None,
            "vision_model_ready": runtime.vision_model_client is not None,
            "ocr": (
                ocr.model_dump(mode="json")
                if ocr is not None
                else {
                    "available": False,
                    "cuda": False,
                    "model_dir": "",
                    "reason": "OCR capability was not probed",
                }
            ),
        }

    @app.get("/api/models", response_model=list[ModelOptionResponse])
    async def models() -> list[ModelOptionResponse]:
        text_provider = (
            runtime.model_client.provider if runtime.model_client is not None else "deepseek"
        )
        return [
            ModelOptionResponse(
                provider=text_provider,
                model=runtime.model,
                capabilities=["text"],
                available=runtime.model_client is not None,
                environment_variables=(
                    ["QWEN_API_KEY", "DASHSCOPE_API_KEY"]
                    if text_provider == "qwen"
                    else ["DEEPSEEK_API_KEY"]
                ),
                reason=(
                    None
                    if runtime.model_client is not None
                    else "The text provider API key is unavailable"
                ),
            ),
            ModelOptionResponse(
                provider="qwen",
                model=runtime.vision_model,
                capabilities=["text", "vision"],
                available=runtime.vision_model_client is not None,
                environment_variables=["QWEN_API_KEY", "DASHSCOPE_API_KEY"],
                reason=(
                    None
                    if runtime.vision_model_client is not None
                    else "Set QWEN_API_KEY or DASHSCOPE_API_KEY"
                ),
            ),
        ]

    @app.get("/api/resources/manga/preview")
    async def preview_manga_resource(
        path: str = Query(min_length=1),
        batch_size: int = Query(default=5, ge=1, le=10),
        debug: bool = False,
    ) -> dict[str, Any]:
        try:
            scan = scan_manga_folder(
                runtime.project_root,
                path,
                batch_size=batch_size,
                max_batches=5 if debug else None,
            )
        except MangaScanError as error:
            raise HTTPException(422, str(error)) from error
        return {
            "path": scan.resource_path,
            "image_count": len(scan.pages),
            "batch_count": len(scan.batches),
            "first_paths": [page.path for page in scan.pages[:10]],
            "skipped": [],
            "errors": [],
        }

    @app.get("/api/chats", response_model=list[ChatSummaryResponse])
    async def chats(archived: bool = False) -> list[ChatSummaryResponse]:
        _cleanup_runtime_state()
        return [
            _chat_summary(value)
            for value in runtime.database.list_chats(archived=archived)
        ]

    @app.post("/api/chats", response_model=ChatSummaryResponse, status_code=201)
    async def create_chat(request: ChatCreateRequest) -> ChatSummaryResponse:
        chat_session = runtime.database.create_chat(request.title)
        return _chat_summary(runtime.database.get_chat(chat_session))

    @app.get("/api/chats/{chat_session}", response_model=ChatSummaryResponse)
    async def chat_detail(chat_session: int) -> ChatSummaryResponse:
        _cleanup_runtime_state()
        return _chat_summary(_require_chat(chat_session))

    @app.patch("/api/chats/{chat_session}", response_model=ChatSummaryResponse)
    async def update_chat(
        chat_session: int, request: ChatUpdateRequest
    ) -> ChatSummaryResponse:
        _require_chat(chat_session)
        if (
            request.archived is True
            and (
                chat_session in active_run_by_chat
                or _active_extraction(chat_session) is not None
            )
        ):
            raise HTTPException(409, "Stop active tasks before archiving this Chat")
        try:
            if request.title is not None:
                runtime.database.rename_chat(chat_session, request.title)
            if request.archived is not None:
                runtime.database.set_chat_archived(chat_session, request.archived)
            return _chat_summary(runtime.database.get_chat(chat_session))
        except ChatNotFoundError as error:
            raise _not_found(error) from error

    @app.delete("/api/chats/{chat_session}", status_code=204)
    async def delete_chat(chat_session: int) -> None:
        _require_chat(chat_session)
        if (
            chat_session in active_run_by_chat
            or _active_extraction(chat_session) is not None
        ):
            raise HTTPException(409, "Stop active tasks before deleting this Chat")
        try:
            runtime.database.delete_chat(chat_session)
        except ChatNotFoundError as error:
            raise _not_found(error) from error

    @app.get("/api/chats/{chat_session}/messages", response_model=ChatMessagePage)
    async def messages(
        chat_session: int,
        before: int | None = None,
        limit: int = Query(default=100, ge=1, le=200),
    ) -> ChatMessagePage:
        _require_chat(chat_session)
        values = runtime.database.list_messages(
            chat_session, before=before, limit=limit + 1
        )
        has_more = len(values) > limit
        if has_more:
            values = values[1:]
        return ChatMessagePage(
            items=[
                ChatMessageResponse(
                    id=message.id,
                    role=message.role,
                    content=message.content,
                    kind=message.kind,
                    created_at=message.created_at,
                    tool_calls=[
                        call.model_dump(mode="json") for call in message.tool_calls
                    ],
                )
                for message in values
            ],
            next_before=values[0].id if has_more and values else None,
        )

    @app.post("/api/chat", response_model=ChatRunAccepted, status_code=202)
    async def chat(request: ChatRequest) -> ChatRunAccepted:
        _cleanup_runtime_state()
        if runtime.model_client is None:
            raise HTTPException(503, "No model provider is configured")
        if request.chat_session is None:
            chat_session = runtime.database.create_chat(request.message)
        else:
            _require_chat(request.chat_session, active=True)
            chat_session = request.chat_session
        extraction = _active_extraction(chat_session)
        if extraction is not None and extraction[1].status != "awaiting_confirmation":
            raise HTTPException(
                409,
                "This chat cannot run the Agent while its extraction is running",
            )
        if chat_session in active_run_by_chat:
            raise HTTPException(409, "This chat already has an active run")
        runtime.database.append_message(chat_session, "user", request.message)
        state = _ChatRunState(str(uuid4()), chat_session)
        chat_runs[state.run_id] = state
        active_run_by_chat[chat_session] = state.run_id
        enabled_tools = set(request.enabled_tools) if request.enabled_tools else None
        harness = create_harness(chat_session)
        _schedule_chat_run(
            state,
            lambda: harness.run(
                request.message,
                chat_session=chat_session,
                enabled_tools=enabled_tools,
                record_user_message=False,
            ),
        )
        return ChatRunAccepted(
            chat_session=chat_session,
            run_id=state.run_id,
            status=state.status,
        )

    @app.get(
        "/api/chats/{chat_session}/runs/{run_id}",
        response_model=ChatRunResponse,
    )
    async def chat_run(chat_session: int, run_id: str) -> ChatRunResponse:
        _cleanup_runtime_state()
        _require_chat(chat_session)
        state = chat_runs.get(run_id)
        if state is None or state.chat_session != chat_session:
            raise HTTPException(404, "Chat run not found")
        return state.response()

    @app.post(
        "/api/chats/{chat_session}/runs/{run_id}/cancel",
        response_model=ChatRunResponse,
    )
    async def cancel_chat_run(chat_session: int, run_id: str) -> ChatRunResponse:
        _cleanup_runtime_state()
        _require_chat(chat_session)
        state = chat_runs.get(run_id)
        if state is None or state.chat_session != chat_session:
            raise HTTPException(404, "Chat run not found")
        if state.status in TERMINAL_CHAT_RUNS:
            return state.response()
        if state.approval is not None:
            pending_approvals.pop(state.approval.token, None)
        task = chat_tasks.get(run_id)
        if task is not None:
            task.cancel()
            await asyncio.sleep(0)
        elif state.status == "awaiting_approval":
            _finish_run(
                state, "cancelled", event_message="本次任务已由用户停止。"
            )
        return state.response()

    @app.post("/api/approvals/{token}", response_model=ChatRunAccepted, status_code=202)
    async def resolve_approval(
        token: str, request: ApprovalDecisionRequest
    ) -> ChatRunAccepted:
        _cleanup_runtime_state()
        pending = pending_approvals.pop(token, None)
        if pending is None:
            raise HTTPException(404, "Approval request not found")
        if pending.expires_at <= time.monotonic():
            raise HTTPException(410, "Approval request expired")
        state = chat_runs.get(pending.run_id)
        if state is None or state.status != "awaiting_approval":
            raise HTTPException(409, "Chat run is not waiting for approval")
        state.approval = None
        harness = create_harness(state.chat_session)
        _schedule_chat_run(
            state,
            lambda: harness.resume(pending.run, approved=request.allowed),
        )
        return ChatRunAccepted(
            chat_session=state.chat_session,
            run_id=state.run_id,
            status=state.status,
        )

    @app.get("/api/datasets")
    async def datasets() -> list[dict[str, Any]]:
        return _list_datasets(runtime.project_root)

    @app.get("/api/datasets/{directory}/dataset")
    async def dataset(directory: str) -> dict[str, Any]:
        path = _dataset_file(runtime.project_root, directory)
        if not path.is_file():
            raise HTTPException(404, "Dataset not found")
        return DatasetBundle.model_validate_json(
            path.read_text(encoding="utf-8")
        ).model_dump(mode="json")

    @app.post("/api/extractions", response_model=ExtractionAccepted, status_code=202)
    async def extract(request: ExtractionRequest) -> ExtractionAccepted:
        if (
            request.chat_session is not None
            and request.chat_session in active_run_by_chat
        ):
            raise HTTPException(409, "This chat already has an active Agent run")
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
        if state.chat_session in active_run_by_chat:
            raise HTTPException(
                409,
                "Finish the active Agent run before continuing the extraction",
            )
        state.status = "running"
        runtime.database.update_workflow_run(
            state.workflow_run, status="running", stage=state.stage
        )
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
            values.append(
                {"directory": path.parent.name, "name": path.parent.name, "valid": False}
            )
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
