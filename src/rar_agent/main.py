"""Default ASGI application used by local development and packaged serving."""

from __future__ import annotations

import os
from pathlib import Path

from rar_agent.api.app import AppRuntime, create_app
from rar_agent.models.base import ModelClient
from rar_agent.models.scheduler import DEFAULT_MODEL_CONCURRENCY
from rar_agent.settings import create_model_client
from rar_agent.text.tokenizer import TikTokenTokenizer
from rar_agent.workflow.manga.ocr_runtime import (
    PaddleOcrWorkerManager,
    probe_paddleocr,
)


def _optional_client(
    provider: str, *, timeout_seconds: float = 120.0
) -> ModelClient | None:
    try:
        return create_model_client(
            provider,
            project_root=project_root,
            timeout_seconds=timeout_seconds,
        )
    except ValueError:
        return None


project_root = Path(os.environ.get("RAR_PROJECT_ROOT", Path.cwd())).resolve()
provider = os.environ.get("RAR_MODEL_PROVIDER", "deepseek")
model = os.environ.get("RAR_MODEL", "deepseek-chat")
max_model_concurrency = int(
    os.environ.get("RAR_MAX_MODEL_CONCURRENCY", str(DEFAULT_MODEL_CONCURRENCY))
)
max_active_chat_turns = int(os.environ.get("RAR_MAX_ACTIVE_CHAT_TURNS", "4"))
max_context_tokens = int(os.environ.get("RAR_MAX_CONTEXT_TOKENS", "32768"))
reserved_output_tokens = int(os.environ.get("RAR_RESERVED_OUTPUT_TOKENS", "4096"))
vision_provider = os.environ.get("RAR_VISION_PROVIDER", "qwen")
vision_model = os.environ.get("RAR_VISION_MODEL", "qwen3.7-flash")
ocr_capability = probe_paddleocr(project_root)
app = create_app(
    AppRuntime(
        project_root=project_root,
        model_client=_optional_client(provider),
        model=model,
        tokenizer=TikTokenTokenizer(),
        vision_model_client=_optional_client(vision_provider, timeout_seconds=180.0),
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
)
