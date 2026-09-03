"""Default ASGI application used by local development and packaged serving."""

from __future__ import annotations

import os
from pathlib import Path

from rar_agent.api.app import AppRuntime, create_app
from rar_agent.models.base import ModelClient
from rar_agent.models.scheduler import DEFAULT_MODEL_CONCURRENCY
from rar_agent.settings import create_model_client
from rar_agent.text.tokenizer import TikTokenTokenizer


def _optional_client(provider: str) -> ModelClient | None:
    try:
        return create_model_client(provider)
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
app = create_app(
    AppRuntime(
        project_root=project_root,
        model_client=_optional_client(provider),
        model=model,
        tokenizer=TikTokenTokenizer(),
        max_model_concurrency=max_model_concurrency,
        max_active_chat_turns=max_active_chat_turns,
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
)
