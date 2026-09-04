"""Environment-only provider configuration; API keys are never persisted."""

from __future__ import annotations

import os
from pathlib import Path

from rar_agent.models.openai_compatible import OpenAICompatibleClient


def create_model_client(
    provider: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    project_root: Path | None = None,
    timeout_seconds: float = 120.0,
) -> OpenAICompatibleClient:
    normalized = provider.lower()
    if normalized == "deepseek":
        resolved_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    elif normalized == "qwen":
        resolved_url = base_url or os.environ.get(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        resolved_key = (
            api_key
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
    else:
        raise ValueError(f"unsupported provider: {provider}")
    if not resolved_key:
        raise ValueError(f"No API key found for {provider}")
    return OpenAICompatibleClient(
        provider=normalized,
        base_url=resolved_url,
        api_key=resolved_key,
        project_root=project_root,
        timeout=timeout_seconds,
    )
