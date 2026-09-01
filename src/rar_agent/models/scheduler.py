"""Process-local provider/model concurrency limits shared across runs."""

from __future__ import annotations

import asyncio

from rar_agent.models.base import ModelClient, ModelRequest, ModelResponse


class ModelScheduler:
    def __init__(
        self,
        *,
        default_limit: int = 4,
        limits: dict[tuple[str, str], int] | None = None,
    ) -> None:
        if default_limit < 1:
            raise ValueError("default_limit must be positive")
        if any(limit < 1 for limit in (limits or {}).values()):
            raise ValueError("model limits must be positive")
        self.default_limit = default_limit
        self.limits = limits or {}
        self._semaphores: dict[tuple[str, str], asyncio.Semaphore] = {}
        self._guard = asyncio.Lock()

    async def complete(
        self, client: ModelClient, request: ModelRequest
    ) -> ModelResponse:
        key = (client.provider, request.model)
        async with self._guard:
            semaphore = self._semaphores.get(key)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.limits.get(key, self.default_limit))
                self._semaphores[key] = semaphore
        async with semaphore:
            return await client.complete(request)
