"""A deterministic model adapter for workflow and harness tests."""

from __future__ import annotations

from collections import deque

from rar_agent.models.base import ModelRequest, ModelResponse


class ScriptedModelClient:
    def __init__(
        self,
        responses: list[str | ModelResponse | Exception],
        *,
        provider: str = "scripted",
    ) -> None:
        self._responses = deque(responses)
        self._provider = provider
        self.requests: list[ModelRequest] = []

    @property
    def provider(self) -> str:
        return self._provider

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("scripted model has no response remaining")
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return ModelResponse(content=response)
        return response
