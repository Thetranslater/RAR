"""OpenAI-compatible adapter used initially by DeepSeek and Qwen."""

from __future__ import annotations

import json
from typing import Any

import httpx

from rar_agent.models.base import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
)


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http_client = http_client
        self._timeout = timeout

    @property
    def provider(self) -> str:
        return self._provider

    async def complete(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._message_payload(message) for message in request.messages],
            **request.provider_options,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]

        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._http_client is not None:
            response = await self._http_client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        usage = body.get("usage", {})
        return ModelResponse(
            content=message.get("content"),
            tool_calls=[self._tool_call(value) for value in message.get("tool_calls", [])],
            usage=ModelUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
        )

    @staticmethod
    def _message_payload(message: ModelMessage) -> dict[str, Any]:
        value = message.model_dump(exclude_none=True, exclude={"tool_calls"})
        if message.tool_calls:
            value["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        return value

    @staticmethod
    def _tool_call(value: dict[str, Any]) -> ToolCall:
        function = value["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return ToolCall(
            call_id=value["id"],
            name=function["name"],
            arguments=arguments,
        )
