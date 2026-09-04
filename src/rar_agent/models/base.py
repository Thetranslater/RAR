"""Provider-neutral request and response contracts."""

from __future__ import annotations

from pathlib import PurePath, PureWindowsPath
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonObject = dict[str, Any]


class ModelContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolDefinition(ModelContract):
    name: str
    description: str
    parameters: JsonObject


class ToolCall(ModelContract):
    call_id: str
    name: str
    arguments: JsonObject


def _workspace_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        raise ValueError("path must not be empty")
    if PurePath(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise ValueError("path must be workspace-relative")
    if ".." in PurePath(normalized).parts:
        raise ValueError("path must be workspace-relative")
    return normalized


class TextContent(ModelContract):
    type: Literal["text"] = "text"
    text: str


class LocalImageContent(ModelContract):
    type: Literal["local_image"] = "local_image"
    path: str

    _validate_path = field_validator("path")(_workspace_relative_path)


ModelContent = Annotated[
    TextContent | LocalImageContent,
    Field(discriminator="type"),
]


class ModelMessage(ModelContract):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ModelContent] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ModelUsage(ModelContract):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)


class ModelRequest(ModelContract):
    model: str
    messages: list[ModelMessage]
    temperature: float | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    provider_options: JsonObject = Field(default_factory=dict)


class ModelResponse(ModelContract):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    finish_reason: str | None = None


class ModelClient(Protocol):
    @property
    def provider(self) -> str: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...
