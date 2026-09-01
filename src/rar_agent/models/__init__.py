"""Model-provider abstractions and adapters."""

from rar_agent.models.base import (
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolDefinition,
)
from rar_agent.models.openai_compatible import OpenAICompatibleClient
from rar_agent.models.scheduler import ModelScheduler
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.models.structured import StructuredModelGateway, StructuredOutputError

__all__ = [
    "ModelClient",
    "ModelMessage",
    "ModelRequest",
    "ModelResponse",
    "ModelScheduler",
    "ModelUsage",
    "OpenAICompatibleClient",
    "ScriptedModelClient",
    "StructuredModelGateway",
    "StructuredOutputError",
    "ToolCall",
    "ToolDefinition",
]
