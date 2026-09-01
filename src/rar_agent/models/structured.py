"""Bounded structured-output parsing shared by every model provider."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from rar_agent.models.base import ModelClient, ModelRequest
from rar_agent.models.scheduler import ModelScheduler

SchemaT = TypeVar("SchemaT", bound=BaseModel)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class StructuredOutputError(RuntimeError):
    pass


class StructuredModelGateway:
    def __init__(self, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts

    async def generate(
        self,
        client: ModelClient,
        request: ModelRequest,
        schema: type[SchemaT],
        *,
        validator: Callable[[SchemaT], SchemaT] | None = None,
        scheduler: ModelScheduler | None = None,
    ) -> SchemaT:
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                response = (
                    await scheduler.complete(client, request)
                    if scheduler is not None
                    else await client.complete(request)
                )
                if response.content is None:
                    raise ValueError("model returned no textual content")
                value = self._parse_json(response.content)
                parsed = schema.model_validate(value)
                return validator(parsed) if validator is not None else parsed
            except Exception as error:
                last_error = error
        raise StructuredOutputError(
            f"structured output failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _parse_json(content: str) -> object:
        candidate = content.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()

        object_start = candidate.find("{")
        array_start = candidate.find("[")
        starts = [index for index in (object_start, array_start) if index >= 0]
        if starts:
            start = min(starts)
            closer = "}" if candidate[start] == "{" else "]"
            end = candidate.rfind(closer)
            if end >= start:
                candidate = candidate[start : end + 1]
        candidate = _TRAILING_COMMA.sub(r"\1", candidate)
        return json.loads(candidate)
