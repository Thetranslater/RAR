import json

import httpx
import pytest
from pydantic import BaseModel

from rar_agent.models.base import ModelMessage, ModelRequest
from rar_agent.models.openai_compatible import OpenAICompatibleClient
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.models.structured import StructuredModelGateway, StructuredOutputError


class Answer(BaseModel):
    value: int


async def test_gateway_repairs_fenced_json_without_spending_another_attempt() -> None:
    client = ScriptedModelClient(["```json\n{\"value\": 7,}\n```"])
    gateway = StructuredModelGateway(max_attempts=3)

    result = await gateway.generate(
        client,
        ModelRequest(model="test", messages=[ModelMessage(role="user", content="go")]),
        Answer,
    )

    assert result.value == 7
    assert len(client.requests) == 1


async def test_gateway_retries_model_validation_failure_with_shared_budget() -> None:
    client = ScriptedModelClient(['{"value":"bad"}', '{"value":9}'])
    gateway = StructuredModelGateway(max_attempts=2)

    result = await gateway.generate(
        client,
        ModelRequest(model="test", messages=[ModelMessage(role="user", content="go")]),
        Answer,
    )

    assert result.value == 9
    assert len(client.requests) == 2


async def test_gateway_reports_last_failure_after_attempt_budget() -> None:
    client = ScriptedModelClient(["not json", "still not json"])

    with pytest.raises(StructuredOutputError, match="after 2 attempts"):
        await StructuredModelGateway(max_attempts=2).generate(
            client,
            ModelRequest(model="test", messages=[ModelMessage(role="user", content="go")]),
            Answer,
        )


async def test_openai_compatible_adapter_preserves_provider_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"value": 3}'}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            provider="deepseek",
            base_url="https://example.invalid/v1",
            api_key="secret",
            http_client=http_client,
        )
        response = await client.complete(
            ModelRequest(
                model="deepseek-chat",
                messages=[ModelMessage(role="user", content="go")],
                temperature=0.2,
                provider_options={"top_p": 0.8},
            )
        )

    assert captured["top_p"] == 0.8
    assert captured["temperature"] == 0.2
    assert response.content == '{"value": 3}'
    assert response.usage.prompt_tokens == 4
