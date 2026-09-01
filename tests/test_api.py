from pathlib import Path

from fastapi.testclient import TestClient

from rar_agent.api.app import AppRuntime, create_app
from rar_agent.models.base import ModelResponse, ToolCall
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.text.tokenizer import CharacterTokenizer


def test_api_exposes_project_and_chat_through_local_service(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "demo"
    dataset.mkdir(parents=True)
    (dataset / "dataset.json").write_text(
        '{"schema_version":1,"name":"Demo","meta":{},"resources":[],"characters":[],"conversations":[]}',
        encoding="utf-8",
    )
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=ScriptedModelClient(["项目中有一个数据集。"]),
        model="test-model",
        tokenizer=CharacterTokenizer(),
    )

    with TestClient(create_app(runtime)) as client:
        health = client.get("/api/health")
        project = client.get("/api/project")
        response = client.post("/api/chat", json={"message": "有哪些数据集？"})  # noqa: RUF001

    assert health.json() == {"status": "ok"}
    assert project.json()["datasets"][0]["name"] == "Demo"
    assert response.status_code == 200
    assert response.json()["content"] == "项目中有一个数据集。"
    assert response.json()["chat_session"] > 0
    assert "start_text_extraction" in {
        tool.name for tool in runtime.model_client.requests[0].tools
    }


def test_api_returns_graceful_response_when_agent_reaches_round_limit(
    tmp_path: Path,
) -> None:
    responses = [
        ModelResponse(
            content="I need to inspect one more thing.",
            tool_calls=[
                ToolCall(
                    call_id=f"call-{index}",
                    name="list_files",
                    arguments={"path": "."},
                )
            ],
        )
        for index in range(12)
    ]
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=ScriptedModelClient(responses),
        model="test-model",
        tokenizer=CharacterTokenizer(),
    )

    with TestClient(create_app(runtime), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/chat",
            json={"message": "检查当前项目中已有的数据集和中间产物。"},
        )

    assert response.status_code == 200
    assert "工具调用轮数上限" in response.json()["content"]
    assert response.json()["tool_calls"] == 12
