import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from rar_agent.api.app import AppRuntime, create_app
from rar_agent.models.base import ModelRequest, ModelResponse, ToolCall
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.text.tokenizer import CharacterTokenizer


def _wait_run(
    client: TestClient,
    accepted: dict[str, Any],
    *statuses: str,
) -> dict[str, Any]:
    expected = set(statuses or ("completed", "failed", "cancelled"))
    path = (
        f"/api/chats/{accepted['chat_session']}/runs/{accepted['run_id']}"
    )
    for _ in range(200):
        value = client.get(path)
        assert value.status_code == 200
        payload = value.json()
        if payload["status"] in expected:
            return payload
        time.sleep(0.005)
    raise AssertionError(f"run did not reach {expected}")


def _wait_extraction(
    client: TestClient,
    task: str,
    *,
    status: str,
    stage: str | None = None,
) -> dict[str, Any]:
    for _ in range(300):
        response = client.get(f"/api/extractions/{task}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] == status and (
            stage is None or payload["stage"] == stage
        ):
            return payload
        time.sleep(0.005)
    raise AssertionError(f"extraction did not reach {status=} {stage=}")


def _extraction_request(*, mode: str = "automatic") -> dict[str, Any]:
    return {
        "mode": mode,
        "manifest": {
            "name": "测试作品",
            "meta": {},
            "resources": [
                {
                    "path": "resources/book.txt",
                    "resource_type": "text",
                    "display_name": "测试作品",
                    "narrative_order": 0,
                    "meta": {},
                }
            ],
        },
    }


def _workflow_responses() -> list[str]:
    source = "甲说：“你好。”"  # noqa: RUF001
    return [
        json.dumps(
            {
                "characters": [],
                "plots": [[source, source]],
                "state": "finished",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {"utterances": [{"speaker": "甲", "content": "你好。"}]},
            ensure_ascii=False,
        ),
    ]


class SlowWorkflowClient(ScriptedModelClient):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(0.15)
        return await super().complete(request)


class PausedWorkflowClient:
    def __init__(self) -> None:
        self.workflow = ScriptedModelClient(_workflow_responses())
        self.chat_calls = 0

    @property
    def provider(self) -> str:
        return "scripted"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if not request.tools:
            return await self.workflow.complete(request)
        self.chat_calls += 1
        if self.chat_calls == 1:
            return ModelResponse(
                content="需要修改文件。",
                tool_calls=[
                    ToolCall(
                        call_id="paused-write",
                        name="write_file",
                        arguments={"path": "paused.txt", "content": "value"},
                    )
                ],
            )
        return ModelResponse(content="已取消修改。")


def test_api_exposes_project_and_background_chat(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "demo"
    dataset.mkdir(parents=True)
    (dataset / "dataset.json").write_text(
        '{"schema_version":1,"name":"Demo","meta":{},"resources":[],'
        '"characters":[],"conversations":[]}',
        encoding="utf-8",
    )
    model = ScriptedModelClient(["There is one Dataset."])
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=model,
        model="test-model",
        tokenizer=CharacterTokenizer(),
    )

    with TestClient(create_app(runtime)) as client:
        health = client.get("/api/health")
        project = client.get("/api/project")
        datasets = client.get("/api/datasets")
        response = client.post("/api/chat", json={"message": "List Datasets"})
        run = _wait_run(client, response.json())
        chats = client.get("/api/chats")
        messages = client.get(
            f"/api/chats/{response.json()['chat_session']}/messages"
        )

    assert health.json() == {"status": "ok"}
    assert project.json()["name"] == tmp_path.name
    assert "datasets" not in project.json()
    assert datasets.json()[0]["name"] == "Demo"
    assert response.status_code == 202
    assert run["content"] == "There is one Dataset."
    assert chats.json()[0]["title"] == "List Datasets"
    assert [item["role"] for item in messages.json()["items"]] == [
        "user",
        "assistant",
    ]
    extraction_tool = next(
        tool
        for tool in model.requests[0].tools
        if tool.name == "start_text_extraction"
    )
    assert extraction_tool.parameters["properties"]["debug"]["default"] is False


def test_api_previews_and_runs_manga_workflow(tmp_path: Path) -> None:
    pages = tmp_path / "resources" / "manga"
    pages.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(pages / "1.jpg")
    Image.new("RGB", (8, 8), "black").save(pages / "2.jpg")
    vision = ScriptedModelClient(
        [
            json.dumps(
                {
                    "utterances": [
                        {"page_index": 0, "speaker": 0, "content": "Hi"}
                    ],
                    "characters": [
                        {"index": 0, "names": ["Alice"], "description": "dark hair"}
                    ],
                    "chapter_starts": [{"page_index": 0, "title": "Chapter 1"}],
                    "plot": "Alice greets someone.",
                }
            )
        ],
        provider="qwen",
    )
    text = ScriptedModelClient(
        [
            json.dumps(
                {
                    "characters": [
                        {"name": "Alice", "aliases": [], "description": "dark hair"}
                    ]
                }
            ),
            json.dumps(
                {
                    "assignments": [
                        {
                            "batch_index": 0,
                            "local_character_index": 0,
                            "name": "Alice",
                        }
                    ]
                }
            ),
            json.dumps(
                {"utterances": [{"speaker": "Alice", "content": "Hi"}]}
            ),
            json.dumps({"profile": "A dark-haired character."}),
        ]
    )
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=text,
        model="text-model",
        tokenizer=CharacterTokenizer(),
        vision_model_client=vision,
    )

    with TestClient(create_app(runtime)) as client:
        models = client.get("/api/models")
        preview = client.get(
            "/api/resources/manga/preview",
            params={"path": "resources/manga"},
        )
        accepted = client.post(
            "/api/extractions",
            json={
                "workflow_type": "manga",
                "manifest": {
                    "name": "Manga API",
                    "meta": {},
                    "resources": [
                        {
                            "path": "resources/manga",
                            "resource_type": "manga",
                            "display_name": "Volume 1",
                            "narrative_order": 0,
                            "meta": {},
                        }
                    ],
                },
            },
        )
        completed = _wait_extraction(
            client,
            accepted.json()["task"],
            status="completed",
        )

    assert models.status_code == 200
    assert any(item["capabilities"] == ["text", "vision"] for item in models.json())
    assert preview.json()["image_count"] == 2
    assert preview.json()["first_paths"] == [
        "resources/manga/1.jpg",
        "resources/manga/2.jpg",
    ]
    dataset = tmp_path / completed["dataset_root"] / "dataset.json"
    assert json.loads(dataset.read_text(encoding="utf-8"))["name"] == "Manga API"
def test_api_returns_graceful_result_at_agent_round_limit(tmp_path: Path) -> None:
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
        accepted = client.post("/api/chat", json={"message": "Inspect this Project"})
        result = _wait_run(client, accepted.json())

    assert result["status"] == "completed"
    assert result["tool_calls"] == 12
    assert result["content"]


def test_api_pauses_for_tool_approval_and_resumes_once(tmp_path: Path) -> None:
    model = ScriptedModelClient(
        [
            ModelResponse(
                content="I need to create a file.",
                tool_calls=[
                    ToolCall(
                        call_id="call-write",
                        name="write_file",
                        arguments={"path": "approved.txt", "content": "approved"},
                    )
                ],
            ),
            "The file was created.",
        ]
    )
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=model,
        model="test-model",
        tokenizer=CharacterTokenizer(),
    )

    with TestClient(create_app(runtime), raise_server_exceptions=False) as client:
        accepted = client.post("/api/chat", json={"message": "Create approved.txt"})
        pending = _wait_run(client, accepted.json(), "awaiting_approval")
        assert (tmp_path / "approved.txt").exists() is False

        token = pending["approval"]["token"]
        resumed = client.post(f"/api/approvals/{token}", json={"allowed": True})
        result = _wait_run(client, resumed.json())
        repeated = client.post(f"/api/approvals/{token}", json={"allowed": True})

    assert pending["approval"]["tools"][0]["name"] == "write_file"
    assert result["status"] == "completed"
    assert result["content"] == "The file was created."
    assert (tmp_path / "approved.txt").read_text(encoding="utf-8") == "approved"
    assert repeated.status_code == 404


def test_running_extraction_blocks_agent_turn_in_the_same_chat(
    tmp_path: Path,
) -> None:
    source = tmp_path / "resources" / "book.txt"
    source.parent.mkdir()
    source.write_text("甲说：“你好。”", encoding="utf-8")  # noqa: RUF001
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=SlowWorkflowClient(_workflow_responses()),
        model="test-model",
        tokenizer=CharacterTokenizer(),
    )

    with TestClient(create_app(runtime)) as client:
        accepted = client.post("/api/extractions", json=_extraction_request())
        task = accepted.json()["task"]
        running = _wait_extraction(
            client, task, status="running", stage="plot_extraction"
        )
        blocked = client.post(
            "/api/chat",
            json={"message": "修改中间结果", "chat_session": running["chat_session"]},
        )
        finished = _wait_extraction(client, task, status="completed")

    assert blocked.status_code == 409
    assert "extraction is running" in blocked.json()["detail"]
    assert finished["status"] == "completed"


def test_staged_extraction_allows_agent_turn_but_waits_for_it_before_continuing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "resources" / "book.txt"
    source.parent.mkdir()
    source.write_text("甲说：“你好。”", encoding="utf-8")  # noqa: RUF001
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=PausedWorkflowClient(),
        model="test-model",
        tokenizer=CharacterTokenizer(),
    )

    with TestClient(create_app(runtime)) as client:
        accepted = client.post(
            "/api/extractions", json=_extraction_request(mode="staged")
        )
        task = accepted.json()["task"]
        paused = _wait_extraction(
            client, task, status="awaiting_confirmation", stage="chunk"
        )
        agent = client.post(
            "/api/chat",
            json={"message": "检查并修改输出", "chat_session": paused["chat_session"]},
        )
        pending = _wait_run(client, agent.json(), "awaiting_approval")
        blocked_confirmation = client.post(f"/api/extractions/{task}/confirm")
        resumed = client.post(
            f"/api/approvals/{pending['approval']['token']}",
            json={"allowed": False},
        )
        agent_finished = _wait_run(client, resumed.json())
        confirmed = client.post(f"/api/extractions/{task}/confirm")

    assert agent.status_code == 202
    assert blocked_confirmation.status_code == 409
    assert "Agent run" in blocked_confirmation.json()["detail"]
    assert agent_finished["status"] == "completed"
    assert confirmed.status_code == 200


def test_api_lists_updates_archives_and_deletes_chats(tmp_path: Path) -> None:
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=ScriptedModelClient([]),
        model="test-model",
        tokenizer=CharacterTokenizer(),
    )

    with TestClient(create_app(runtime)) as client:
        created = client.post("/api/chats", json={"title": "Draft chat"})
        chat_id = created.json()["id"]
        renamed = client.patch(
            f"/api/chats/{chat_id}", json={"title": "Renamed chat"}
        )
        archived = client.patch(
            f"/api/chats/{chat_id}", json={"archived": True}
        )
        active_list = client.get("/api/chats")
        archived_list = client.get("/api/chats?archived=true")
        restored = client.patch(
            f"/api/chats/{chat_id}", json={"archived": False}
        )
        deleted = client.delete(f"/api/chats/{chat_id}")
        missing = client.get(f"/api/chats/{chat_id}")

    assert renamed.json()["title"] == "Renamed chat"
    assert archived.json()["archived_at"] is not None
    assert active_list.json() == []
    assert archived_list.json()[0]["id"] == chat_id
    assert restored.json()["archived_at"] is None
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_api_rejects_unknown_chat_and_paginates_messages(tmp_path: Path) -> None:
    runtime = AppRuntime(
        project_root=tmp_path,
        model_client=ScriptedModelClient([]),
        model="test-model",
        tokenizer=CharacterTokenizer(),
    )
    chat_id = runtime.database.create_chat("Messages")
    for index in range(5):
        runtime.database.append_message(chat_id, "user", f"message-{index}")

    with TestClient(create_app(runtime)) as client:
        first = client.get(f"/api/chats/{chat_id}/messages?limit=2")
        second = client.get(
            f"/api/chats/{chat_id}/messages"
            f"?limit=2&before={first.json()['next_before']}"
        )
        missing = client.post(
            "/api/chat", json={"message": "orphan", "chat_session": 9999}
        )

    assert [item["content"] for item in first.json()["items"]] == [
        "message-3",
        "message-4",
    ]
    assert [item["content"] for item in second.json()["items"]] == [
        "message-1",
        "message-2",
    ]
    assert missing.status_code == 404
