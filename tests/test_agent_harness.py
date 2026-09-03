import json
from pathlib import Path

from rar_agent.agent.harness import AgentHarness
from rar_agent.agent.tools import ToolDispatcher, build_workspace_tools
from rar_agent.models.base import ModelResponse, ToolCall
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.storage.database import ProjectDatabase


async def test_agent_harness_finishes_without_tools(tmp_path: Path) -> None:
    database = ProjectDatabase(tmp_path)
    client = ScriptedModelClient(["分析完成。"])
    harness = AgentHarness(
        model_client=client,
        model="test-model",
        dispatcher=ToolDispatcher(tmp_path, build_workspace_tools(tmp_path)),
        database=database,
    )

    result = await harness.run("检查当前项目。")

    assert result.content == "分析完成。"
    assert result.tool_calls == 0
    messages = database.list_messages(result.chat_session)
    assert [message.role for message in messages] == ["user", "assistant"]


async def test_agent_harness_executes_multi_round_workspace_change(tmp_path: Path) -> None:
    target = tmp_path / "datasets" / "demo" / "dataset.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"name":"old"}\n', encoding="utf-8")
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        call_id="call-1",
                        name="replace_text",
                        arguments={
                            "path": "datasets/demo/dataset.json",
                            "old": '"old"',
                            "new": '"new"',
                        },
                    )
                ]
            ),
            "已修改并验证数据集名称。",
        ]
    )
    database = ProjectDatabase(tmp_path)
    harness = AgentHarness(
        model_client=client,
        model="test-model",
        dispatcher=ToolDispatcher(tmp_path, build_workspace_tools(tmp_path)),
        database=database,
    )

    pending = await harness.run("把数据集名称从 old 改成 new。")

    assert pending.status == "awaiting_approval"
    assert pending.pending is not None
    assert pending.approvals[0].tool_name == "replace_text"
    assert '"old"' in target.read_text(encoding="utf-8")
    assert [message.role for message in database.list_messages(pending.chat_session)] == [
        "user"
    ]

    result = await harness.resume(pending.pending, approved=True)

    assert result.status == "completed"
    assert result.content == "已修改并验证数据集名称。"
    assert result.tool_calls == 1
    assert '"new"' in target.read_text(encoding="utf-8")
    assert client.requests[1].messages[-1].role == "tool"
    assert database.list_tool_calls(result.chat_session)[0].status == "succeeded"


async def test_agent_harness_reports_user_denial_to_model(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        call_id="call-write",
                        name="write_file",
                        arguments={"path": "note.txt", "content": "content"},
                    )
                ]
            ),
            "文件未创建。",
        ]
    )
    database = ProjectDatabase(tmp_path)
    harness = AgentHarness(
        model_client=client,
        model="test-model",
        dispatcher=ToolDispatcher(tmp_path, build_workspace_tools(tmp_path)),
        database=database,
    )

    pending = await harness.run("创建 note.txt。")
    assert pending.pending is not None

    result = await harness.resume(pending.pending, approved=False)

    assert result.content == "文件未创建。"
    assert target.exists() is False
    assert database.list_tool_calls(result.chat_session)[0].status == "denied"
    assert "denied by user" in client.requests[1].messages[-1].content


async def test_agent_harness_can_continue_existing_chat(tmp_path: Path) -> None:
    database = ProjectDatabase(tmp_path)
    dispatcher = ToolDispatcher(tmp_path, build_workspace_tools(tmp_path))
    first = AgentHarness(
        model_client=ScriptedModelClient(["第一步完成。"]),
        model="test-model",
        dispatcher=dispatcher,
        database=database,
    )
    initial = await first.run("开始检查。")
    second_client = ScriptedModelClient(["已根据已有记录继续。"])
    second = AgentHarness(
        model_client=second_client,
        model="test-model",
        dispatcher=dispatcher,
        database=database,
    )

    continued = await second.run("请继续。", chat_session=initial.chat_session)

    assert continued.chat_session == initial.chat_session
    assert [message.content for message in second_client.requests[0].messages[-3:]] == [
        "开始检查。",
        "第一步完成。",
        "请继续。",
    ]


async def test_workspace_inspection_reports_dataset_intermediate_artifacts(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets" / "demo"
    work = dataset_root / "work"
    work.mkdir(parents=True)
    (dataset_root / "input_manifest.json").write_text(
        json.dumps({"name": "Demo", "meta": {}, "resources": []}),
        encoding="utf-8",
    )
    (work / "text_chunks.jsonl").write_text(
        '{"file":"book.txt","index":0,"text":"demo","tokens":1}\n',
        encoding="utf-8",
    )

    dispatcher = ToolDispatcher(tmp_path, build_workspace_tools(tmp_path))
    observation = await dispatcher.dispatch(
        ToolCall(call_id="inspect-1", name="inspect_project", arguments={})
    )

    assert observation.success is True
    assert observation.content["summary"] == {
        "datasets": 1,
        "complete": 0,
        "in_progress": 1,
        "intermediate_artifacts": 2,
    }
    dataset = observation.content["datasets"][0]
    assert dataset["directory"] == "demo"
    assert dataset["name"] == "Demo"
    assert dataset["status"] == "in_progress"
    assert [artifact["kind"] for artifact in dataset["intermediate_artifacts"]] == [
        "input_manifest",
        "text_chunks",
    ]
