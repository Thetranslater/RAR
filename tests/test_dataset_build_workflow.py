import json
from pathlib import Path

import pytest

from rar_agent.domain.models import InputManifest, InputResource
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.text.tokenizer import CharacterTokenizer
from rar_agent.workflow.dataset_build import (
    DEBUG_STAGE_UNIT_LIMIT,
    PLOT_OPENING_PLUGIN,
    DatasetBuildWorkflow,
    IncompleteStageError,
    WorkflowConfig,
)


def _manifest() -> InputManifest:
    return InputManifest(
        name="测试作品",
        meta={"language": "zh"},
        resources=[
            InputResource(
                path="resources/book.txt",
                resource_type="text",
                display_name="第一章",
                narrative_order=0,
                meta={"chapter": "第一章"},
            )
        ],
    )


def _plot_response() -> str:
    return json.dumps(
        {
            "characters": [
                {
                    "name": ["雪之下雪乃", "雪乃"],
                    "description": "冷静地看书。",
                    "plot_indexes": [0],
                },
                {
                    "name": ["比企谷八幡", "八幡"],
                    "description": "询问雪乃是否回去。",
                    "plot_indexes": [0],
                },
            ],
            "plots": [["雪乃看书。", "雪乃答：“我只是在等这里安静下来。”"]],  # noqa: RUF001
            "state": "finished",
        },
        ensure_ascii=False,
    )


def _profile_response(name: str, content: str) -> str:
    return json.dumps(
        {
            "profile": {
                "name": name,
                "content": content,
            }
        },
        ensure_ascii=False,
    )


def _dialogue_response() -> str:
    return json.dumps(
        {
            "utterances": [
                {"speaker": "Environment", "content": "*(雪乃看书。)*"},
                {"speaker": "八幡", "content": "你还没回去吗？"},  # noqa: RUF001
                {"speaker": "雪乃", "content": "我只是在等这里安静下来。"},
            ]
        },
        ensure_ascii=False,
    )


def _workflow(client: ScriptedModelClient) -> DatasetBuildWorkflow:
    return DatasetBuildWorkflow(
        model_client=client,
        tokenizer=CharacterTokenizer(),
        config=WorkflowConfig(
            model="test-model",
            text_chunk_tokens=10_000,
            plot_chunk_tokens=10_000,
            max_concurrency=2,
        ),
    )


def _request_payload(content: str | None) -> dict[str, object]:
    assert content is not None
    return json.loads(content[content.index("{") :])


async def test_dataset_build_workflow_produces_inspectable_dataset(tmp_path: Path) -> None:
    source = tmp_path / "resources" / "book.txt"
    source.parent.mkdir()
    source.write_text(
        "雪乃看书。八幡问：“你还没回去吗？”雪乃答：“我只是在等这里安静下来。”",  # noqa: RUF001
        encoding="utf-8",
    )
    client = ScriptedModelClient(
        [
            _plot_response(),
            _profile_response("雪乃", "冷静而理性。"),
            _profile_response("比企谷八幡", "观察敏锐。"),
            _dialogue_response(),
        ]
    )

    result = await _workflow(client).run(tmp_path, _manifest())

    assert result.bundle.name == "测试作品"
    assert len(result.bundle.conversations) == 1
    assert [character.name for character in result.bundle.characters] == [
        "雪乃",
        "比企谷八幡",
    ]
    assert result.bundle.characters[0].aliases == ["雪之下雪乃"]
    assert result.plots.plots[0].character_refs[0].path == (
        "work/character_profiles.jsonl"
    )
    assert result.export_report.sample_count == 2
    assert result.paths.dataset.exists()
    assert result.paths.plots.exists()
    assert (result.paths.exports / "sharegpt" / "all.jsonl").exists()
    assert len(client.requests) == 4
    plot_request = client.requests[0]
    profile_request = client.requests[1]
    dialogue_request = client.requests[3]
    plot_payload = _request_payload(plot_request.messages[-1].content)
    profile_payload = _request_payload(profile_request.messages[-1].content)
    dialogue_payload = _request_payload(dialogue_request.messages[-1].content)
    assert plot_request.messages[-1].content.startswith("===输入===\n")
    assert dialogue_request.messages[-1].content.startswith("===输入===\n")
    assert "----------" not in (plot_request.messages[0].content or "")
    assert "----------" not in (dialogue_request.messages[0].content or "")
    assert PLOT_OPENING_PLUGIN in (dialogue_request.messages[0].content or "")
    assert set(plot_payload) == {"input", "previous", "next"}
    assert set(profile_payload) == {"character"}
    assert set(profile_payload["character"]) == {"names", "description"}
    assert profile_request.messages[-1].content.startswith("===输入===\n")
    assert set(dialogue_payload) == {"input", "characters"}
    assert set(dialogue_payload["characters"][0]) == {"names", "description"}
    assert dialogue_payload["characters"][0] == {
        "names": ["雪乃", "雪之下雪乃"],
        "description": "冷静地看书。",
    }


async def test_workflow_reports_each_stage_start_and_completion(tmp_path: Path) -> None:
    source = tmp_path / "resources" / "book.txt"
    source.parent.mkdir()
    source.write_text(
        "雪乃看书。八幡问：“你还没回去吗？”雪乃答：“我只是在等这里安静下来。”",  # noqa: RUF001
        encoding="utf-8",
    )
    events: list[tuple[str, str]] = []

    async def record_stage(stage: str, phase: str, _artifact: Path) -> None:
        events.append((stage, phase))

    workflow = DatasetBuildWorkflow(
        model_client=ScriptedModelClient(
                [
                    _plot_response(),
                    _profile_response("雪乃", "冷静而理性。"),
                    _profile_response("比企谷八幡", "观察敏锐。"),
                    _dialogue_response(),
                ]
        ),
        tokenizer=CharacterTokenizer(),
        config=WorkflowConfig(
            model="test-model",
            text_chunk_tokens=10_000,
            plot_chunk_tokens=10_000,
            max_concurrency=2,
        ),
        stage_callback=record_stage,
    )

    await workflow.run(tmp_path, _manifest())

    assert events == [
        (stage, phase)
        for stage in (
            "chunk",
            "plot_extraction",
            "character_profile",
            "plot_reconstruction",
            "dialogue_extraction",
            "dataset",
        )
        for phase in ("started", "completed")
    ]


async def test_workflow_reloads_stage_artifact_after_confirmation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "resources" / "book.txt"
    source.parent.mkdir()
    source.write_text("修改前的内容。", encoding="utf-8")
    edited = "甲说：“你好。”"  # noqa: RUF001
    client = ScriptedModelClient(
        [
            json.dumps(
                {
                    "characters": [],
                    "plots": [[edited, edited]],
                    "state": "finished",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {"utterances": [{"speaker": "甲", "content": "你好。"}]},
                ensure_ascii=False,
            ),
        ]
    )

    async def edit_chunk(stage: str, phase: str, artifact: Path) -> None:
        if (stage, phase) != ("chunk", "completed"):
            return
        rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
        rows[0]["text"] = edited
        rows[0]["token_count"] = len(edited)
        artifact.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    workflow = DatasetBuildWorkflow(
        model_client=client,
        tokenizer=CharacterTokenizer(),
        config=WorkflowConfig(
            model="test-model",
            text_chunk_tokens=10_000,
            plot_chunk_tokens=10_000,
        ),
        stage_callback=edit_chunk,
    )

    await workflow.run(tmp_path, _manifest())

    plot_payload = _request_payload(client.requests[0].messages[-1].content)
    assert plot_payload["input"] == edited


async def test_debug_mode_limits_expensive_stages_but_chunks_all_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "resources" / "book.txt"
    source.parent.mkdir()
    source.write_text(
        "".join(
            f"\u7b2c{index + 1}\u7ae0\nsection {index}.\n"
            for index in range(DEBUG_STAGE_UNIT_LIMIT + 3)
        ),
        encoding="utf-8",
    )
    plot_response = '{"characters":[],"plots":[],"state":"finished"}'
    dialogue_response = '{"utterances":[]}'
    client = ScriptedModelClient(
        [
            *[plot_response] * DEBUG_STAGE_UNIT_LIMIT,
            *[dialogue_response] * DEBUG_STAGE_UNIT_LIMIT,
        ]
    )
    workflow = DatasetBuildWorkflow(
        model_client=client,
        tokenizer=CharacterTokenizer(),
        config=WorkflowConfig(
            model="test-model",
            debug=True,
            text_chunk_tokens=10_000,
            plot_chunk_tokens=10_000,
        ),
    )

    result = await workflow.run(tmp_path, _manifest())

    saved_chunks = result.paths.text_chunks.read_text(encoding="utf-8").splitlines()
    payloads = [
        _request_payload(request.messages[-1].content)
        for request in client.requests
    ]
    plot_payloads = [
        payload
        for payload in payloads
        if set(payload) == {"input", "previous", "next"}
    ]
    dialogue_payloads = [
        payload
        for payload in payloads
        if set(payload) == {"input", "characters"}
    ]

    assert len(saved_chunks) == DEBUG_STAGE_UNIT_LIMIT + 3
    assert len(plot_payloads) == DEBUG_STAGE_UNIT_LIMIT
    assert len(result.plots.plots) == DEBUG_STAGE_UNIT_LIMIT
    assert len(dialogue_payloads) == DEBUG_STAGE_UNIT_LIMIT
    assert len(result.bundle.conversations) == DEBUG_STAGE_UNIT_LIMIT
    assert len(client.requests) == DEBUG_STAGE_UNIT_LIMIT * 2


async def test_resume_reruns_only_empty_stage_units(tmp_path: Path) -> None:
    source = tmp_path / "resources" / "book.txt"
    source.parent.mkdir()
    source.write_text(
        "雪乃看书。八幡问：“你还没回去吗？”雪乃答：“我只是在等这里安静下来。”",  # noqa: RUF001
        encoding="utf-8",
    )
    first_client = ScriptedModelClient(
        [
            _plot_response(),
            _profile_response("雪乃", "冷静而理性。"),
            _profile_response("比企谷八幡", "观察敏锐。"),
            "invalid",
            "invalid",
            "invalid",
        ]
    )

    with pytest.raises(IncompleteStageError, match="dialogue_extraction") as error:
        await _workflow(first_client).run(tmp_path, _manifest())

    dataset_root = error.value.dataset_root
    second_client = ScriptedModelClient(
        [_dialogue_response()]
    )
    result = await _workflow(second_client).run(
        tmp_path,
        _manifest(),
        dataset_root=dataset_root,
    )

    assert result.bundle.characters[0].profile == "冷静而理性。"
    assert len(second_client.requests) == 1
