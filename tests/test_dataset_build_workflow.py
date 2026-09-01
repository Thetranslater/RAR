import json
from pathlib import Path

import pytest

from rar_agent.domain.models import InputManifest, InputResource
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.text.tokenizer import CharacterTokenizer
from rar_agent.workflow.dataset_build import (
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
                    "names": ["雪之下雪乃", "雪乃"],
                    "description": "冷静地看书。",
                    "plot_indexes": [0],
                },
                {
                    "names": ["比企谷八幡", "八幡"],
                    "description": "询问雪乃是否回去。",
                    "plot_indexes": [0],
                },
            ],
            "plots": [["雪乃看书。", "雪乃答：“我只是在等这里安静下来。”"]],  # noqa: RUF001
            "state": "finished",
        },
        ensure_ascii=False,
    )


def _filter_response() -> str:
    return json.dumps(
        {
            "characters": [
                {
                    "name": "雪之下雪乃",
                    "aliases": ["雪乃"],
                    "candidate_indexes": [0],
                },
                {
                    "name": "比企谷八幡",
                    "aliases": ["八幡"],
                    "candidate_indexes": [1],
                },
            ]
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
            _filter_response(),
            _dialogue_response(),
            '{"profile":"冷静而理性。"}',
            '{"profile":"观察敏锐。"}',
        ]
    )

    result = await _workflow(client).run(tmp_path, _manifest())

    assert result.bundle.name == "测试作品"
    assert len(result.bundle.conversations) == 1
    assert [character.name for character in result.bundle.characters] == [
        "雪之下雪乃",
        "比企谷八幡",
    ]
    assert result.export_report.sample_count == 2
    assert result.paths.dataset.exists()
    assert result.paths.plots.exists()
    assert (result.paths.exports / "sharegpt" / "all.jsonl").exists()
    assert len(client.requests) == 5


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
            _filter_response(),
            "invalid",
            "invalid",
            "invalid",
        ]
    )

    with pytest.raises(IncompleteStageError, match="dialogue_extraction") as error:
        await _workflow(first_client).run(tmp_path, _manifest())

    dataset_root = error.value.dataset_root
    second_client = ScriptedModelClient(
        [
            _dialogue_response(),
            '{"profile":"冷静而理性。"}',
            '{"profile":"观察敏锐。"}',
        ]
    )
    result = await _workflow(second_client).run(
        tmp_path,
        _manifest(),
        dataset_root=dataset_root,
    )

    assert result.bundle.characters[0].profile == "冷静而理性。"
    assert len(second_client.requests) == 3


async def test_long_character_descriptions_are_summarized_hierarchically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "resources" / "book.txt"
    source.parent.mkdir()
    source.write_text("雪乃安静地看书。", encoding="utf-8")
    description = "冷静理性并且善于观察细节。"
    partial_count = (len(description) + 4) // 5
    client = ScriptedModelClient(
        [
            json.dumps(
                {
                    "characters": [
                        {
                            "names": ["雪之下雪乃", "雪乃"],
                            "description": description,
                            "plot_indexes": [0],
                        }
                    ],
                    "plots": [["雪乃安静地看书。", "雪乃安静地看书。"]],
                    "state": "finished",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "characters": [
                        {
                            "name": "雪之下雪乃",
                            "aliases": ["雪乃"],
                            "candidate_indexes": [0],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            '{"utterances":[]}',
            *[
                json.dumps({"profile": f"部分{i}"}, ensure_ascii=False)
                for i in range(partial_count)
            ],
            '{"profile":"最终角色档案"}',
        ]
    )
    workflow = DatasetBuildWorkflow(
        model_client=client,
        tokenizer=CharacterTokenizer(),
        config=WorkflowConfig(
            model="test-model",
            text_chunk_tokens=10_000,
            plot_chunk_tokens=10_000,
            profile_description_chars=5,
        ),
    )

    result = await workflow.run(tmp_path, _manifest())

    assert result.bundle.characters[0].profile == "最终角色档案"
    profile_payloads = [
        json.loads(request.messages[-1].content or "{}")
        for request in client.requests[3:]
    ]
    assert [payload["mode"] for payload in profile_payloads[:-1]] == [
        "partial"
    ] * partial_count
    assert profile_payloads[-1]["mode"] == "final"
