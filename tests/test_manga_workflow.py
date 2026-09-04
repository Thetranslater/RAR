import json
from pathlib import Path

import pytest
from PIL import Image

from rar_agent.domain.models import InputManifest, InputResource, PlotChunkRef
from rar_agent.models.base import LocalImageContent, TextContent
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.text.tokenizer import CharacterTokenizer
from rar_agent.workflow.dataset_build import IncompleteStageError
from rar_agent.workflow.manga.dataset_build import (
    MangaDatasetBuildWorkflow,
    MangaWorkflowConfig,
)


def _manifest() -> InputManifest:
    return InputManifest(
        name="Manga Test",
        meta={"language": "en"},
        resources=[
            InputResource(
                path="resources/manga",
                resource_type="manga",
                display_name="Volume 1",
                narrative_order=0,
            )
        ],
    )


def _write_pages(project: Path) -> None:
    root = project / "resources" / "manga"
    root.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(root / "1.jpg")
    Image.new("RGB", (8, 8), "black").save(root / "2.jpg")


async def test_manga_workflow_builds_dataset_and_sharegpt_from_local_images(
    tmp_path: Path,
) -> None:
    _write_pages(tmp_path)
    visual_client = ScriptedModelClient(
        [
            json.dumps(
                {
                    "utterances": [
                        {"page_index": 0, "speaker": 0, "content": "Hi"},
                        {"page_index": 1, "speaker": None, "content": "Who?"},
                    ],
                    "characters": [
                        {
                            "index": 0,
                            "names": ["Alice"],
                            "description": "dark hair",
                        }
                    ],
                    "chapter_starts": [{"page_index": 0, "title": "Chapter 1"}],
                    "plot": "Alice meets someone.",
                }
            )
        ],
        provider="qwen",
    )
    text_client = ScriptedModelClient(
        [
            json.dumps(
                {
                    "characters": [
                        {
                            "name": "Alice",
                            "aliases": ["Al"],
                            "description": "dark hair",
                        }
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
                {
                    "utterances": [
                        {"speaker": "Unknown", "content": "Who?"},
                        {"speaker": "Alice", "content": "Hello."},
                    ]
                }
            ),
            json.dumps({"profile": "A dark-haired student."}),
        ],
        provider="scripted-text",
    )
    events: list[tuple[str, str]] = []
    progress: list[tuple[str, int, int]] = []

    async def record_stage(stage: str, phase: str, _artifact: Path) -> None:
        events.append((stage, phase))

    async def record_progress(stage: str, current: int, total: int) -> None:
        progress.append((stage, current, total))

    workflow = MangaDatasetBuildWorkflow(
        vision_model_client=visual_client,
        text_model_client=text_client,
        tokenizer=CharacterTokenizer(),
        config=MangaWorkflowConfig(
            vision_model="qwen3.7-flash",
            text_model="test-text",
            max_concurrency=2,
        ),
        stage_callback=record_stage,
        progress_callback=record_progress,
    )

    result = await workflow.run(tmp_path, _manifest())

    assert result.bundle.name == "Manga Test"
    assert result.bundle.resources[0].media_type == "manga"
    assert result.bundle.resources[0].meta["page_count"] == 2
    assert [character.name for character in result.bundle.characters] == ["Alice"]
    assert result.bundle.characters[0].aliases == ["Al"]
    assert result.bundle.characters[0].profile == "A dark-haired student."
    assert [line.speaker for line in result.bundle.conversations[0].utterances] == [
        "Unknown",
        "Alice",
    ]
    assert isinstance(
        result.bundle.conversations[0].utterances[0].source_refs[0], PlotChunkRef
    )
    assert result.export_report.sample_count == 1
    assert result.paths.dataset.exists()
    assert result.paths.visual_extractions.exists()
    assert result.paths.character_catalog.exists()
    assert result.paths.report.exists()

    visual_request = visual_client.requests[0]
    assert visual_request.model == "qwen3.7-flash"
    assert visual_request.temperature == 1
    assert visual_request.provider_options == {
        "top_p": 0.9,
        "enable_thinking": True,
        "reasoning_effort": "medium",
        "max_completion_tokens": 16_384,
    }
    visual_content = visual_request.messages[-1].content
    assert isinstance(visual_content, list)
    assert [type(block) for block in visual_content] == [
        TextContent,
        TextContent,
        LocalImageContent,
        TextContent,
        LocalImageContent,
    ]

    assert events == [
        (stage, phase)
        for stage in (
            "image_scan",
            "visual_extraction",
            "ocr_alignment",
            "character_catalog",
            "character_assignment",
            "chapter_reconstruction",
            "dialogue_revision",
            "character_profile",
            "dataset",
            "export",
        )
        for phase in ("started", "completed")
    ]
    assert ("visual_extraction", 1, 1) in progress
    assert ("character_profile", 1, 1) in progress

    resumed_visual = ScriptedModelClient([], provider="qwen")
    resumed_text = ScriptedModelClient([], provider="scripted-text")
    resumed = MangaDatasetBuildWorkflow(
        vision_model_client=resumed_visual,
        text_model_client=resumed_text,
        tokenizer=CharacterTokenizer(),
        config=MangaWorkflowConfig(
            vision_model="qwen3.7-flash",
            text_model="test-text",
            max_concurrency=2,
        ),
    )
    resumed_result = await resumed.run(
        tmp_path,
        _manifest(),
        dataset_root=result.dataset_root,
    )
    assert resumed_result.bundle == result.bundle
    assert resumed_visual.requests == []
    assert resumed_text.requests == []


async def test_manga_workflow_writes_summary_for_failed_visual_batches(
    tmp_path: Path,
) -> None:
    _write_pages(tmp_path)
    workflow = MangaDatasetBuildWorkflow(
        vision_model_client=ScriptedModelClient(
            ["not json", "still not json"],
            provider="qwen",
        ),
        text_model_client=ScriptedModelClient([]),
        tokenizer=CharacterTokenizer(),
        config=MangaWorkflowConfig(visual_max_attempts=2),
    )

    with pytest.raises(IncompleteStageError) as captured:
        await workflow.run(tmp_path, _manifest())

    report_path = captured.value.dataset_root / "reports" / "manga_extraction_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert report["counts"]["failed_visual_batches"] == 1
