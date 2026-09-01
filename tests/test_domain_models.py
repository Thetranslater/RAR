import pytest
from pydantic import ValidationError

from rar_agent.domain.models import (
    CharacterProfile,
    Conversation,
    DatasetBundle,
    DatasetResource,
    PlotChunkRef,
    TextChunk,
    TextChunkSpanRef,
    Utterance,
)


def test_dataset_bundle_validates_the_persisted_no_id_contract() -> None:
    chunk = TextChunk(
        file="resources/book.txt",
        index=0,
        text="比企谷：早上好。",  # noqa: RUF001 - persisted CJK dialogue fixture
        token_count=8,
        meta={"volume": "第一卷", "chapter": "第一章"},
    )
    plot_ref = PlotChunkRef(path="work/plots.json", plot_index=0, chunk_index=0)
    text_ref = TextChunkSpanRef(
        path="work/text_chunks.jsonl", index=0, start=4, end=8
    )
    conversation = Conversation(
        plot_ref={"path": "work/plots.json", "index": 0},
        utterances=[
            Utterance(
                index=0,
                speaker="比企谷八幡",
                content="早上好。",
                source_refs=[plot_ref, text_ref],
            )
        ],
    )
    bundle = DatasetBundle(
        schema_version=1,
        name="测试作品",
        meta={"language": "zh"},
        resources=[
            DatasetResource(path=chunk.file, media_type="text", meta=chunk.meta)
        ],
        characters=[
            CharacterProfile(
                name="比企谷八幡",
                aliases=["比企谷"],
                profile="总武高中的学生。",
                plot_refs=[{"path": "work/plots.json", "index": 0}],
            )
        ],
        conversations=[conversation],
    )

    payload = bundle.model_dump(mode="json")
    assert payload["schema_version"] == 1
    assert payload["conversations"][0]["utterances"][0]["source_refs"][1] == {
        "path": "work/text_chunks.jsonl",
        "index": 0,
        "start": 4,
        "end": 8,
    }
    assert "id" not in repr(payload).lower()


def test_environment_requires_narration_markers() -> None:
    with pytest.raises(ValidationError, match="Environment content"):
        Utterance(
            index=0,
            speaker="Environment",
            content="教室安静下来。",
            source_refs=[
                PlotChunkRef(path="work/plots.json", plot_index=0, chunk_index=0)
            ],
        )


def test_utterance_requires_a_plot_chunk_reference() -> None:
    with pytest.raises(ValidationError, match="PlotChunk reference"):
        Utterance(index=0, speaker="雪之下雪乃", content="早上好。", source_refs=[])
