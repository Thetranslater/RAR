import json
from pathlib import Path

from rar_agent.domain.models import (
    CharacterProfile,
    Conversation,
    DatasetBundle,
    DatasetResource,
    Plot,
    PlotChunk,
    PlotChunkRef,
    PlotRef,
    PlotsDocument,
    TextChunkSpanRef,
    Utterance,
)
from rar_agent.export.sharegpt import ShareGPTExporter


def _utterance(index: int, speaker: str, content: str) -> Utterance:
    return Utterance(
        index=index,
        speaker=speaker,
        content=content,
        source_refs=[
            PlotChunkRef(path="work/plots.json", plot_index=0, chunk_index=0)
        ],
    )


def test_exporter_builds_one_loss_marked_sample_per_formal_speaker(tmp_path: Path) -> None:
    plots = PlotsDocument(
        plots=[
            Plot(
                index=0,
                meta={"chapter": "第一章"},
                character_refs=[],
                chunks=[
                    PlotChunk(
                        index=0,
                        text="原始剧情。",
                        token_count=5,
                        source_refs=[
                            TextChunkSpanRef(
                                path="work/text_chunks.jsonl",
                                index=0,
                                start=0,
                                end=5,
                            )
                        ],
                    )
                ],
            )
        ]
    )
    bundle = DatasetBundle(
        name="测试作品",
        resources=[
            DatasetResource(path="resources/book.txt", media_type="text", meta={})
        ],
        characters=[
            CharacterProfile(
                name="雪之下雪乃",
                aliases=["雪乃"],
                profile="冷静。",
                plot_refs=[PlotRef(path="work/plots.json", index=0)],
            ),
            CharacterProfile(
                name="比企谷八幡",
                aliases=["比企谷"],
                profile="观察敏锐。",
                plot_refs=[PlotRef(path="work/plots.json", index=0)],
            ),
        ],
        conversations=[
            Conversation(
                plot_ref=PlotRef(path="work/plots.json", index=0),
                utterances=[
                    _utterance(0, "Environment", "*(雪乃放下手中的书。)*"),
                    _utterance(1, "比企谷八幡", "你还没回去吗？"),  # noqa: RUF001
                    _utterance(2, "雪之下雪乃", "我只是在等这里安静下来。"),
                ],
            )
        ],
    )

    report = ShareGPTExporter().export(
        bundle,
        plots,
        tmp_path,
        system_template="角色={character};档案={profile};剧情={plot}",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "all.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert report.sample_count == 2
    snow = next(row for row in rows if "角色=雪之下雪乃" in row["messages"][0]["content"])
    assert snow["messages"] == [
        {"role": "system", "content": "角色=雪之下雪乃;档案=冷静。;剧情=原始剧情。"},
        {
            "role": "user",
            "content": "Environment：*(雪乃放下手中的书。)*\n比企谷八幡：你还没回去吗？",  # noqa: RUF001
        },
        {
            "role": "assistant",
            "content": "雪之下雪乃：我只是在等这里安静下来。",  # noqa: RUF001
            "loss": True,
        },
    ]
    assert len(list((tmp_path / "characters").glob("*.jsonl"))) == 2
