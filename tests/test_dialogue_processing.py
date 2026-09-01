import pytest
from pydantic import ValidationError

from rar_agent.domain.models import Plot, PlotChunk, TextChunkSpanRef
from rar_agent.text.dialogue import ConversationAssembler, DialogueAligner, RawUtterance


def _plot(text: str) -> Plot:
    return Plot(
        index=0,
        meta={"chapter": "第一章"},
        character_refs=[],
        chunks=[
            PlotChunk(
                index=0,
                text=text,
                token_count=len(text),
                source_refs=[
                    TextChunkSpanRef(
                        path="work/text_chunks.jsonl",
                        index=0,
                        start=0,
                        end=len(text),
                    )
                ],
            )
        ],
    )


def test_raw_environment_requires_stable_markers() -> None:
    with pytest.raises(ValidationError, match="Environment content"):
        RawUtterance(speaker="Environment", content="雪乃放下了书。")


def test_alignment_uses_a_forward_cursor_for_repeated_dialogue() -> None:
    plot = _plot("甲说：嗯。乙说：嗯。")  # noqa: RUF001 - CJK dialogue fixture

    batch = DialogueAligner().align(
        plot.index,
        plot.chunks[0],
        [RawUtterance(speaker="甲", content="嗯。"), RawUtterance(speaker="乙", content="嗯。")],
        aliases={},
    )

    first_span = batch.utterances[0].source_refs[1]
    second_span = batch.utterances[1].source_refs[1]
    assert isinstance(first_span, TextChunkSpanRef)
    assert isinstance(second_span, TextChunkSpanRef)
    assert first_span.start < second_span.start


def test_environment_keeps_markers_and_does_not_receive_a_text_span() -> None:
    plot = _plot("雪乃放下了书。")

    batch = DialogueAligner().align(
        plot.index,
        plot.chunks[0],
        [RawUtterance(speaker="Environment", content="*(雪乃放下了书。)*")],
        aliases={},
    )

    assert batch.utterances[0].content == "*(雪乃放下了书。)*"
    assert len(batch.utterances[0].source_refs) == 1


def test_out_of_order_alignment_preserves_model_order_and_unmatched_dialogue() -> None:
    plot = _plot("先发生。后发生。")

    batch = DialogueAligner().align(
        plot.index,
        plot.chunks[0],
        [
            RawUtterance(speaker="角色", content="后发生。"),
            RawUtterance(speaker="角色", content="先发生。"),
        ],
        aliases={},
    )

    assert [item.content for item in batch.utterances] == ["后发生。", "先发生。"]
    assert len(batch.utterances[0].source_refs) == 2
    assert len(batch.utterances[1].source_refs) == 1


def test_assembler_orders_batches_by_plot_chunk_and_reindexes_utterances() -> None:
    plot = _plot("第一句。")
    plot.chunks.append(
        PlotChunk(
            index=1,
            text="第二句。",
            token_count=4,
            source_refs=[
                TextChunkSpanRef(
                    path="work/text_chunks.jsonl", index=1, start=0, end=4
                )
            ],
        )
    )
    aligner = DialogueAligner()
    later = aligner.align(
        0, plot.chunks[1], [RawUtterance(speaker="角色", content="第二句。")], aliases={}
    )
    earlier = aligner.align(
        0, plot.chunks[0], [RawUtterance(speaker="角色", content="第一句。")], aliases={}
    )

    conversation = ConversationAssembler().assemble(plot, [later, earlier])

    assert [item.index for item in conversation.utterances] == [0, 1]
    assert [item.content for item in conversation.utterances] == ["第一句。", "第二句。"]
