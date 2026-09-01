from rar_agent.domain.models import PlotExtractionResult, TextChunk
from rar_agent.text.plot_rebuilder import PlotRebuilder
from rar_agent.text.tokenizer import CharacterTokenizer


def _chunk(index: int, text: str, chapter: str = "第一章") -> TextChunk:
    return TextChunk(
        file="resources/book.txt",
        index=index,
        text=text,
        token_count=len(text),
        meta={"chapter": chapter},
    )


def test_rebuilder_closes_a_cross_chunk_plot_at_the_section_end() -> None:
    chunks = [_chunk(0, "开端。持续。"), _chunk(1, "继续。结束。")]
    results = [
        PlotExtractionResult(characters=[], plots=[("开端。", None)], state="truncated"),
        PlotExtractionResult(characters=[], plots=[], state="finished"),
    ]

    document = PlotRebuilder(CharacterTokenizer(), target_tokens=100).rebuild(
        chunks, results
    )

    assert len(document.plots) == 1
    assert "".join(chunk.text for chunk in document.plots[0].chunks) == "开端。持续。继续。结束。"
    assert [ref.index for ref in document.plots[0].chunks[0].source_refs] == [0, 1]


def test_empty_plots_without_an_open_plot_use_the_whole_section() -> None:
    chunks = [
        _chunk(0, "第一章正文。", "第一章"),
        _chunk(1, "第二章正文。", "第二章"),
    ]
    results = [
        PlotExtractionResult(characters=[], plots=[], state="finished"),
        PlotExtractionResult(characters=[], plots=[], state="finished"),
    ]

    document = PlotRebuilder(CharacterTokenizer(), target_tokens=100).rebuild(
        chunks, results
    )

    assert [plot.chunks[0].text for plot in document.plots] == [
        "第一章正文。",
        "第二章正文。",
    ]


def test_rebuilder_splits_large_plots_without_splitting_sentences() -> None:
    chunks = [_chunk(0, "第一句。第二句。第三句。")]
    results = [
        PlotExtractionResult(
            characters=[], plots=[("第一句。", "第三句。")], state="finished"
        )
    ]

    document = PlotRebuilder(CharacterTokenizer(), target_tokens=5).rebuild(chunks, results)

    assert [chunk.text for chunk in document.plots[0].chunks] == [
        "第一句。",
        "第二句。",
        "第三句。",
    ]
