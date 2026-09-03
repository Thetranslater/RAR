from rar_agent.text.chunking import TextChunker
from rar_agent.text.tokenizer import CharacterTokenizer


def test_chunker_accumulates_complete_sentences_until_budget() -> None:
    chunker = TextChunker(CharacterTokenizer())

    chunks = chunker.chunk_text(
        "resources/book.txt",
        "第一句。第二句很长。第三句。",
        target_tokens=8,
        meta={"chapter": "第一章"},
    )

    assert [(chunk.index, chunk.text) for chunk in chunks] == [
        (0, "第一句。第二句很长。"),
        (1, "第三句。"),
    ]
    assert all(chunk.file == "resources/book.txt" for chunk in chunks)
    assert all(chunk.meta == {"chapter": "第一章"} for chunk in chunks)


def test_oversized_sentence_is_kept_whole() -> None:
    chunks = TextChunker(CharacterTokenizer()).chunk_text(
        "resources/book.txt",
        "这是一个远远超过预算但必须保持完整的句子。短句。",
        target_tokens=5,
    )

    assert chunks[0].text == "这是一个远远超过预算但必须保持完整的句子。"


def test_chunker_splits_volume_and_chapter_before_token_chunking() -> None:
    chunks = TextChunker(CharacterTokenizer()).chunk_text(
        "resources/book.txt",
        (
            "轻小说文库\n\n"
            "第一卷 关野帆乃佳篇 序曲\n\n"
            "秋吉走进教室。帆乃佳向他挥手。\n\n"
            "第二卷 新的故事 第一章 相遇\n\n"
            "两人在车站再次相遇。"
        ),
        target_tokens=1_000,
        meta={"language": "zh"},
    )

    assert [chunk.text for chunk in chunks] == [
        "秋吉走进教室。帆乃佳向他挥手。",
        "两人在车站再次相遇。",
    ]
    assert [chunk.meta for chunk in chunks] == [
        {"language": "zh", "volume": "第一卷", "chapter": "序曲"},
        {"language": "zh", "volume": "第二卷", "chapter": "第一章 相遇"},
    ]
