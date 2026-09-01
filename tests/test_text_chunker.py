from rar_agent.text.chunking import TextChunker
from rar_agent.text.tokenizer import CharacterTokenizer


def test_chunker_prefers_complete_sentences_and_uses_file_local_indexes() -> None:
    chunker = TextChunker(CharacterTokenizer())

    chunks = chunker.chunk_text(
        "resources/book.txt",
        "第一句。第二句很长。第三句。",
        target_tokens=8,
        meta={"chapter": "第一章"},
    )

    assert [(chunk.index, chunk.text) for chunk in chunks] == [
        (0, "第一句。"),
        (1, "第二句很长。"),
        (2, "第三句。"),
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
