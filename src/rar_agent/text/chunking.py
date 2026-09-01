"""Sentence-aware TextChunk production."""

from __future__ import annotations

from typing import Any

from rar_agent.domain.models import TextChunk
from rar_agent.text.tokenizer import Tokenizer

_SENTENCE_ENDINGS = frozenset("。！？!?；;\n")  # noqa: RUF001 - CJK punctuation
_CLOSING_MARKS = frozenset("”’」』】）》）")  # noqa: RUF001 - CJK closing marks


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] not in _SENTENCE_ENDINGS:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in _CLOSING_MARKS:
            end += 1
        while end < len(text) and text[end].isspace() and text[end] != "\n":
            end += 1
        segment = text[start:end]
        if segment:
            sentences.append(segment)
        start = end
        index = end
    if start < len(text):
        sentences.append(text[start:])
    return [sentence for sentence in sentences if sentence]


class TextChunker:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def chunk_text(
        self,
        file: str,
        text: str,
        *,
        target_tokens: int,
        meta: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        sentences = split_sentences(text)
        chunks: list[TextChunk] = []
        current = ""
        for sentence in sentences:
            candidate = current + sentence
            if current and self.tokenizer.count(candidate) > target_tokens:
                chunks.append(self._make_chunk(file, len(chunks), current, meta))
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(self._make_chunk(file, len(chunks), current, meta))
        return chunks

    def _make_chunk(
        self,
        file: str,
        index: int,
        text: str,
        meta: dict[str, Any] | None,
    ) -> TextChunk:
        return TextChunk(
            file=file,
            index=index,
            text=text,
            token_count=self.tokenizer.count(text),
            meta=meta or {},
        )
