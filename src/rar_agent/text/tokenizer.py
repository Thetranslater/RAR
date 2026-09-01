"""Tokenizer adapters used by deterministic Chunking."""

from __future__ import annotations

from typing import Protocol

import tiktoken
from tiktoken.core import Encoding


class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class CharacterTokenizer:
    """Deterministic lightweight tokenizer for tests and explicit fallback use."""

    name = "characters"

    def count(self, text: str) -> int:
        return len(text)


class TikTokenTokenizer:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        self.name = encoding
        self._encoding: Encoding | None = None

    def count(self, text: str) -> int:
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding(self.name)
        return len(self._encoding.encode(text))
