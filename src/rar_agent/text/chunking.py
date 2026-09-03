"""Chapter-aware, sentence-complete TextChunk production."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rar_agent.domain.models import TextChunk
from rar_agent.text.tokenizer import Tokenizer

NUMBER_PATTERN = r"[零〇一二两三四五六七八九十百千万兩\d]+"
VOLUME_PATTERN = rf"第{NUMBER_PATTERN}[卷部篇]"
DEFAULT_VOLUME_HEADING_RE = re.compile(
    rf"(?m)^[ \t\u3000]*{VOLUME_PATTERN}[^\r\n]*$"
)
VOLUME_SEARCH_RE = re.compile(VOLUME_PATTERN)
DEFAULT_CHAPTER_SPLITTERS = (
    "第{num}章",
    "第{num}话",
    "第{num}節",
    "第{num}节",
    "序章",
    "序曲",
    "终章",
    "終章",
    "尾声",
    "尾聲",
    "后记",
    "後記",
)
SENTENCE_END_RE = re.compile(
    r'(?:[。！？!?]+)[”’"」』】））》〉〕〗〙〛）)\]]*|\n[ \t\u3000]*\n'  # noqa: RUF001
)


@dataclass(frozen=True, slots=True)
class TextSection:
    """Text associated with one detected volume/chapter heading."""

    heading: str
    volume: str | None
    chapter: str | None
    text: str


@dataclass(frozen=True, slots=True)
class _TitleLine:
    start: int
    end: int
    heading: str
    volume: str
    chapter: str | None


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _compile_marker_parts(splitters: Sequence[str]) -> list[str]:
    patterns: list[str] = []
    for splitter in splitters:
        if not splitter:
            raise ValueError("title splitter cannot be empty")
        if splitter.startswith("re:"):
            raw_pattern = splitter[3:]
            if not raw_pattern:
                raise ValueError("a re: splitter must contain a regular expression")
            patterns.append(f"(?:{raw_pattern})")
            continue
        marker = re.escape(splitter).replace(re.escape("{num}"), NUMBER_PATTERN)
        patterns.append(f"(?:{marker})")
    return patterns


def compile_title_pattern(
    splitters: Sequence[str] | None,
    *,
    default_pattern: re.Pattern[str] | None = None,
) -> re.Pattern[str]:
    if not splitters:
        if default_pattern is None:
            raise ValueError("a default title pattern is required")
        return default_pattern
    combined = "|".join(_compile_marker_parts(splitters))
    return re.compile(rf"(?m)^[ \t\u3000]*(?:{combined})[^\r\n]*$")


def compile_title_search_pattern(splitters: Sequence[str]) -> re.Pattern[str]:
    return re.compile("|".join(_compile_marker_parts(splitters)))


def _append_section(
    sections: list[TextSection],
    *,
    text: str,
    start: int,
    end: int,
    heading: str,
    volume: str | None,
    chapter: str | None,
    include_heading: bool,
) -> None:
    body = text[start:end].strip()
    if include_heading:
        body = f"{heading}\n{body}".strip()
    if body:
        sections.append(TextSection(heading, volume, chapter, body))


def _chapter_in_volume_heading(
    heading: str,
    *,
    volume_match: re.Match[str],
    chapter_search_pattern: re.Pattern[str] | None,
) -> str | None:
    if chapter_search_pattern is None:
        return None
    match = chapter_search_pattern.search(heading, volume_match.end())
    return heading[match.start() :].strip() if match else None


def split_sections(
    text: str,
    *,
    volume_splitters: Sequence[str] | None = None,
    chapter_splitters: Sequence[str] | None = None,
    include_heading: bool = False,
    include_front_matter: bool = False,
) -> list[TextSection]:
    """Split by complete-line volume/chapter headings before token chunking."""

    text = normalize_newlines(text)
    volume_pattern = compile_title_pattern(
        volume_splitters, default_pattern=DEFAULT_VOLUME_HEADING_RE
    )
    volume_search_pattern = (
        VOLUME_SEARCH_RE
        if not volume_splitters
        else compile_title_search_pattern(volume_splitters)
    )
    chapter_values = (
        DEFAULT_CHAPTER_SPLITTERS
        if chapter_splitters is None
        else tuple(chapter_splitters)
    )
    chapter_pattern = (
        compile_title_pattern(chapter_values) if chapter_values else None
    )
    chapter_search_pattern = (
        compile_title_search_pattern(chapter_values) if chapter_values else None
    )
    volume_matches = list(volume_pattern.finditer(text))

    if not volume_matches:
        if chapter_pattern is not None:
            chapter_matches = list(chapter_pattern.finditer(text))
            if chapter_matches:
                sections: list[TextSection] = []
                if include_front_matter:
                    _append_section(
                        sections,
                        text=text,
                        start=0,
                        end=chapter_matches[0].start(),
                        heading="前置内容",
                        volume=None,
                        chapter=None,
                        include_heading=False,
                    )
                for index, match in enumerate(chapter_matches):
                    heading = match.group(0).strip()
                    body_end = (
                        chapter_matches[index + 1].start()
                        if index + 1 < len(chapter_matches)
                        else len(text)
                    )
                    _append_section(
                        sections,
                        text=text,
                        start=match.end(),
                        end=body_end,
                        heading=heading,
                        volume=None,
                        chapter=heading or None,
                        include_heading=include_heading,
                    )
                return sections
        body = text.strip()
        return [TextSection("", None, None, body)] if body else []

    groups: list[tuple[str, list[_TitleLine]]] = []
    for match in volume_matches:
        heading = match.group(0).strip()
        volume_match = volume_search_pattern.search(heading)
        volume = volume_match.group(0).strip() if volume_match else heading
        chapter = (
            _chapter_in_volume_heading(
                heading,
                volume_match=volume_match,
                chapter_search_pattern=chapter_search_pattern,
            )
            if volume_match
            else None
        )
        line = _TitleLine(match.start(), match.end(), heading, volume, chapter)
        if groups and groups[-1][0] == volume:
            groups[-1][1].append(line)
        else:
            groups.append((volume, [line]))

    sections = []
    if include_front_matter:
        _append_section(
            sections,
            text=text,
            start=0,
            end=groups[0][1][0].start,
            heading="前置内容",
            volume=None,
            chapter=None,
            include_heading=False,
        )

    for group_index, (volume, volume_lines) in enumerate(groups):
        region_end = (
            groups[group_index + 1][1][0].start
            if group_index + 1 < len(groups)
            else len(text)
        )
        same_line_chapters = [line for line in volume_lines if line.chapter]
        if same_line_chapters:
            for index, line in enumerate(same_line_chapters):
                body_end = (
                    same_line_chapters[index + 1].start
                    if index + 1 < len(same_line_chapters)
                    else region_end
                )
                _append_section(
                    sections,
                    text=text,
                    start=line.end,
                    end=body_end,
                    heading=line.heading,
                    volume=volume,
                    chapter=line.chapter,
                    include_heading=include_heading,
                )
            continue

        body_start = volume_lines[0].end
        chapter_matches = (
            list(chapter_pattern.finditer(text, body_start, region_end))
            if chapter_pattern is not None
            else []
        )
        if chapter_matches:
            _append_section(
                sections,
                text=text,
                start=body_start,
                end=chapter_matches[0].start(),
                heading=volume_lines[0].heading,
                volume=volume,
                chapter=None,
                include_heading=include_heading,
            )
            for index, match in enumerate(chapter_matches):
                heading = match.group(0).strip()
                body_end = (
                    chapter_matches[index + 1].start()
                    if index + 1 < len(chapter_matches)
                    else region_end
                )
                _append_section(
                    sections,
                    text=text,
                    start=match.end(),
                    end=body_end,
                    heading=heading,
                    volume=volume,
                    chapter=heading or None,
                    include_heading=include_heading,
                )
        else:
            _append_section(
                sections,
                text=text,
                start=body_start,
                end=region_end,
                heading=volume_lines[0].heading,
                volume=volume,
                chapter=None,
                include_heading=include_heading,
            )
    return sections


def split_sentences(text: str) -> list[str]:
    """Return sentence/paragraph units without dropping source characters."""

    if not text or not text.strip():
        return []
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        if match.end() <= start:
            continue
        end = match.end()
        while end < len(text) and text[end].isspace():
            end += 1
        sentences.append(text[start:end])
        start = end
    if start < len(text):
        sentences.append(text[start:])
    return sentences


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
        volume_splitters: Sequence[str] | None = None,
        chapter_splitters: Sequence[str] | None = None,
        include_heading: bool = False,
        include_front_matter: bool = False,
    ) -> list[TextChunk]:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        sections = split_sections(
            text,
            volume_splitters=volume_splitters,
            chapter_splitters=chapter_splitters,
            include_heading=include_heading,
            include_front_matter=include_front_matter,
        )
        chunks: list[TextChunk] = []
        for section in sections:
            section_meta = dict(meta or {})
            if section.volume is not None:
                section_meta["volume"] = section.volume
            if section.chapter is not None:
                section_meta["chapter"] = section.chapter
            pending: list[str] = []
            for sentence in split_sentences(section.text):
                pending.append(sentence)
                candidate = "".join(pending)
                if self.tokenizer.count(candidate) >= target_tokens:
                    chunks.append(
                        self._make_chunk(file, len(chunks), candidate, section_meta)
                    )
                    pending.clear()
            if pending:
                chunks.append(
                    self._make_chunk(file, len(chunks), "".join(pending), section_meta)
                )
        return chunks

    def _make_chunk(
        self,
        file: str,
        index: int,
        text: str,
        meta: dict[str, Any],
    ) -> TextChunk:
        return TextChunk(
            file=file,
            index=index,
            text=text,
            token_count=self.tokenizer.count(text),
            meta=meta,
        )
