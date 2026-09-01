"""Split Chinese novels into chapter-aware, sentence-complete token chunks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

NUMBER_PATTERN = r"[零〇一二两三四五六七八九十百千万两\d]+"
VOLUME_PATTERN = rf"第{NUMBER_PATTERN}[卷部篇]"

# A volume heading is a complete line whose first non-space characters are a
# volume marker. Its suffix may contain a chapter marker and title text.
DEFAULT_HEADING_RE = re.compile(
    rf"(?m)^[ \t\u3000]*{VOLUME_PATTERN}[^\r\n]*$"
)
VOLUME_RE = re.compile(VOLUME_PATTERN)

SENTENCE_END_RE = re.compile(
    r"(?:[。！？!?]+)"
    r"[”’\"」』】）》〉〕〗〙〛）)\]]*"
    r"|\n[ \t\u3000]*\n"
)
class TokenCounter(Protocol):
    """Minimal token counter interface used by the chunking algorithm."""

    @property
    def name(self) -> str: ...

    def count(self, text: str) -> int: ...


class TiktokenCounter:
    """Count tokens with a named tiktoken encoding."""

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised by CLI environments
            raise RuntimeError(
                "tiktoken is required for token-aware chunking. "
                "Install the project dependencies in the rlff-dev environment."
            ) from exc
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._name = encoding_name

    @property
    def name(self) -> str:
        return self._name

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))


@dataclass(frozen=True)
class Section:
    """Content belonging to one detected volume-title line."""

    section_index: int
    heading: str
    volume: str | None
    chapter: str | None
    text: str


@dataclass(frozen=True)
class Chunk:
    """A sentence-complete piece of one section."""

    chunk_id: str
    section_index: int
    chunk_index: int
    heading: str
    volume: str | None
    chapter: str | None
    token_count: int
    text: str


def read_text(path: Path, encoding: str = "auto") -> tuple[str, str]:
    """Read UTF-8 or common legacy Chinese text without silent replacement."""

    raw = path.read_bytes()
    candidates = ("utf-8-sig", "gb18030") if encoding == "auto" else (encoding,)
    errors: list[str] = []
    for candidate in candidates:
        try:
            text = raw.decode(candidate)
            return normalize_newlines(text), candidate
        except (LookupError, UnicodeDecodeError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise UnicodeError(f"Unable to decode {path}: {'; '.join(errors)}")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _compile_marker_parts(splitters: list[str]) -> list[str]:
    patterns: list[str] = []
    for splitter in splitters:
        if not splitter:
            raise ValueError("title splitter cannot be empty")
        if splitter.startswith("re:"):
            raw_pattern = splitter[3:]
            if not raw_pattern:
                raise ValueError("A re: splitter must contain a regular expression")
            patterns.append(f"(?:{raw_pattern})")
            continue
        marker = re.escape(splitter).replace(re.escape("{num}"), NUMBER_PATTERN)
        patterns.append(f"(?:{marker})")
    return patterns


def compile_title_pattern(
    splitters: list[str] | None,
    *,
    default_pattern: re.Pattern[str] | None = None,
) -> re.Pattern[str]:
    """Compile a complete-line title matcher.

    Literal values use ``{num}`` as a Chinese/Arabic number placeholder. Values
    beginning with ``re:`` are raw regular expressions. The resulting matcher
    always requires a complete line whose first non-space characters match the
    supplied title pattern.
    """

    if not splitters:
        if default_pattern is None:
            raise ValueError("a default title pattern is required")
        return default_pattern

    combined = "|".join(_compile_marker_parts(splitters))
    return re.compile(rf"(?m)^[ \t\u3000]*(?:{combined})[^\r\n]*$")


def compile_title_search_pattern(splitters: list[str]) -> re.Pattern[str]:
    """Compile an unanchored matcher for a chapter marker in a volume line."""

    return re.compile("|".join(_compile_marker_parts(splitters)))


def compile_heading_pattern(splitters: list[str] | None) -> re.Pattern[str]:
    """Backward-compatible alias for compiling volume-style title lines."""

    return compile_title_pattern(splitters, default_pattern=DEFAULT_HEADING_RE)


@dataclass(frozen=True)
class _TitleLine:
    start: int
    end: int
    heading: str
    volume: str
    chapter: str | None


def _append_section(
    sections: list[Section],
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
    if not body:
        return
    sections.append(
        Section(
            section_index=len(sections),
            heading=heading,
            volume=volume,
            chapter=chapter,
            text=body,
        )
    )


def _chapter_in_volume_heading(
    heading: str,
    *,
    volume_match: re.Match[str],
    chapter_search_pattern: re.Pattern[str] | None,
) -> str | None:
    if chapter_search_pattern is None:
        return None
    chapter_match = chapter_search_pattern.search(heading, volume_match.end())
    return heading[chapter_match.start() :].strip() if chapter_match else None


def split_sections(
    text: str,
    *,
    volume_pattern: re.Pattern[str] = DEFAULT_HEADING_RE,
    volume_search_pattern: re.Pattern[str] = VOLUME_RE,
    chapter_pattern: re.Pattern[str] | None = None,
    chapter_search_pattern: re.Pattern[str] | None = None,
    # Retain the old keyword for callers that imported split_sections directly.
    heading_pattern: re.Pattern[str] | None = None,
    include_heading: bool = False,
    include_front_matter: bool = False,
) -> list[Section]:
    """Split text into chapter sections, falling back to volume sections.

    A chapter marker may occur in a volume heading line, or it may be a
    complete line inside the volume. Repeated lines with the same volume marker
    are kept in one volume region so formats such as ``第一部 第一话 ...`` do
    not accidentally create one volume per chapter.
    """

    if heading_pattern is not None:
        volume_pattern = heading_pattern

    volume_matches = list(volume_pattern.finditer(text))
    chapter_search_pattern = chapter_search_pattern or (
        chapter_pattern if chapter_pattern is not None else None
    )

    if not volume_matches:
        # With no detectable volumes, a supplied chapter pattern can still be
        # used as a document-wide chapter splitter.
        if chapter_pattern is not None:
            chapter_matches = list(chapter_pattern.finditer(text))
            if chapter_matches:
                sections: list[Section] = []
                _append_section(
                    sections,
                    text=text,
                    start=0,
                    end=chapter_matches[0].start(),
                    heading="",
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
        return (
            [Section(section_index=0, heading="", volume=None, chapter=None, text=body)]
            if body
            else []
        )

    groups: list[tuple[str, list[_TitleLine]]] = []
    for match in volume_matches:
        heading = match.group(0).strip()
        volume_match = volume_search_pattern.search(heading)
        volume = volume_match.group(0).strip() if volume_match else heading
        chapter = _chapter_in_volume_heading(
            heading,
            volume_match=volume_match,
            chapter_search_pattern=chapter_search_pattern,
        ) if volume_match else None
        item = _TitleLine(match.start(), match.end(), heading, volume, chapter)
        if groups and groups[-1][0] == volume:
            groups[-1][1].append(item)
        else:
            groups.append((volume, [item]))

    sections = []
    first_volume_line = groups[0][1][0]
    if include_front_matter:
        _append_section(
            sections,
            text=text,
            start=0,
            end=first_volume_line.start,
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
        same_line_chapters = [line for line in volume_lines if line.chapter is not None]

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
            # Preserve a volume preamble (for example an unmatched 序章) as a
            # volume-level section instead of dropping source text.
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


def chunk_section(section: Section, budget: int, counter: TokenCounter) -> list[Chunk]:
    """Accumulate complete sentences until a chunk reaches or crosses budget."""

    if budget < 1:
        raise ValueError("token budget must be a positive integer")

    chunks: list[Chunk] = []
    pending: list[str] = []

    def emit() -> None:
        text = "".join(pending)
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"s{section.section_index:05d}-c{chunk_index:04d}",
                section_index=section.section_index,
                chunk_index=chunk_index,
                heading=section.heading,
                volume=section.volume,
                chapter=section.chapter,
                token_count=counter.count(text),
                text=text,
            )
        )
        pending.clear()

    for sentence in split_sentences(section.text):
        pending.append(sentence)
        candidate = "".join(pending)
        if counter.count(candidate) >= budget:
            emit()

    if pending:
        emit()
    return chunks


def chunk_book(sections: list[Section], budget: int, counter: TokenCounter) -> list[Chunk]:
    return [
        chunk
        for section in sections
        for chunk in chunk_section(section, budget, counter)
    ]


def build_output(
    *,
    source_path: Path,
    title: str,
    source_encoding: str,
    budget: int,
    counter: TokenCounter,
    splitters: list[str] | None,
    sections: list[Section],
    chunks: list[Chunk],
) -> dict[str, object]:
    return {
        "schema_version": "rlff-plot-chunks-v1",
        "title": title,
        "source": {
            "path": str(source_path.resolve()),
            "encoding": source_encoding,
        },
        "chunking": {
            "token_budget": budget,
            "token_encoding": counter.name,
            "splitters": splitters or [],
            "section_count": len(sections),
            "chunk_count": len(chunks),
        },
        "chunks": [
            {
                "volume": chunk.volume,
                "chapter": chunk.chapter,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a Chinese novel by volume/chapter headings, then create "
            "sentence-complete chunks near a token budget."
        )
    )
    parser.add_argument("input_txt", type=Path, help="Input novel .txt path.")
    parser.add_argument("output_json", type=Path, help="Output chunk JSON path.")
    parser.add_argument(
        "--token",
        type=int,
        default=4096,
        help="Target token budget per chunk (default: 4096).",
    )
    parser.add_argument(
        "--volume-splitter",
        action="append",
        help=(
            "Custom volume heading marker; repeat as needed. Use {num} for "
            "Chinese/Arabic numbers or prefix with re: for a regular expression. "
            "Default: a line beginning with 第…卷/部/篇."
        ),
    )
    parser.add_argument(
        "--chapter-splitter",
        action="append",
        help=(
            "Custom chapter heading marker; repeat as needed. Use {num} for "
            "Chinese/Arabic numbers or prefix with re: for a regular expression."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="auto",
        help="Source text encoding; auto tries UTF-8 and GB18030 (default: auto).",
    )
    parser.add_argument(
        "--token-encoding",
        default="o200k_base",
        help="tiktoken encoding used for budget accounting (default: o200k_base).",
    )
    parser.add_argument(
        "--title",
        help="Book title in output metadata (default: input filename stem).",
    )
    parser.add_argument(
        "--include-heading",
        action="store_true",
        help="Include each heading at the start of its section text.",
    )
    parser.add_argument(
        "--include-front-matter",
        action="store_true",
        help="Keep text before the first recognized heading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.token < 1:
        raise SystemExit("--token must be a positive integer")

    text, source_encoding = read_text(args.input_txt, args.encoding)
    try:
        volume_pattern = compile_title_pattern(
            args.volume_splitter,
            default_pattern=DEFAULT_HEADING_RE,
        )
        volume_search_pattern = (
            VOLUME_RE
            if not args.volume_splitter
            else compile_title_search_pattern(args.volume_splitter)
        )
        chapter_pattern = (
            compile_title_pattern(args.chapter_splitter)
            if args.chapter_splitter
            else None
        )
        chapter_search_pattern = (
            compile_title_search_pattern(args.chapter_splitter)
            if args.chapter_splitter
            else None
        )
        counter = TiktokenCounter(args.token_encoding)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    sections = split_sections(
        text,
        volume_pattern=volume_pattern,
        volume_search_pattern=volume_search_pattern,
        chapter_pattern=chapter_pattern,
        chapter_search_pattern=chapter_search_pattern,
        include_heading=args.include_heading,
        include_front_matter=args.include_front_matter,
    )
    chunks = chunk_book(sections, args.token, counter)
    output = build_output(
        source_path=args.input_txt,
        title=args.title or args.input_txt.stem,
        source_encoding=source_encoding,
        budget=args.token,
        counter=counter,
        splitters=(args.volume_splitter or []) + (args.chapter_splitter or []),
        sections=sections,
        chunks=chunks,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(chunks)} chunks from {len(sections)} sections "
        f"to {args.output_json}"
    )


if __name__ == "__main__":
    main()
