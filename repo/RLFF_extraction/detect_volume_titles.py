"""Detect and print loose, complete-line volume titles without splitting text."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from script.build.chunker import DEFAULT_HEADING_RE, read_text
except ModuleNotFoundError:  # Allows direct execution from this directory.
    from chunker import DEFAULT_HEADING_RE, read_text


@dataclass(frozen=True)
class DetectedVolumeTitle:
    line_number: int
    title: str


def detect_volume_titles(text: str) -> list[DetectedVolumeTitle]:
    """Return every complete line beginning with 第…卷/部/篇."""

    return [
        DetectedVolumeTitle(
            line_number=text.count("\n", 0, match.start()) + 1,
            title=match.group(0).strip(),
        )
        for match in DEFAULT_HEADING_RE.finditer(text)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect complete-line volume titles beginning with 第…卷/部/篇. "
            "The novel is not split or modified."
        )
    )
    parser.add_argument("input_txt", type=Path, help="Input novel .txt path.")
    parser.add_argument(
        "--encoding",
        default="auto",
        help="Source encoding; auto tries UTF-8 and GB18030 (default: auto).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional UTF-8 text file. By default titles are printed to stdout.",
    )
    parser.add_argument(
        "--line-numbers",
        action="store_true",
        help="Prefix each title with its 1-based source line number.",
    )
    return parser.parse_args()


def format_title(item: DetectedVolumeTitle, *, line_numbers: bool) -> str:
    return f"{item.line_number}\t{item.title}" if line_numbers else item.title


def main() -> None:
    args = parse_args()
    try:
        text, source_encoding = read_text(args.input_txt, args.encoding)
    except (OSError, UnicodeError) as exc:
        raise SystemExit(str(exc)) from exc

    titles = detect_volume_titles(text)
    output = "\n".join(
        format_title(item, line_numbers=args.line_numbers) for item in titles
    )
    if output:
        output += "\n"

    if args.output is None:
        sys.stdout.write(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(
            f"Detected {len(titles)} volume titles using {source_encoding}; "
            f"wrote {args.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
