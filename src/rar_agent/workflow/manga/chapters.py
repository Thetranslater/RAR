"""Deterministically rebuild manga chapters from explicit visual markers."""

from __future__ import annotations

from rar_agent.workflow.manga.models import (
    ChapterDialogueLine,
    CharacterAssignment,
    MangaChapter,
    MangaChaptersDocument,
    MangaScan,
    VisualExtractionResult,
)


def reconstruct_manga_chapters(
    scan: MangaScan,
    results: list[VisualExtractionResult],
    assignments: list[CharacterAssignment],
    *,
    hard_directory_boundaries: bool = False,
) -> MangaChaptersDocument:
    if len(results) != len(scan.batches):
        raise ValueError("every image batch requires one visual extraction result")
    assignment_map = {
        (value.batch_index, value.local_character_index): value.name
        for value in assignments
    }
    starts: dict[int, str | None] = {}
    global_lines: list[tuple[int, int, str, str]] = []
    sequence = 0
    for batch, result in zip(scan.batches, results, strict=True):
        for start in result.chapter_starts:
            global_page = batch.page_indexes[start.page_index]
            if global_page not in starts or starts[global_page] is None:
                starts[global_page] = start.title
        for line in result.utterances:
            global_page = batch.page_indexes[line.page_index]
            if line.speaker is None:
                speaker = "Unknown"
            else:
                key = (batch.batch_index, line.speaker)
                if key not in assignment_map:
                    raise ValueError(f"missing character assignment for {key}")
                speaker = assignment_map[key] or "Unknown"
            global_lines.append((global_page, sequence, speaker, line.content))
            sequence += 1

    if hard_directory_boundaries:
        previous: object = object()
        for page in scan.pages:
            directory = page.meta.get("directory", "")
            if directory != previous:
                starts.setdefault(page.page_index, str(directory) or None)
                previous = directory

    if not starts:
        starts[0] = None
    first_start = min(starts)
    if first_start > 0 and any(page < first_start for page, *_ in global_lines):
        starts[0] = None

    ordered_starts = sorted(starts)
    chapters: list[MangaChapter] = []
    for position, start_page in enumerate(ordered_starts):
        end_page = (
            ordered_starts[position + 1]
            if position + 1 < len(ordered_starts)
            else len(scan.pages)
        )
        page_indexes = list(range(start_page, end_page))
        if not page_indexes:
            continue
        plot_parts: list[str] = []
        for batch, result in zip(scan.batches, results, strict=True):
            if (
                result.plot
                and any(start_page <= page < end_page for page in batch.page_indexes)
                and result.plot not in plot_parts
            ):
                plot_parts.append(result.plot)
        chapter_lines = [
            ChapterDialogueLine(
                page_index=page,
                speaker=speaker,
                content=content,
            )
            for page, _, speaker, content in sorted(global_lines)
            if start_page <= page < end_page
        ]
        chapters.append(
            MangaChapter(
                index=len(chapters),
                title=starts[start_page],
                page_indexes=page_indexes,
                plot="\n".join(plot_parts),
                utterances=chapter_lines,
            )
        )
    return MangaChaptersDocument(chapters=chapters)
