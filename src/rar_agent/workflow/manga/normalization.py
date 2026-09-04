"""Normalize permissive VLM output into strict batch-local manga results."""

from __future__ import annotations

from rar_agent.workflow.manga.models import (
    ChapterStart,
    ImageBatch,
    LocalCharacter,
    MangaUtterance,
    VisualCharacterPayload,
    VisualExtractionPayload,
    VisualExtractionResult,
)
from rar_agent.workflow.manga.parsing import non_negative_int

_UNKNOWN = {"", "unknown", "null", "none"}


def normalize_visual_extraction(
    payload: VisualExtractionPayload,
    batch: ImageBatch,
) -> VisualExtractionResult:
    characters: dict[int, LocalCharacter] = {}
    raw_order: list[int] = []
    for raw_character in payload.characters:
        old_index = non_negative_int(raw_character.index, "character index")
        if old_index in characters:
            raise ValueError(f"duplicate local character index: {old_index}")
        characters[old_index] = LocalCharacter(
            index=old_index,
            names=_names(raw_character),
            description=raw_character.description.strip(),
        )
        raw_order.append(old_index)

    resolved_utterances: list[tuple[int, int | None, str]] = []
    first_appearance: list[int] = []
    for raw_utterance in payload.utterances:
        page_index = _local_page(raw_utterance.page_index, batch)
        content = raw_utterance.content.strip()
        if not content:
            raise ValueError("utterance content must not be empty")
        speaker = _speaker(raw_utterance.speaker, characters)
        if speaker is not None and speaker not in first_appearance:
            first_appearance.append(speaker)
        resolved_utterances.append((page_index, speaker, content))

    ordered_indexes = [*first_appearance]
    ordered_indexes.extend(index for index in raw_order if index not in first_appearance)
    remap = {old: new for new, old in enumerate(ordered_indexes)}
    normalized_characters = [
        characters[old].model_copy(update={"index": remap[old]})
        for old in ordered_indexes
    ]
    normalized_utterances = [
        MangaUtterance(
            page_index=page_index,
            speaker=remap[speaker] if speaker is not None else None,
            content=content,
        )
        for page_index, speaker, content in resolved_utterances
    ]
    chapter_starts = [
        ChapterStart(
            page_index=_local_page(raw_start.page_index, batch),
            title=(raw_start.title.strip() or None)
            if raw_start.title is not None
            else None,
        )
        for raw_start in payload.chapter_starts
    ]
    return VisualExtractionResult(
        utterances=normalized_utterances,
        characters=normalized_characters,
        chapter_starts=chapter_starts,
        plot=payload.plot.strip(),
    )


def _local_page(value: int | str, batch: ImageBatch) -> int:
    page_index = non_negative_int(value, "page_index")
    if page_index >= len(batch.page_indexes):
        raise ValueError(
            f"page_index {page_index} is outside batch {batch.batch_index}"
        )
    return page_index


def _names(raw: VisualCharacterPayload) -> list[str]:
    values = [raw.names] if isinstance(raw.names, str) else raw.names or []
    names: list[str] = []
    for value in values:
        name = value.strip()
        if name.casefold() in _UNKNOWN or name in names:
            continue
        names.append(name)
    return names


def _speaker(
    value: int | str | None,
    characters: dict[int, LocalCharacter],
) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        speaker = value
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.casefold() in _UNKNOWN:
            return None
        if normalized.isdigit():
            speaker = int(normalized)
        else:
            matches = [
                character.index
                for character in characters.values()
                if normalized in character.names
            ]
            if len(matches) != 1:
                raise ValueError(f"speaker name is not a unique local character: {value}")
            speaker = matches[0]
    else:
        raise ValueError("speaker must reference a local character or be unknown")
    if speaker not in characters:
        raise ValueError(f"speaker references unknown local character: {speaker}")
    return speaker
