"""Deterministic OCR sentence preparation and VLM dialogue alignment."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.fuzz import ratio

from rar_agent.workflow.manga.models import (
    MangaScan,
    OcrPageResult,
    OcrSentence,
    VisualExtractionResult,
)

_SENTENCE = re.compile(r".*?[。！？!?；;]+|.+$", re.DOTALL)  # noqa: RUF001


def split_ocr_sentences(pages: list[OcrPageResult]) -> list[OcrSentence]:
    sentences: list[OcrSentence] = []
    for page in pages:
        if page.error is not None:
            continue
        for block in page.blocks:
            for match in _SENTENCE.finditer(block.text):
                text = match.group(0).strip()
                if text:
                    sentences.append(
                        OcrSentence(
                            page_index=page.page_index,
                            text=text,
                            bbox=block.bbox,
                        )
                    )
    return sentences


def align_ocr_results(
    scan: MangaScan,
    visual_results: list[VisualExtractionResult],
    ocr_pages: list[OcrPageResult],
    *,
    threshold: int = 70,
) -> list[VisualExtractionResult]:
    if not 0 <= threshold <= 100:
        raise ValueError("OCR threshold must be between 0 and 100")
    if len(visual_results) != len(scan.batches):
        raise ValueError("every image batch requires one visual extraction result")

    aligned = [result.model_copy(deep=True) for result in visual_results]
    visual_by_page: dict[int, list[tuple[int, int, str]]] = {}
    for batch_position, (batch, result) in enumerate(
        zip(scan.batches, aligned, strict=True)
    ):
        for utterance_position, utterance in enumerate(result.utterances):
            global_page = batch.page_indexes[utterance.page_index]
            visual_by_page.setdefault(global_page, []).append(
                (batch_position, utterance_position, utterance.content)
            )

    ocr_by_page: dict[int, list[OcrSentence]] = {}
    for sentence in split_ocr_sentences(ocr_pages):
        ocr_by_page.setdefault(sentence.page_index, []).append(sentence)

    for page_index, visual_lines in visual_by_page.items():
        ocr_lines = ocr_by_page.get(page_index, [])
        candidates: list[tuple[float, int, int]] = []
        for visual_position, (_, _, visual_text) in enumerate(visual_lines):
            normalized_visual = _match_text(visual_text)
            if not normalized_visual:
                continue
            for ocr_position, ocr_line in enumerate(ocr_lines):
                normalized_ocr = _match_text(ocr_line.text)
                if not normalized_ocr:
                    continue
                score = ratio(normalized_visual, normalized_ocr)
                if score >= threshold:
                    candidates.append((score, visual_position, ocr_position))
        used_visual: set[int] = set()
        used_ocr: set[int] = set()
        for _, visual_position, ocr_position in sorted(
            candidates,
            key=lambda value: (-value[0], value[1], value[2]),
        ):
            if visual_position in used_visual or ocr_position in used_ocr:
                continue
            batch_position, utterance_position, _ = visual_lines[visual_position]
            current = aligned[batch_position].utterances[utterance_position]
            aligned[batch_position].utterances[utterance_position] = current.model_copy(
                update={"content": ocr_lines[ocr_position].text}
            )
            used_visual.add(visual_position)
            used_ocr.add(ocr_position)
    return aligned


def _match_text(value: str) -> str:
    return "".join(
        character.casefold()
        for character in value
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )
