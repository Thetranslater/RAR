import pytest

from rar_agent.workflow.manga.chapters import reconstruct_manga_chapters
from rar_agent.workflow.manga.characters import (
    clean_character_catalog,
    validate_character_assignments,
)
from rar_agent.workflow.manga.models import (
    CharacterAssignment,
    CharacterAssignmentPayload,
    CharacterObservation,
    ImageBatch,
    ImagePage,
    MangaScan,
    NamedCharacterCatalogPayload,
    OcrBlock,
    OcrPageResult,
    VisualExtractionPayload,
    VisualExtractionResult,
)
from rar_agent.workflow.manga.normalization import normalize_visual_extraction
from rar_agent.workflow.manga.ocr import align_ocr_results, split_ocr_sentences


def test_visual_extraction_normalizes_local_indexes_names_and_speakers() -> None:
    batch = ImageBatch(batch_index=4, page_indexes=[20, 21, 22])
    payload = VisualExtractionPayload.model_validate(
        {
            "utterances": [
                {"page_index": "0", "speaker": "7", "content": " First "},
                {"page_index": 1, "speaker": "Alice", "content": "Second"},
                {"page_index": 2, "speaker": "unknown", "content": "Third"},
            ],
            "characters": [
                {
                    "index": "2",
                    "names": ["Alice", " Alice ", "unknown"],
                    "description": "dark hair",
                },
                {"index": 7, "names": None, "description": "white hair"},
            ],
            "chapter_starts": [{"page_index": "1", "title": " Chapter 1 "}],
            "plot": " Plot ",
        }
    )

    result = normalize_visual_extraction(payload, batch)

    assert [character.model_dump() for character in result.characters] == [
        {"index": 0, "names": [], "description": "white hair"},
        {"index": 1, "names": ["Alice"], "description": "dark hair"},
    ]
    assert [utterance.model_dump() for utterance in result.utterances] == [
        {"page_index": 0, "speaker": 0, "content": "First"},
        {"page_index": 1, "speaker": 1, "content": "Second"},
        {"page_index": 2, "speaker": None, "content": "Third"},
    ]
    assert result.chapter_starts[0].model_dump() == {
        "page_index": 1,
        "title": "Chapter 1",
    }
    assert result.plot == "Plot"


def test_visual_extraction_rejects_unknown_local_references() -> None:
    payload = VisualExtractionPayload.model_validate(
        {
            "utterances": [{"page_index": 0, "speaker": 9, "content": "Hi"}],
            "characters": [],
            "chapter_starts": [],
            "plot": "",
        }
    )

    with pytest.raises(ValueError, match="unknown local character"):
        normalize_visual_extraction(
            payload,
            ImageBatch(batch_index=0, page_indexes=[0]),
        )


def test_character_catalog_cleanup_keeps_aliases_and_merges_exact_names() -> None:
    catalog = clean_character_catalog(
        NamedCharacterCatalogPayload.model_validate(
            {
                "characters": [
                    {
                        "name": " Alice ",
                        "aliases": ["Al", "Alice", "Al", "unknown"],
                        "description": "dark hair",
                    },
                    {
                        "name": "Alice",
                        "aliases": ["A"],
                        "description": "school uniform",
                    },
                    {"name": "unknown", "aliases": [], "description": "ignored"},
                ]
            }
        )
    )

    assert catalog.model_dump() == {
        "characters": [
            {
                "name": "Alice",
                "aliases": ["Al", "A"],
                "description": "dark hair\nschool uniform",
            }
        ]
    }


def test_character_assignments_resolve_unique_aliases_and_require_every_role() -> None:
    observations = [
        CharacterObservation(
            batch_index=0,
            local_character_index=0,
            names=["Al"],
            description="dark hair",
        ),
        CharacterObservation(
            batch_index=0,
            local_character_index=1,
            names=[],
            description="white hair",
        ),
    ]
    catalog = clean_character_catalog(
        NamedCharacterCatalogPayload.model_validate(
            {
                "characters": [
                    {"name": "Alice", "aliases": ["Al"], "description": "dark"}
                ]
            }
        )
    )

    assignments = validate_character_assignments(
        CharacterAssignmentPayload.model_validate(
            {
                "assignments": [
                    {"batch_index": 0, "local_character_index": 0, "name": "Al"},
                    {"batch_index": 0, "local_character_index": 1, "name": None},
                ]
            }
        ),
        observations,
        catalog,
    )

    assert [assignment.model_dump() for assignment in assignments] == [
        {"batch_index": 0, "local_character_index": 0, "name": "Alice"},
        {"batch_index": 0, "local_character_index": 1, "name": None},
    ]

    with pytest.raises(ValueError, match="exactly once"):
        validate_character_assignments(
            CharacterAssignmentPayload(assignments=[]),
            observations,
            catalog,
        )


def test_chapter_reconstruction_uses_explicit_starts_and_shares_crossing_plot() -> None:
    scan = MangaScan(
        resource_path="manga",
        pages=[
            ImagePage(page_index=index, path=f"manga/{index}.jpg")
            for index in range(6)
        ],
        batches=[
            ImageBatch(batch_index=0, page_indexes=[0, 1, 2]),
            ImageBatch(batch_index=1, page_indexes=[3, 4, 5]),
        ],
    )
    results = [
        VisualExtractionResult.model_validate(
            {
                "utterances": [
                    {"page_index": 0, "speaker": 0, "content": "Preface"},
                    {"page_index": 1, "speaker": 0, "content": "Chapter one"},
                ],
                "characters": [
                    {"index": 0, "names": ["Alice"], "description": "dark"}
                ],
                "chapter_starts": [{"page_index": 1, "title": "One"}],
                "plot": "batch zero plot",
            }
        ),
        VisualExtractionResult.model_validate(
            {
                "utterances": [
                    {"page_index": 0, "speaker": 0, "content": "Still one"},
                    {"page_index": 1, "speaker": 0, "content": "Chapter two"},
                ],
                "characters": [
                    {"index": 0, "names": ["Bob"], "description": "white"}
                ],
                "chapter_starts": [{"page_index": 1, "title": "Two"}],
                "plot": "batch one plot",
            }
        ),
    ]
    assignments = [
        CharacterAssignment(batch_index=0, local_character_index=0, name="Alice"),
        CharacterAssignment(batch_index=1, local_character_index=0, name="Bob"),
    ]

    chapters = reconstruct_manga_chapters(scan, results, assignments)

    assert [chapter.title for chapter in chapters.chapters] == [None, "One", "Two"]
    assert [chapter.page_indexes for chapter in chapters.chapters] == [
        [0],
        [1, 2, 3],
        [4, 5],
    ]
    assert [chapter.plot for chapter in chapters.chapters] == [
        "batch zero plot",
        "batch zero plot\nbatch one plot",
        "batch one plot",
    ]
    assert [
        [(line.page_index, line.speaker, line.content) for line in chapter.utterances]
        for chapter in chapters.chapters
    ] == [
        [(0, "Alice", "Preface")],
        [(1, "Alice", "Chapter one"), (3, "Bob", "Still one")],
        [(4, "Bob", "Chapter two")],
    ]


def test_ocr_alignment_replaces_only_one_matching_dialogue_on_the_same_page() -> None:
    scan = MangaScan(
        resource_path="manga",
        pages=[ImagePage(page_index=0, path="manga/0.jpg")],
        batches=[ImageBatch(batch_index=0, page_indexes=[0])],
    )
    visual = VisualExtractionResult.model_validate(
        {
            "utterances": [
                {"page_index": 0, "speaker": None, "content": "你 好 !"},
                {"page_index": 0, "speaker": None, "content": "其他对白"},
            ],
            "characters": [],
            "chapter_starts": [],
            "plot": "",
        }
    )
    ocr = OcrPageResult(
        page_index=0,
        blocks=[OcrBlock(text="你好！\n背景文字", bbox=[0, 0, 10, 10])],  # noqa: RUF001
    )

    sentences = split_ocr_sentences([ocr])
    aligned = align_ocr_results(scan, [visual], [ocr], threshold=70)

    assert [sentence.text for sentence in sentences] == [
        "你好！",  # noqa: RUF001
        "背景文字",
    ]
    assert [line.content for line in aligned[0].utterances] == [
        "你好！",  # noqa: RUF001
        "其他对白",
    ]


def test_ocr_failure_keeps_visual_dialogue() -> None:
    scan = MangaScan(
        resource_path="manga",
        pages=[ImagePage(page_index=0, path="manga/0.jpg")],
        batches=[ImageBatch(batch_index=0, page_indexes=[0])],
    )
    visual = VisualExtractionResult.model_validate(
        {
            "utterances": [
                {"page_index": 0, "speaker": None, "content": "Keep me"}
            ],
            "characters": [],
            "chapter_starts": [],
            "plot": "",
        }
    )

    aligned = align_ocr_results(
        scan,
        [visual],
        [OcrPageResult(page_index=0, blocks=[], error="worker failed")],
    )

    assert aligned[0].utterances[0].content == "Keep me"
