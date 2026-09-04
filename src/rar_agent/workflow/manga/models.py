"""Persisted and model-facing contracts for the Manga Workflow."""

from __future__ import annotations

from pathlib import PurePath, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonObject = dict[str, Any]


def workspace_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        raise ValueError("path must not be empty")
    if PurePath(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise ValueError("path must be workspace-relative")
    if ".." in PurePath(normalized).parts:
        raise ValueError("path must be workspace-relative")
    return normalized


class MangaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImagePage(MangaModel):
    page_index: int = Field(ge=0)
    path: str
    meta: JsonObject = Field(default_factory=dict)

    _validate_path = field_validator("path")(workspace_relative_path)


class ImageBatch(MangaModel):
    batch_index: int = Field(ge=0)
    page_indexes: list[int] = Field(min_length=1)

    @field_validator("page_indexes")
    @classmethod
    def validate_page_indexes(cls, indexes: list[int]) -> list[int]:
        if any(index < 0 for index in indexes):
            raise ValueError("page indexes must not be negative")
        if indexes != sorted(set(indexes)):
            raise ValueError("page indexes must be unique and ordered")
        return indexes


class MangaScan(MangaModel):
    resource_path: str
    pages: list[ImagePage] = Field(min_length=1)
    batches: list[ImageBatch] = Field(min_length=1)

    _validate_resource_path = field_validator("resource_path")(workspace_relative_path)

    @model_validator(mode="after")
    def validate_indexes(self) -> MangaScan:
        if [page.page_index for page in self.pages] != list(range(len(self.pages))):
            raise ValueError("page indexes must be contiguous")
        if [batch.batch_index for batch in self.batches] != list(
            range(len(self.batches))
        ):
            raise ValueError("batch indexes must be contiguous")
        page_indexes = {page.page_index for page in self.pages}
        referenced = [index for batch in self.batches for index in batch.page_indexes]
        if referenced != list(range(len(self.pages))) or set(referenced) != page_indexes:
            raise ValueError("batches must reference every page exactly once")
        return self


class MangaPreviewIssue(MangaModel):
    path: str
    reason: str


class MangaPreview(MangaModel):
    path: str
    image_count: int = Field(ge=0)
    batch_count: int = Field(ge=0)
    first_paths: list[str]
    skipped: list[MangaPreviewIssue]
    errors: list[MangaPreviewIssue]


IndexValue = int | str


class VisualUtterancePayload(MangaModel):
    page_index: IndexValue
    speaker: IndexValue | None = None
    content: str


class VisualCharacterPayload(MangaModel):
    index: IndexValue
    names: list[str] | str | None = None
    description: str = ""


class ChapterStartPayload(MangaModel):
    page_index: IndexValue
    title: str | None = None


class VisualExtractionPayload(MangaModel):
    utterances: list[VisualUtterancePayload]
    characters: list[VisualCharacterPayload]
    chapter_starts: list[ChapterStartPayload]
    plot: str


class MangaUtterance(MangaModel):
    page_index: int = Field(ge=0)
    speaker: int | None
    content: str = Field(min_length=1)


class LocalCharacter(MangaModel):
    index: int = Field(ge=0)
    names: list[str]
    description: str


class ChapterStart(MangaModel):
    page_index: int = Field(ge=0)
    title: str | None


class VisualExtractionResult(MangaModel):
    utterances: list[MangaUtterance]
    characters: list[LocalCharacter]
    chapter_starts: list[ChapterStart]
    plot: str


class NamedCharacterPayload(MangaModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class NamedCharacterCatalogPayload(MangaModel):
    characters: list[NamedCharacterPayload]


class NamedCharacter(MangaModel):
    name: str = Field(min_length=1)
    aliases: list[str]
    description: str


class NamedCharacterCatalog(MangaModel):
    characters: list[NamedCharacter]


class CharacterObservation(MangaModel):
    batch_index: int = Field(ge=0)
    local_character_index: int = Field(ge=0)
    names: list[str]
    description: str


class CharacterAssignmentValue(MangaModel):
    batch_index: int | str
    local_character_index: int | str
    name: str | None = None


class CharacterAssignmentPayload(MangaModel):
    assignments: list[CharacterAssignmentValue]


class CharacterAssignment(MangaModel):
    batch_index: int = Field(ge=0)
    local_character_index: int = Field(ge=0)
    name: str | None


class CharacterAssignmentsResult(MangaModel):
    assignments: list[CharacterAssignment]


class ChapterDialogueLine(MangaModel):
    page_index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    content: str = Field(min_length=1)


class MangaChapter(MangaModel):
    index: int = Field(ge=0)
    title: str | None
    page_indexes: list[int] = Field(min_length=1)
    plot: str
    utterances: list[ChapterDialogueLine]


class MangaChaptersDocument(MangaModel):
    chapters: list[MangaChapter]


class RevisedUtterance(MangaModel):
    speaker: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @field_validator("speaker", "content")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class DialogueRevisionResult(MangaModel):
    utterances: list[RevisedUtterance]


class MangaCharacterProfileResult(MangaModel):
    profile: str = Field(min_length=1)

    @field_validator("profile")
    @classmethod
    def strip_profile(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("profile must not be empty")
        return stripped


class OcrBlock(MangaModel):
    text: str
    bbox: list[float] | None = None


class OcrPageResult(MangaModel):
    page_index: int = Field(ge=0)
    blocks: list[OcrBlock]
    error: str | None = None


class OcrSentence(MangaModel):
    page_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    bbox: list[float] | None = None
