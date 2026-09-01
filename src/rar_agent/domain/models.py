"""Persisted domain contracts for RAR's text workflow."""

from __future__ import annotations

from pathlib import PurePath, PureWindowsPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonObject = dict[str, Any]


def _workspace_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        raise ValueError("path must not be empty")
    if PurePath(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise ValueError("path must be workspace-relative")
    if ".." in PurePath(normalized).parts:
        raise ValueError("path must not escape the workspace")
    return normalized


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InputResource(DomainModel):
    path: str
    resource_type: Annotated[str, Field(min_length=1)]
    display_name: Annotated[str, Field(min_length=1)]
    narrative_order: Annotated[int, Field(ge=0)]
    meta: JsonObject = Field(default_factory=dict)

    _validate_path = field_validator("path")(_workspace_relative_path)


class InputManifest(DomainModel):
    name: Annotated[str, Field(min_length=1)]
    meta: JsonObject = Field(default_factory=dict)
    resources: Annotated[list[InputResource], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_resource_order(self) -> InputManifest:
        orders = [resource.narrative_order for resource in self.resources]
        if len(set(orders)) != len(orders):
            raise ValueError("resource narrative_order values must be unique")
        return self


class TextChunk(DomainModel):
    file: str
    index: Annotated[int, Field(ge=0)]
    text: str
    token_count: Annotated[int, Field(ge=0)]
    meta: JsonObject = Field(default_factory=dict)

    _validate_file = field_validator("file")(_workspace_relative_path)


class PlotRef(DomainModel):
    path: str
    index: Annotated[int, Field(ge=0)]

    _validate_path = field_validator("path")(_workspace_relative_path)


class CharacterRef(DomainModel):
    path: str
    index: Annotated[int, Field(ge=0)]

    _validate_path = field_validator("path")(_workspace_relative_path)


class CharacterCandidate(DomainModel):
    names: Annotated[list[str], Field(min_length=1)]
    description: Annotated[str, Field(min_length=1)]
    plot_indexes: list[Annotated[int, Field(ge=0)]] | None = None

    @field_validator("names")
    @classmethod
    def validate_names(cls, names: list[str]) -> list[str]:
        if any(not name.strip() for name in names):
            raise ValueError("candidate names must be non-empty")
        return names


class CharacterFilterRequestCandidate(DomainModel):
    candidate_indexes: Annotated[list[int], Field(min_length=1)]
    names: Annotated[list[str], Field(min_length=1)]
    description: str


class CharacterFilterRequest(DomainModel):
    characters: list[CharacterFilterRequestCandidate]


class CharacterGroup(DomainModel):
    name: Annotated[str, Field(min_length=1)]
    aliases: list[str]
    candidate_indexes: Annotated[list[int], Field(min_length=1)]


class CharacterFilterResult(DomainModel):
    characters: list[CharacterGroup]


class PlotExtractionResult(DomainModel):
    characters: list[CharacterCandidate]
    plots: list[tuple[str | None, str | None]]
    state: Literal["truncated", "finished"]

    @model_validator(mode="after")
    def validate_result(self) -> PlotExtractionResult:
        for boundary in self.plots:
            if any(value == "" for value in boundary):
                raise ValueError("Plot boundaries must be non-empty or null")
        for candidate in self.characters:
            if candidate.plot_indexes is None:
                continue
            if not self.plots:
                raise ValueError("plot_indexes must be null when plots is empty")
            if any(index >= len(self.plots) for index in candidate.plot_indexes):
                raise ValueError("candidate references an unknown local Plot")
        return self


class PlotChunkRef(DomainModel):
    path: str
    plot_index: Annotated[int, Field(ge=0)]
    chunk_index: Annotated[int, Field(ge=0)]

    _validate_path = field_validator("path")(_workspace_relative_path)


class TextChunkSpanRef(DomainModel):
    path: str
    index: Annotated[int, Field(ge=0)]
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]

    _validate_path = field_validator("path")(_workspace_relative_path)

    @model_validator(mode="after")
    def validate_span(self) -> TextChunkSpanRef:
        if self.end <= self.start:
            raise ValueError("TextChunk span end must be greater than start")
        return self


SourceRef = PlotChunkRef | TextChunkSpanRef


class PlotChunk(DomainModel):
    index: Annotated[int, Field(ge=0)]
    text: str
    token_count: Annotated[int, Field(ge=0)]
    source_refs: list[TextChunkSpanRef]


class Plot(DomainModel):
    index: Annotated[int, Field(ge=0)]
    meta: JsonObject = Field(default_factory=dict)
    character_refs: list[CharacterRef] = Field(default_factory=list)
    chunks: list[PlotChunk]


class PlotsDocument(DomainModel):
    plots: list[Plot]


class Utterance(DomainModel):
    index: Annotated[int, Field(ge=0)]
    speaker: Annotated[str, Field(min_length=1)]
    content: Annotated[str, Field(min_length=1)]
    source_refs: list[SourceRef]

    @model_validator(mode="after")
    def validate_utterance(self) -> Utterance:
        if self.speaker == "Environment":
            if not (
                self.content.startswith("*(")
                and self.content.endswith(")*")
                and self.content[2:-2].strip()
            ):
                raise ValueError("Environment content must use a non-empty *(...)* form")
            if any(isinstance(ref, TextChunkSpanRef) for ref in self.source_refs):
                raise ValueError("Environment may only reference its PlotChunk")
        if not any(isinstance(ref, PlotChunkRef) for ref in self.source_refs):
            raise ValueError("Utterance requires a PlotChunk reference")
        return self


class Conversation(DomainModel):
    plot_ref: PlotRef
    utterances: list[Utterance]


class CharacterProfile(DomainModel):
    name: Annotated[str, Field(min_length=1)]
    aliases: list[str] = Field(default_factory=list)
    profile: Annotated[str, Field(min_length=1)]
    plot_refs: list[PlotRef]

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, aliases: list[str]) -> list[str]:
        if any(not alias.strip() for alias in aliases):
            raise ValueError("aliases must be non-empty strings")
        if len(set(aliases)) != len(aliases):
            raise ValueError("aliases must not contain duplicates")
        return aliases


class DatasetResource(DomainModel):
    path: str
    media_type: Annotated[str, Field(min_length=1)]
    meta: JsonObject = Field(default_factory=dict)

    _validate_path = field_validator("path")(_workspace_relative_path)


class DatasetBundle(DomainModel):
    schema_version: Annotated[int, Field(ge=1)] = 1
    name: Annotated[str, Field(min_length=1)]
    meta: JsonObject = Field(default_factory=dict)
    resources: list[DatasetResource]
    characters: list[CharacterProfile]
    conversations: list[Conversation]
