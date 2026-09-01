"""The fixed, file-checkpointed V1 text Dataset workflow."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from rar_agent.domain.models import (
    CharacterFilterResult,
    CharacterProfile,
    CharacterRef,
    DatasetBundle,
    DatasetResource,
    InputManifest,
    PlotExtractionResult,
    PlotRef,
    PlotsDocument,
    TextChunk,
)
from rar_agent.export.sharegpt import ShareGPTExporter, ShareGPTExportReport
from rar_agent.models.base import ModelClient, ModelMessage, ModelRequest
from rar_agent.models.scheduler import ModelScheduler
from rar_agent.models.structured import StructuredModelGateway, StructuredOutputError
from rar_agent.prompts.loader import PromptCatalog
from rar_agent.storage.artifacts import DatasetArtifactStore, DatasetPaths
from rar_agent.text.characters import CharacterResolver
from rar_agent.text.chunking import TextChunker, split_sentences
from rar_agent.text.dialogue import (
    ConversationAssembler,
    DialogueAligner,
    DialogueBatch,
    DialogueExtractionResult,
)
from rar_agent.text.plot_rebuilder import PlotRebuilder
from rar_agent.text.tokenizer import Tokenizer

SchemaT = TypeVar("SchemaT", bound=BaseModel)
StageCallback = Callable[[str, Path], Awaitable[None]]
JsonObject = dict[str, Any]
CHARACTER_FILTER_PATH = "work/character_filter.json"
_UNSAFE_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


class ProfileGenerationResult(BaseModel):
    profile: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    model: str
    text_chunk_tokens: int = 5120
    plot_chunk_tokens: int = 5120
    character_description_budget: int = 24_000
    adjacent_plot_sentences: int = 3
    max_attempts: int = 3
    max_concurrency: int = 4
    profile_description_chars: int = 16_000
    stage_models: dict[str, str] = field(default_factory=dict)
    prompt_overrides: dict[str, Path] = field(default_factory=dict)
    sharegpt_system_template: str = (
        "你将扮演{character}。\n角色档案:\n{profile}\n当前剧情:\n{plot}"
    )

    def __post_init__(self) -> None:
        for name in (
            "text_chunk_tokens",
            "plot_chunk_tokens",
            "max_attempts",
            "max_concurrency",
            "profile_description_chars",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.character_description_budget < 0:
            raise ValueError("character_description_budget must not be negative")
        if self.adjacent_plot_sentences < 0:
            raise ValueError("adjacent_plot_sentences must not be negative")


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    dataset_root: Path
    paths: DatasetPaths
    bundle: DatasetBundle
    plots: PlotsDocument
    export_report: ShareGPTExportReport


class IncompleteStageError(RuntimeError):
    def __init__(self, stage: str, dataset_root: Path, failed_indexes: list[int]) -> None:
        self.stage = stage
        self.dataset_root = dataset_root
        self.failed_indexes = failed_indexes
        super().__init__(f"{stage} is incomplete; failed units: {failed_indexes}")


class DatasetBuildWorkflow:
    """Stable extraction seam; model choice never changes its stage graph."""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        tokenizer: Tokenizer,
        config: WorkflowConfig,
        stage_callback: StageCallback | None = None,
        scheduler: ModelScheduler | None = None,
    ) -> None:
        self.model_client = model_client
        self.tokenizer = tokenizer
        self.config = config
        self.stage_callback = stage_callback
        self.gateway = StructuredModelGateway(max_attempts=config.max_attempts)
        self.prompts = PromptCatalog(config.prompt_overrides)
        self.scheduler = scheduler or ModelScheduler(
            default_limit=config.max_concurrency
        )

    async def run(
        self,
        project_root: Path,
        manifest: InputManifest,
        *,
        dataset_root: Path | None = None,
    ) -> DatasetBuildResult:
        project_root = project_root.resolve()
        if dataset_root is None:
            store = DatasetArtifactStore.create(project_root, manifest.name)
            store.write_json(store.paths.input_manifest, manifest.model_dump(mode="json"))
        else:
            store = DatasetArtifactStore.open(project_root, dataset_root)
            if store.paths.input_manifest.exists():
                manifest = InputManifest.model_validate(
                    store.read_json(store.paths.input_manifest)
                )
            else:
                store.write_json(store.paths.input_manifest, manifest.model_dump(mode="json"))

        chunks = self._load_or_create_chunks(store, manifest)
        await self._stage_complete("chunk", store.paths.text_chunks)

        plot_results = await self._plot_extraction(store, chunks)
        await self._stage_complete("plot_extraction", store.paths.plot_extractions)

        candidates = [candidate for result in plot_results for candidate in result.characters]
        character_filter = await self._character_filter(store, candidates)
        await self._stage_complete("character_filter", store.paths.character_filter)

        candidate_refs = {
            candidate_index: CharacterRef(path=CHARACTER_FILTER_PATH, index=group_index)
            for group_index, group in enumerate(character_filter.characters)
            for candidate_index in group.candidate_indexes
        }
        plots = PlotRebuilder(
            self.tokenizer, target_tokens=self.config.plot_chunk_tokens
        ).rebuild(chunks, plot_results, candidate_refs)
        store.write_json(store.paths.plots, plots.model_dump(mode="json"))
        await self._stage_complete("plot_reconstruction", store.paths.plots)

        conversations = await self._dialogue_extraction(
            store, plots, character_filter, candidates
        )
        await self._stage_complete(
            "dialogue_extraction", store.paths.dialogue_extractions
        )

        profiles = await self._profile_generation(
            store, plots, character_filter, candidates
        )
        await self._stage_complete(
            "character_profile", store.paths.character_profiles
        )

        bundle = DatasetBundle(
            name=manifest.name,
            meta=manifest.meta,
            resources=[
                DatasetResource(
                    path=resource.path,
                    media_type=resource.resource_type,
                    meta={"display_name": resource.display_name, **resource.meta},
                )
                for resource in sorted(
                    manifest.resources, key=lambda item: item.narrative_order
                )
            ],
            characters=profiles,
            conversations=conversations,
        )
        store.write_json(store.paths.dataset, bundle.model_dump(mode="json"))
        self._write_character_files(store, bundle.characters)
        export_report = ShareGPTExporter().export(
            bundle,
            plots,
            store.paths.exports / "sharegpt",
            system_template=self.config.sharegpt_system_template,
        )
        await self._stage_complete("dataset", store.paths.dataset)
        return DatasetBuildResult(store.root, store.paths, bundle, plots, export_report)

    def _load_or_create_chunks(
        self, store: DatasetArtifactStore, manifest: InputManifest
    ) -> list[TextChunk]:
        if store.paths.text_chunks.exists():
            return [
                TextChunk.model_validate(row)
                for row in store.read_jsonl(store.paths.text_chunks)
            ]
        chunker = TextChunker(self.tokenizer)
        chunks: list[TextChunk] = []
        for resource in sorted(manifest.resources, key=lambda item: item.narrative_order):
            if resource.resource_type != "text":
                raise ValueError(
                    f"V1 text workflow cannot process {resource.resource_type!r}"
                )
            source = self._workspace_file(store.project_root, resource.path)
            text = source.read_text(encoding="utf-8-sig")
            chunks.extend(
                chunker.chunk_text(
                    resource.path,
                    text,
                    target_tokens=self.config.text_chunk_tokens,
                    meta=resource.meta,
                )
            )
        store.write_jsonl(
            store.paths.text_chunks,
            [chunk.model_dump(mode="json") for chunk in chunks],
        )
        return chunks

    async def _plot_extraction(
        self, store: DatasetArtifactStore, chunks: list[TextChunk]
    ) -> list[PlotExtractionResult]:
        inputs = [
            {"path": "work/text_chunks.jsonl", "index": index}
            for index in range(len(chunks))
        ]
        payloads = [self._plot_payload(chunks, index) for index in range(len(chunks))]

        def validator(index: int) -> Callable[[PlotExtractionResult], PlotExtractionResult]:
            return lambda value: self._validate_plot_result(chunks, index, value)

        return await self._run_jsonl_stage(
            store,
            stage="plot_extraction",
            path=store.paths.plot_extractions,
            inputs=inputs,
            payloads=payloads,
            schema=PlotExtractionResult,
            validator_factory=validator,
        )

    async def _character_filter(
        self,
        store: DatasetArtifactStore,
        candidates: list[Any],
    ) -> CharacterFilterResult:
        resolver = CharacterResolver()
        request_value = resolver.build_request(
            candidates,
            description_budget=self.config.character_description_budget,
        )
        input_locator = {"path": "work/plot_extractions.jsonl"}
        if store.paths.character_filter.exists():
            record = store.read_json(store.paths.character_filter)
            if record.get("input") != input_locator:
                raise ValueError("character_filter.json input mismatch")
            if record.get("result") != {}:
                return resolver.resolve(
                    candidates,
                    CharacterFilterResult.model_validate(record["result"]),
                )

        prompt = self.prompts.load("character_filter")
        record = self._record(
            input_locator,
            prompt.filename,
            {},
            model=self.config.stage_models.get("character_filter", self.config.model),
        )
        store.write_json(store.paths.character_filter, record)
        try:
            value = await self._generate(
                "character_filter",
                prompt.text,
                request_value.model_dump(mode="json"),
                CharacterFilterResult,
                validator=lambda result: resolver.resolve(candidates, result),
            )
            record["result"] = value.model_dump(mode="json")
        except StructuredOutputError:
            record["result"] = {}
        store.write_json(store.paths.character_filter, record)
        if record["result"] == {}:
            raise IncompleteStageError("character_filter", store.root, [0])
        return CharacterFilterResult.model_validate(record["result"])

    async def _dialogue_extraction(
        self,
        store: DatasetArtifactStore,
        plots: PlotsDocument,
        character_filter: CharacterFilterResult,
        candidates: list[Any],
    ) -> list[Any]:
        units = [(plot, chunk) for plot in plots.plots for chunk in plot.chunks]
        inputs = [
            {
                "path": "work/plots.json",
                "plot_index": plot.index,
                "chunk_index": chunk.index,
            }
            for plot, chunk in units
        ]
        payloads = [
            {
                "plot_chunk": chunk.text,
                "characters": self._plot_characters(
                    plot.character_refs, character_filter, candidates
                ),
            }
            for plot, chunk in units
        ]
        results = await self._run_jsonl_stage(
            store,
            stage="dialogue_extraction",
            path=store.paths.dialogue_extractions,
            inputs=inputs,
            payloads=payloads,
            schema=DialogueExtractionResult,
        )
        aliases = {
            alias: group.name
            for group in character_filter.characters
            for alias in [group.name, *group.aliases]
        }
        aligner = DialogueAligner()
        batches_by_plot: dict[int, list[DialogueBatch]] = {}
        for (plot, chunk), result in zip(units, results, strict=True):
            batch = aligner.align(
                plot.index,
                chunk,
                result.utterances,
                aliases=aliases,
            )
            batches_by_plot.setdefault(plot.index, []).append(batch)
        assembler = ConversationAssembler()
        return [
            assembler.assemble(plot, batches_by_plot.get(plot.index, []))
            for plot in plots.plots
        ]

    async def _profile_generation(
        self,
        store: DatasetArtifactStore,
        plots: PlotsDocument,
        character_filter: CharacterFilterResult,
        candidates: list[Any],
    ) -> list[CharacterProfile]:
        inputs = [
            {"path": CHARACTER_FILTER_PATH, "index": index}
            for index in range(len(character_filter.characters))
        ]
        descriptions = [
            list(
                dict.fromkeys(
                    candidates[index].description for index in group.candidate_indexes
                )
            )
            for group in character_filter.characters
        ]
        payloads = [
            {
                "name": group.name,
                "aliases": group.aliases,
                "descriptions": value,
            }
            for group, value in zip(
                character_filter.characters, descriptions, strict=True
            )
        ]
        generated = await self._run_jsonl_stage(
            store,
            stage="character_profile",
            path=store.paths.character_profiles,
            inputs=inputs,
            payloads=payloads,
            schema=ProfileGenerationResult,
            custom_generator=lambda index: self._generate_profile(
                character_filter.characters[index].name,
                character_filter.characters[index].aliases,
                descriptions[index],
            ),
        )
        return [
            CharacterProfile(
                name=group.name,
                aliases=group.aliases,
                profile=result.profile,
                plot_refs=[
                    PlotRef(path="work/plots.json", index=plot.index)
                    for plot in plots.plots
                    if CharacterRef(path=CHARACTER_FILTER_PATH, index=group_index)
                    in plot.character_refs
                ],
            )
            for group_index, (group, result) in enumerate(
                zip(character_filter.characters, generated, strict=True)
            )
        ]

    async def _run_jsonl_stage(
        self,
        store: DatasetArtifactStore,
        *,
        stage: str,
        path: Path,
        inputs: list[JsonObject],
        payloads: list[JsonObject],
        schema: type[SchemaT],
        validator_factory: Callable[
            [int], Callable[[SchemaT], SchemaT]
        ]
        | None = None,
        custom_generator: Callable[[int], Awaitable[SchemaT]] | None = None,
    ) -> list[SchemaT]:
        if len(inputs) != len(payloads):
            raise ValueError("stage inputs and payloads must have equal length")
        prompt = self.prompts.load(stage)
        existing = store.read_jsonl(path)
        if len(existing) > len(inputs):
            raise ValueError(f"{path.name} contains too many records")
        records: list[JsonObject] = []
        pending: list[int] = []
        for index, input_value in enumerate(inputs):
            if index < len(existing):
                record = existing[index]
                if record.get("input") != input_value:
                    raise ValueError(f"{path.name} input mismatch at index {index}")
                result = record.get("result")
                if result != {}:
                    try:
                        parsed = schema.model_validate(result)
                        if validator_factory is not None:
                            validator_factory(index)(parsed)
                    except (ValueError, TypeError):
                        record["result"] = {}
                if record.get("result") == {}:
                    pending.append(index)
                records.append(record)
            else:
                records.append(
                    self._record(
                        input_value,
                        prompt.filename,
                        {},
                        model=self.config.stage_models.get(stage, self.config.model),
                    )
                )
                pending.append(index)
        store.write_jsonl(path, records)
        lock = asyncio.Lock()

        async def execute(index: int) -> None:
            try:
                if custom_generator is not None:
                    value = await custom_generator(index)
                else:
                    value = await self._generate(
                        stage,
                        prompt.text,
                        payloads[index],
                        schema,
                        validator=(
                            validator_factory(index)
                            if validator_factory is not None
                            else None
                        ),
                    )
                records[index]["result"] = value.model_dump(mode="json")
            except StructuredOutputError:
                records[index]["result"] = {}
            async with lock:
                store.write_jsonl(path, records)

        await asyncio.gather(*(execute(index) for index in pending))
        failed = [
            index for index, record in enumerate(records) if record.get("result") == {}
        ]
        if failed:
            raise IncompleteStageError(stage, store.root, failed)
        return [schema.model_validate(record["result"]) for record in records]

    async def _generate(
        self,
        stage: str,
        system_prompt: str,
        payload: JsonObject,
        schema: type[SchemaT],
        validator: Callable[[SchemaT], SchemaT] | None = None,
    ) -> SchemaT:
        request = ModelRequest(
            model=self.config.stage_models.get(stage, self.config.model),
            messages=[
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            ],
            temperature=0,
        )
        return await self.gateway.generate(
            self.model_client,
            request,
            schema,
            validator=validator,
            scheduler=self.scheduler,
        )

    async def _generate_profile(
        self,
        name: str,
        aliases: list[str],
        descriptions: list[str],
    ) -> ProfileGenerationResult:
        prompt = self.prompts.load("character_profile")
        batches = self._description_batches(
            descriptions, self.config.profile_description_chars
        )
        if len(batches) == 1:
            return await self._generate(
                "character_profile",
                prompt.text,
                {"name": name, "aliases": aliases, "descriptions": batches[0]},
                ProfileGenerationResult,
            )
        partials = await asyncio.gather(
            *(
                self._generate(
                    "character_profile",
                    prompt.text,
                    {
                        "mode": "partial",
                        "name": name,
                        "aliases": aliases,
                        "descriptions": batch,
                    },
                    ProfileGenerationResult,
                )
                for batch in batches
            )
        )
        return await self._generate(
            "character_profile",
            prompt.text,
            {
                "mode": "final",
                "name": name,
                "aliases": aliases,
                "partial_profiles": [value.profile for value in partials],
            },
            ProfileGenerationResult,
        )

    @staticmethod
    def _description_batches(
        descriptions: list[str], limit: int
    ) -> list[list[str]]:
        pieces = [
            description[start : start + limit]
            for description in descriptions
            for start in range(0, len(description), limit)
        ]
        if not pieces:
            return [[]]
        batches: list[list[str]] = []
        current: list[str] = []
        size = 0
        for piece in pieces:
            if current and size + len(piece) > limit:
                batches.append(current)
                current = []
                size = 0
            current.append(piece)
            size += len(piece)
        if current:
            batches.append(current)
        return batches

    def _plot_payload(self, chunks: list[TextChunk], index: int) -> JsonObject:
        current = chunks[index]
        previous = ""
        following = ""
        count = self.config.adjacent_plot_sentences
        if count and index > 0 and chunks[index - 1].meta == current.meta:
            previous = "".join(split_sentences(chunks[index - 1].text)[-count:])
        if count and index + 1 < len(chunks) and chunks[index + 1].meta == current.meta:
            following = "".join(split_sentences(chunks[index + 1].text)[:count])
        return {"previous": previous, "current": current.text, "next": following}

    @staticmethod
    def _validate_plot_result(
        chunks: list[TextChunk], index: int, result: PlotExtractionResult
    ) -> PlotExtractionResult:
        chunk = chunks[index]
        for start, end in result.plots:
            if start is not None and start not in chunk.text:
                raise ValueError("Plot start boundary is not in current TextChunk")
            if end is not None and end not in chunk.text:
                raise ValueError("Plot end boundary is not in current TextChunk")
        section_end = index == len(chunks) - 1 or chunks[index + 1].meta != chunk.meta
        if (
            section_end
            and result.state == "truncated"
            and result.plots
            and result.plots[-1][1] is None
        ):
            sentences = split_sentences(chunk.text)
            if not sentences:
                raise ValueError("cannot close a Plot in an empty TextChunk")
            plots = list(result.plots)
            plots[-1] = (plots[-1][0], sentences[-1])
            return result.model_copy(update={"plots": plots, "state": "finished"})
        return result

    @staticmethod
    def _plot_characters(
        refs: list[CharacterRef],
        character_filter: CharacterFilterResult,
        candidates: list[Any],
    ) -> list[JsonObject]:
        values: list[JsonObject] = []
        for ref in refs:
            group = character_filter.characters[ref.index]
            values.append(
                {
                    "name": group.name,
                    "aliases": group.aliases,
                    "descriptions": [
                        candidates[index].description
                        for index in group.candidate_indexes
                    ],
                }
            )
        return values

    def _record(
        self,
        input_value: JsonObject,
        prompt_file: str,
        result: JsonObject,
        *,
        model: str,
    ) -> JsonObject:
        return {
            "input": input_value,
            "prompt_file": prompt_file,
            "provider": self.model_client.provider,
            "model": model,
            "result": result,
        }

    async def _stage_complete(self, stage: str, artifact: Path) -> None:
        if self.stage_callback is not None:
            await self.stage_callback(stage, artifact)

    @staticmethod
    def _workspace_file(project_root: Path, relative_path: str) -> Path:
        path = (project_root / relative_path).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise ValueError(f"resource escapes Project: {relative_path}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _write_character_files(
        store: DatasetArtifactStore, characters: list[CharacterProfile]
    ) -> None:
        for index, character in enumerate(characters):
            safe_name = _UNSAFE_FILE_CHARS.sub("-", character.name).strip(" .-")
            path = store.paths.characters / f"{index:04d}-{safe_name or 'character'}.txt"
            path.write_text(character.profile + "\n", encoding="utf-8", newline="\n")
