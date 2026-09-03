"""The fixed, file-checkpointed V1 text Dataset workflow."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from rar_agent.domain.models import (
    CharacterProfile,
    CharacterProfileGenerationResult,
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
from rar_agent.models.scheduler import DEFAULT_WORKFLOW_CONCURRENCY, ModelScheduler
from rar_agent.models.structured import StructuredModelGateway, StructuredOutputError
from rar_agent.prompts.loader import PromptCatalog, PromptTemplate
from rar_agent.storage.artifacts import DatasetArtifactStore, DatasetPaths
from rar_agent.text.characters import CharacterResolver, ResolvedCharacter
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
StagePhase = Literal["started", "completed"]
StageCallback = Callable[[str, StagePhase, Path], Awaitable[None]]
JsonObject = dict[str, Any]
CHARACTER_PROFILES_PATH = "work/character_profiles.jsonl"
DEBUG_STAGE_UNIT_LIMIT = 5
_UNSAFE_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
PLOT_OPENING_PLUGIN = (
    "5. 此次对话提取中第一个对话内容必须是旁白用以开场，要求简短不能过长。"  # noqa: RUF001
)


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    model: str
    debug: bool = False
    text_chunk_tokens: int = 5120
    plot_chunk_tokens: int = 5120
    character_description_budget: int = 24_000
    adjacent_plot_sentences: int = 3
    max_attempts: int = 3
    max_concurrency: int = DEFAULT_WORKFLOW_CONCURRENCY
    volume_splitters: tuple[str, ...] | None = None
    chapter_splitters: tuple[str, ...] | None = None
    include_headings: bool = False
    include_front_matter: bool = False
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
            default_limit=config.max_concurrency,
            default_workflow_limit=config.max_concurrency,
        )
        self.scheduler_workload_id = str(uuid4())

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

        await self._stage_event("chunk", "started", store.paths.text_chunks)
        chunks = self._load_or_create_chunks(store, manifest)
        await self._stage_complete("chunk", store.paths.text_chunks)
        chunks = [
            TextChunk.model_validate(row)
            for row in store.read_jsonl(store.paths.text_chunks)
        ]

        workflow_chunks = (
            chunks[:DEBUG_STAGE_UNIT_LIMIT] if self.config.debug else chunks
        )
        await self._stage_event(
            "plot_extraction", "started", store.paths.plot_extractions
        )
        plot_results = await self._plot_extraction(store, workflow_chunks)
        await self._stage_complete("plot_extraction", store.paths.plot_extractions)
        plot_results = self._read_stage_results(
            store, store.paths.plot_extractions, PlotExtractionResult
        )

        candidates = [candidate for result in plot_results for candidate in result.characters]
        await self._stage_event(
            "character_profile", "started", store.paths.character_profiles
        )
        characters = await self._character_profiles(store, candidates)
        await self._stage_complete("character_profile", store.paths.character_profiles)
        characters = self._read_character_profiles(store, candidates)

        candidate_refs = {
            candidate_index: CharacterRef(path=CHARACTER_PROFILES_PATH, index=group_index)
            for group_index, character in enumerate(characters)
            for candidate_index in character.candidate_indexes
        }
        await self._stage_event("plot_reconstruction", "started", store.paths.plots)
        plots = PlotRebuilder(
            self.tokenizer, target_tokens=self.config.plot_chunk_tokens
        ).rebuild(workflow_chunks, plot_results, candidate_refs)
        store.write_json(store.paths.plots, plots.model_dump(mode="json"))
        await self._stage_complete("plot_reconstruction", store.paths.plots)
        plots = PlotsDocument.model_validate(store.read_json(store.paths.plots))

        await self._stage_event(
            "dialogue_extraction", "started", store.paths.dialogue_extractions
        )
        conversations = await self._dialogue_extraction(
            store, plots, characters
        )
        await self._stage_complete(
            "dialogue_extraction", store.paths.dialogue_extractions
        )
        conversations = self._read_conversations(store, plots, characters)

        await self._stage_event("dataset", "started", store.paths.dataset)
        profiles = self._assemble_profiles(plots, characters)

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
                    volume_splitters=self.config.volume_splitters,
                    chapter_splitters=self.config.chapter_splitters,
                    include_heading=self.config.include_headings,
                    include_front_matter=self.config.include_front_matter,
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

    async def _character_profiles(
        self,
        store: DatasetArtifactStore,
        candidates: list[Any],
    ) -> list[ResolvedCharacter]:
        resolver = CharacterResolver()
        jobs = resolver.build_jobs(
            candidates,
            description_budget=self.config.character_description_budget,
        )
        inputs = [
            {
                "path": "work/plot_extractions.jsonl",
                "candidate_indexes": list(job.candidate_indexes),
            }
            for job in jobs
        ]

        def validator(
            index: int,
        ) -> Callable[
            [CharacterProfileGenerationResult], CharacterProfileGenerationResult
        ]:
            def validate(
                value: CharacterProfileGenerationResult,
            ) -> CharacterProfileGenerationResult:
                resolver.resolve([jobs[index]], [value])
                return value

            return validate

        generated = await self._run_jsonl_stage(
            store,
            stage="character_profile",
            path=store.paths.character_profiles,
            inputs=inputs,
            payloads=[job.request.model_dump(mode="json") for job in jobs],
            schema=CharacterProfileGenerationResult,
            validator_factory=validator,
        )
        return resolver.resolve(jobs, generated)

    def _read_character_profiles(
        self,
        store: DatasetArtifactStore,
        candidates: list[Any],
    ) -> list[ResolvedCharacter]:
        resolver = CharacterResolver()
        jobs = resolver.build_jobs(
            candidates,
            description_budget=self.config.character_description_budget,
        )
        generated = self._read_stage_results(
            store,
            store.paths.character_profiles,
            CharacterProfileGenerationResult,
        )
        return resolver.resolve(jobs, generated)

    async def _dialogue_extraction(
        self,
        store: DatasetArtifactStore,
        plots: PlotsDocument,
        characters: list[ResolvedCharacter],
    ) -> list[Any]:
        units = [(plot, chunk) for plot in plots.plots for chunk in plot.chunks]
        if self.config.debug:
            units = units[:DEBUG_STAGE_UNIT_LIMIT]
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
                "input": chunk.text,
                "characters": self._plot_characters(plot.character_refs, characters),
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
            prompt_replacements=[
                {
                    "plugin_a": PLOT_OPENING_PLUGIN
                    if chunk.index == 0
                    else ""
                }
                for _, chunk in units
            ],
        )
        return self._assemble_conversations(units, results, plots, characters)

    def _read_conversations(
        self,
        store: DatasetArtifactStore,
        plots: PlotsDocument,
        characters: list[ResolvedCharacter],
    ) -> list[Any]:
        units = [(plot, chunk) for plot in plots.plots for chunk in plot.chunks]
        if self.config.debug:
            units = units[:DEBUG_STAGE_UNIT_LIMIT]
        results = self._read_stage_results(
            store,
            store.paths.dialogue_extractions,
            DialogueExtractionResult,
        )
        return self._assemble_conversations(units, results, plots, characters)

    @staticmethod
    def _assemble_conversations(
        units: list[Any],
        results: list[DialogueExtractionResult],
        plots: PlotsDocument,
        characters: list[ResolvedCharacter],
    ) -> list[Any]:
        aliases = {
            alias: group.name
            for group in characters
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
        processed_plots = {plot.index for plot, _ in units}
        return [
            assembler.assemble(plot, batches_by_plot.get(plot.index, []))
            for plot in plots.plots
            if plot.index in processed_plots
        ]

    @staticmethod
    def _read_stage_results(
        store: DatasetArtifactStore,
        path: Path,
        schema: type[SchemaT],
    ) -> list[SchemaT]:
        results: list[SchemaT] = []
        for index, record in enumerate(store.read_jsonl(path)):
            result = record.get("result")
            if result == {}:
                raise ValueError(f"{path.name} result is empty at index {index}")
            results.append(schema.model_validate(result))
        return results

    @staticmethod
    def _assemble_profiles(
        plots: PlotsDocument,
        characters: list[ResolvedCharacter],
    ) -> list[CharacterProfile]:
        return [
            CharacterProfile(
                name=character.name,
                aliases=character.aliases,
                profile=character.profile,
                plot_refs=[
                    PlotRef(path="work/plots.json", index=plot.index)
                    for plot in plots.plots
                    if CharacterRef(
                        path=CHARACTER_PROFILES_PATH,
                        index=character_index,
                    )
                    in plot.character_refs
                ],
            )
            for character_index, character in enumerate(characters)
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
        prompt_replacements: list[dict[str, str]] | None = None,
    ) -> list[SchemaT]:
        if len(inputs) != len(payloads):
            raise ValueError("stage inputs and payloads must have equal length")
        if prompt_replacements is not None and len(prompt_replacements) != len(inputs):
            raise ValueError("prompt replacements and inputs must have equal length")
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
                        prompt,
                        payloads[index],
                        schema,
                        validator=(
                            validator_factory(index)
                            if validator_factory is not None
                            else None
                        ),
                        replacements=(
                            prompt_replacements[index]
                            if prompt_replacements is not None
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
        prompt: PromptTemplate,
        payload: JsonObject,
        schema: type[SchemaT],
        validator: Callable[[SchemaT], SchemaT] | None = None,
        replacements: dict[str, str] | None = None,
    ) -> SchemaT:
        render_values: dict[str, str] = {}
        if stage == "plot_extraction":
            render_values["k"] = str(self.config.adjacent_plot_sentences)
        elif stage == "dialogue_extraction":
            render_values["plugin_a"] = ""
        render_values.update(replacements or {})
        rendered_prompt = prompt.render(render_values)
        serialized_payload = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        user_content = (
            f"{rendered_prompt.user_prefix}\n{serialized_payload}"
            if rendered_prompt.user_prefix
            else serialized_payload
        )
        request = ModelRequest(
            model=self.config.stage_models.get(stage, self.config.model),
            messages=[
                ModelMessage(role="system", content=rendered_prompt.system),
                ModelMessage(role="user", content=user_content),
            ],
            temperature=0,
        )
        return await self.gateway.generate(
            self.model_client,
            request,
            schema,
            validator=validator,
            scheduler=self.scheduler,
            workload_id=self.scheduler_workload_id,
            workload_limit=self.config.max_concurrency,
        )

    def _plot_payload(self, chunks: list[TextChunk], index: int) -> JsonObject:
        current = chunks[index]
        previous = ""
        following = ""
        count = self.config.adjacent_plot_sentences
        if count and index > 0 and chunks[index - 1].meta == current.meta:
            previous = "".join(split_sentences(chunks[index - 1].text)[-count:])
        if count and index + 1 < len(chunks) and chunks[index + 1].meta == current.meta:
            following = "".join(split_sentences(chunks[index + 1].text)[:count])
        return {"input": current.text, "previous": previous, "next": following}

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
        characters: list[ResolvedCharacter],
    ) -> list[JsonObject]:
        values: list[JsonObject] = []
        for ref in refs:
            character = characters[ref.index]
            values.append(
                {
                    "names": list(character.names),
                    "description": character.description,
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
        await self._stage_event(stage, "completed", artifact)

    async def _stage_event(
        self, stage: str, phase: StagePhase, artifact: Path
    ) -> None:
        if self.stage_callback is not None:
            await self.stage_callback(stage, phase, artifact)

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
