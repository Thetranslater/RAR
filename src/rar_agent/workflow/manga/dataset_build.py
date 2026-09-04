"""File-checkpointed Manga Workflow for local image folders."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from rar_agent.domain.models import (
    CharacterProfile,
    CharacterRef,
    Conversation,
    DatasetBundle,
    DatasetResource,
    InputManifest,
    Plot,
    PlotChunk,
    PlotChunkRef,
    PlotRef,
    PlotsDocument,
    Utterance,
)
from rar_agent.export.sharegpt import ShareGPTExporter, ShareGPTExportReport
from rar_agent.models.base import (
    LocalImageContent,
    ModelClient,
    ModelMessage,
    ModelRequest,
    TextContent,
)
from rar_agent.models.scheduler import DEFAULT_WORKFLOW_CONCURRENCY, ModelScheduler
from rar_agent.models.structured import StructuredModelGateway, StructuredOutputError
from rar_agent.prompts.loader import PromptCatalog, PromptTemplate
from rar_agent.storage.artifacts import DatasetArtifactStore
from rar_agent.text.tokenizer import Tokenizer
from rar_agent.workflow.dataset_build import IncompleteStageError, StageCallback
from rar_agent.workflow.manga.chapters import reconstruct_manga_chapters
from rar_agent.workflow.manga.characters import (
    aggregate_character_descriptions,
    clean_character_catalog,
    validate_character_assignments,
)
from rar_agent.workflow.manga.models import (
    CharacterAssignment,
    CharacterAssignmentPayload,
    CharacterAssignmentsResult,
    CharacterObservation,
    DialogueRevisionResult,
    ImageBatch,
    ImagePage,
    MangaChaptersDocument,
    MangaCharacterProfileResult,
    MangaScan,
    NamedCharacterCatalog,
    NamedCharacterCatalogPayload,
    OcrPageResult,
    VisualExtractionPayload,
    VisualExtractionResult,
)
from rar_agent.workflow.manga.normalization import normalize_visual_extraction
from rar_agent.workflow.manga.ocr import align_ocr_results
from rar_agent.workflow.manga.scanning import scan_manga_folder

JsonObject = dict[str, Any]
SchemaT = TypeVar("SchemaT", bound=BaseModel)
StagePhase = Literal["started", "completed"]
MANGA_PLOTS_PATH = "work/manga/plots.json"
MANGA_CHARACTER_PROFILES_PATH = "work/manga/character_profiles.jsonl"
_UNSAFE_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


class OcrRunner(Protocol):
    async def recognize(
        self,
        project_root: Path,
        pages: list[ImagePage],
        *,
        batch_size: int,
    ) -> list[OcrPageResult]: ...


@dataclass(frozen=True, slots=True)
class MangaWorkflowConfig:
    vision_model: str = "qwen3.7-flash"
    text_model: str = "deepseek-chat"
    debug: bool = False
    image_batch_size: int = 5
    hard_directory_boundaries: bool = False
    ocr_enabled: bool = False
    ocr_batch_size: int = 4
    ocr_threshold: int = 70
    visual_max_attempts: int = 2
    text_max_attempts: int = 3
    max_concurrency: int = DEFAULT_WORKFLOW_CONCURRENCY
    text_context_tokens: int = 32_768
    reserved_output_tokens: int = 4_096
    visual_temperature: float = 1.0
    visual_top_p: float = 0.9
    enable_thinking: bool = True
    reasoning_effort: str = "medium"
    max_output_tokens: int = 16_384
    visual_timeout_seconds: float = 180.0
    prompt_overrides: dict[str, Path] = field(default_factory=dict)
    sharegpt_system_template: str = (
        "你将扮演{character}。\n角色档案:\n{profile}\n当前剧情:\n{plot}"
    )

    def __post_init__(self) -> None:
        if not 1 <= self.image_batch_size <= 10:
            raise ValueError("image_batch_size must be between 1 and 10")
        if self.ocr_batch_size < 1:
            raise ValueError("ocr_batch_size must be positive")
        if not 0 <= self.ocr_threshold <= 100:
            raise ValueError("ocr_threshold must be between 0 and 100")
        for name in (
            "visual_max_attempts",
            "text_max_attempts",
            "max_concurrency",
            "text_context_tokens",
            "max_output_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must not be negative")
        if self.reserved_output_tokens >= self.text_context_tokens:
            raise ValueError("reserved_output_tokens must be below text_context_tokens")

    def persisted(self) -> JsonObject:
        value = asdict(self)
        value["prompt_overrides"] = {
            name: str(path) for name, path in self.prompt_overrides.items()
        }
        return value


@dataclass(frozen=True, slots=True)
class MangaPaths:
    root: Path
    workflow_config: Path
    input_manifest: Path
    work: Path
    image_pages: Path
    image_batches: Path
    visual_extractions: Path
    ocr_results: Path
    aligned_extractions: Path
    character_catalog: Path
    character_assignments: Path
    plots: Path
    dialogue_revisions: Path
    character_profiles: Path
    dataset: Path
    characters: Path
    exports: Path
    report: Path

    @classmethod
    def from_store(cls, store: DatasetArtifactStore) -> MangaPaths:
        work = store.root / "work" / "manga"
        return cls(
            root=store.root,
            workflow_config=store.root / "workflow_config.json",
            input_manifest=store.paths.input_manifest,
            work=work,
            image_pages=work / "image_pages.jsonl",
            image_batches=work / "image_batches.jsonl",
            visual_extractions=work / "visual_extractions.jsonl",
            ocr_results=work / "ocr_results.jsonl",
            aligned_extractions=work / "aligned_extractions.jsonl",
            character_catalog=work / "character_catalog.json",
            character_assignments=work / "character_assignments.jsonl",
            plots=work / "plots.json",
            dialogue_revisions=work / "dialogue_revisions.jsonl",
            character_profiles=work / "character_profiles.jsonl",
            dataset=store.paths.dataset,
            characters=store.paths.characters,
            exports=store.paths.exports,
            report=store.root / "reports" / "manga_extraction_summary.json",
        )


@dataclass(frozen=True, slots=True)
class MangaDatasetBuildResult:
    dataset_root: Path
    paths: MangaPaths
    bundle: DatasetBundle
    plots: PlotsDocument
    export_report: ShareGPTExportReport


class MangaDatasetBuildWorkflow:
    """Stable public seam for one local manga folder extraction."""

    def __init__(
        self,
        *,
        vision_model_client: ModelClient,
        text_model_client: ModelClient,
        tokenizer: Tokenizer,
        config: MangaWorkflowConfig,
        stage_callback: StageCallback | None = None,
        scheduler: ModelScheduler | None = None,
        ocr_runner: OcrRunner | None = None,
    ) -> None:
        self.vision_model_client = vision_model_client
        self.text_model_client = text_model_client
        self.tokenizer = tokenizer
        self.config = config
        self.stage_callback = stage_callback
        self.prompts = PromptCatalog(config.prompt_overrides)
        self.scheduler = scheduler or ModelScheduler(
            default_limit=config.max_concurrency,
            default_workflow_limit=config.max_concurrency,
        )
        self.scheduler_workload_id = str(uuid4())
        self.ocr_runner = ocr_runner

    async def run(
        self,
        project_root: Path,
        manifest: InputManifest,
        *,
        dataset_root: Path | None = None,
    ) -> MangaDatasetBuildResult:
        project = project_root.resolve()
        if dataset_root is None:
            store = DatasetArtifactStore.create(project, manifest.name)
            store.write_json(store.paths.input_manifest, manifest.model_dump(mode="json"))
        else:
            store = DatasetArtifactStore.open(project, dataset_root)
            if store.paths.input_manifest.exists():
                manifest = InputManifest.model_validate(
                    store.read_json(store.paths.input_manifest)
                )
            else:
                store.write_json(
                    store.paths.input_manifest,
                    manifest.model_dump(mode="json"),
                )
        resource = self._manga_resource(manifest)
        paths = MangaPaths.from_store(store)
        paths.work.mkdir(parents=True, exist_ok=True)
        paths.report.parent.mkdir(parents=True, exist_ok=True)
        if not paths.workflow_config.exists():
            store.write_json(paths.workflow_config, self.config.persisted())

        await self._stage_event("image_scan", "started", paths.image_pages)
        scan = self._load_or_scan(store, paths, resource.path)
        await self._stage_event("image_scan", "completed", paths.image_pages)
        scan = self._read_scan(store, paths, resource.path)

        await self._stage_event(
            "visual_extraction", "started", paths.visual_extractions
        )
        ocr_task: asyncio.Task[list[OcrPageResult]] | None = None
        if self.config.ocr_enabled:
            if self.ocr_runner is None:
                raise RuntimeError("OCR is enabled but no OCR runner is available")
            ocr_task = asyncio.create_task(
                self._ocr_pages(
                    project,
                    store,
                    paths,
                    scan,
                )
            )
        try:
            visual_results = await self._visual_extractions(store, paths, scan)
            ocr_pages = await ocr_task if ocr_task is not None else []
        except BaseException:
            if ocr_task is not None and not ocr_task.done():
                ocr_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ocr_task
            raise
        await self._stage_event(
            "visual_extraction", "completed", paths.visual_extractions
        )
        visual_results = self._read_stage_results(
            store,
            paths.visual_extractions,
            VisualExtractionResult,
        )

        await self._stage_event("ocr_alignment", "started", paths.aligned_extractions)
        if self.config.ocr_enabled:
            ocr_pages = [
                OcrPageResult.model_validate(row)
                for row in store.read_jsonl(paths.ocr_results)
            ]
            visual_results = align_ocr_results(
                scan,
                visual_results,
                ocr_pages,
                threshold=self.config.ocr_threshold,
            )
            store.write_jsonl(
                paths.aligned_extractions,
                [result.model_dump(mode="json") for result in visual_results],
            )
        await self._stage_event(
            "ocr_alignment",
            "completed",
            paths.aligned_extractions if self.config.ocr_enabled else paths.visual_extractions,
        )
        if self.config.ocr_enabled:
            visual_results = [
                VisualExtractionResult.model_validate(row)
                for row in store.read_jsonl(paths.aligned_extractions)
            ]

        observations = self._observations(visual_results)
        await self._stage_event(
            "character_catalog", "started", paths.character_catalog
        )
        catalog = await self._character_catalog(store, paths, observations)
        await self._stage_event(
            "character_catalog", "completed", paths.character_catalog
        )
        catalog = NamedCharacterCatalog.model_validate(
            store.read_json(paths.character_catalog)
        )

        await self._stage_event(
            "character_assignment", "started", paths.character_assignments
        )
        assignments = await self._character_assignments(
            store,
            paths,
            observations,
            catalog,
        )
        await self._stage_event(
            "character_assignment", "completed", paths.character_assignments
        )
        assignments = [
            assignment
            for result in self._read_stage_results(
                store,
                paths.character_assignments,
                CharacterAssignmentsResult,
            )
            for assignment in result.assignments
        ]

        await self._stage_event("chapter_reconstruction", "started", paths.plots)
        chapters = reconstruct_manga_chapters(
            scan,
            visual_results,
            assignments,
            hard_directory_boundaries=self.config.hard_directory_boundaries,
        )
        plots = self._plots(chapters)
        store.write_json(paths.plots, plots.model_dump(mode="json"))
        await self._stage_event("chapter_reconstruction", "completed", paths.plots)
        plots = PlotsDocument.model_validate(store.read_json(paths.plots))

        await self._stage_event(
            "dialogue_revision", "started", paths.dialogue_revisions
        )
        revisions = await self._dialogue_revisions(
            store,
            paths,
            chapters,
            plots,
            catalog,
        )
        await self._stage_event(
            "dialogue_revision", "completed", paths.dialogue_revisions
        )
        revisions = self._read_stage_results(
            store,
            paths.dialogue_revisions,
            DialogueRevisionResult,
        )

        await self._stage_event(
            "character_profile", "started", paths.character_profiles
        )
        await self._character_profiles(
            store,
            paths,
            observations,
            assignments,
            catalog,
            revisions,
        )
        await self._stage_event(
            "character_profile", "completed", paths.character_profiles
        )
        profiles = self._assemble_profiles(
            store,
            paths,
            catalog,
            revisions,
        )

        await self._stage_event("dataset", "started", paths.dataset)
        conversations = self._conversations(revisions)
        plots = self._attach_character_refs(plots, profiles, conversations)
        store.write_json(paths.plots, plots.model_dump(mode="json"))
        bundle = DatasetBundle(
            name=manifest.name,
            meta=manifest.meta,
            resources=[
                DatasetResource(
                    path=resource.path,
                    media_type="manga",
                    meta={
                        "display_name": resource.display_name,
                        "page_count": len(scan.pages),
                        **resource.meta,
                    },
                )
            ],
            characters=profiles,
            conversations=conversations,
        )
        store.write_json(paths.dataset, bundle.model_dump(mode="json"))
        self._write_character_files(paths, profiles)
        await self._stage_event("dataset", "completed", paths.dataset)

        await self._stage_event("export", "started", paths.exports)
        export_report = ShareGPTExporter().export(
            bundle,
            plots,
            paths.exports / "sharegpt",
            system_template=self.config.sharegpt_system_template,
        )
        await self._stage_event("export", "completed", paths.exports)
        self._write_report(
            store,
            paths,
            scan,
            visual_results,
            ocr_pages,
            catalog,
            assignments,
            revisions,
            profiles,
            export_report,
        )
        return MangaDatasetBuildResult(store.root, paths, bundle, plots, export_report)

    def _load_or_scan(
        self,
        store: DatasetArtifactStore,
        paths: MangaPaths,
        resource_path: str,
    ) -> MangaScan:
        if paths.image_pages.exists() and paths.image_batches.exists():
            return self._read_scan(store, paths, resource_path)
        scan = scan_manga_folder(
            store.project_root,
            resource_path,
            batch_size=self.config.image_batch_size,
            max_batches=5 if self.config.debug else None,
        )
        store.write_jsonl(
            paths.image_pages,
            [page.model_dump(mode="json") for page in scan.pages],
        )
        store.write_jsonl(
            paths.image_batches,
            [batch.model_dump(mode="json") for batch in scan.batches],
        )
        return scan

    @staticmethod
    def _read_scan(
        store: DatasetArtifactStore,
        paths: MangaPaths,
        resource_path: str,
    ) -> MangaScan:
        return MangaScan(
            resource_path=resource_path,
            pages=[ImagePage.model_validate(row) for row in store.read_jsonl(paths.image_pages)],
            batches=[
                ImageBatch.model_validate(row)
                for row in store.read_jsonl(paths.image_batches)
            ],
        )

    async def _visual_extractions(
        self,
        store: DatasetArtifactStore,
        paths: MangaPaths,
        scan: MangaScan,
    ) -> list[VisualExtractionResult]:
        pages = {page.page_index: page for page in scan.pages}
        inputs = [
            {"path": "work/manga/image_batches.jsonl", "index": batch.batch_index}
            for batch in scan.batches
        ]
        prompt = self.prompts.load("manga_visual_extraction")

        async def generate(index: int) -> VisualExtractionResult:
            batch = scan.batches[index]
            rendered = prompt.render()
            task = {
                "batch_index": batch.batch_index,
                "pages": [
                    {"page_index": local_index}
                    for local_index in range(len(batch.page_indexes))
                ],
            }
            content: list[TextContent | LocalImageContent] = [
                TextContent(
                    text=json.dumps(task, ensure_ascii=False, separators=(",", ":"))
                )
            ]
            for local_index, page_index in enumerate(batch.page_indexes):
                content.extend(
                    [
                        TextContent(text=f"Page {local_index}"),
                        LocalImageContent(path=pages[page_index].path),
                    ]
                )
            request = ModelRequest(
                model=self.config.vision_model,
                messages=[
                    ModelMessage(role="system", content=rendered.system),
                    ModelMessage(role="user", content=content),
                ],
                temperature=self.config.visual_temperature,
                provider_options={
                    "top_p": self.config.visual_top_p,
                    "enable_thinking": self.config.enable_thinking,
                    "reasoning_effort": self.config.reasoning_effort,
                    "max_completion_tokens": self.config.max_output_tokens,
                },
            )

            def validate(value: VisualExtractionPayload) -> VisualExtractionPayload:
                normalize_visual_extraction(value, batch)
                return value

            raw = await StructuredModelGateway(
                max_attempts=self.config.visual_max_attempts
            ).generate(
                self.vision_model_client,
                request,
                VisualExtractionPayload,
                validator=validate,
                scheduler=self.scheduler,
                workload_id=self.scheduler_workload_id,
                workload_limit=self.config.max_concurrency,
                stop_on_length=True,
            )
            return normalize_visual_extraction(raw, batch)

        return await self._run_jsonl_stage(
            store,
            stage="visual_extraction",
            path=paths.visual_extractions,
            inputs=inputs,
            schema=VisualExtractionResult,
            prompt=prompt,
            client=self.vision_model_client,
            model=self.config.vision_model,
            generator=generate,
        )

    async def _ocr_pages(
        self,
        project_root: Path,
        store: DatasetArtifactStore,
        paths: MangaPaths,
        scan: MangaScan,
    ) -> list[OcrPageResult]:
        existing: dict[int, OcrPageResult] = {}
        if paths.ocr_results.exists():
            for row in store.read_jsonl(paths.ocr_results):
                try:
                    result = OcrPageResult.model_validate(row)
                except ValueError:
                    continue
                existing[result.page_index] = result
        pending = [page for page in scan.pages if page.page_index not in existing]
        if pending:
            if self.ocr_runner is None:
                raise RuntimeError("OCR is enabled but no OCR runner is available")
            generated = await self.ocr_runner.recognize(
                project_root,
                pending,
                batch_size=self.config.ocr_batch_size,
            )
            existing.update((result.page_index, result) for result in generated)
            ordered = [existing[page.page_index] for page in scan.pages if page.page_index in existing]
            store.write_jsonl(
                paths.ocr_results,
                [page.model_dump(mode="json") for page in ordered],
            )
        missing = [page.page_index for page in scan.pages if page.page_index not in existing]
        if missing:
            raise IncompleteStageError("ocr", store.root, missing)
        return [existing[page.page_index] for page in scan.pages]

    async def _character_catalog(
        self,
        store: DatasetArtifactStore,
        paths: MangaPaths,
        observations: list[CharacterObservation],
    ) -> NamedCharacterCatalog:
        if paths.character_catalog.exists():
            try:
                return NamedCharacterCatalog.model_validate(
                    store.read_json(paths.character_catalog)
                )
            except ValueError:
                pass
        named = [value for value in observations if value.names]
        if not named:
            catalog = NamedCharacterCatalog(characters=[])
            store.write_json(paths.character_catalog, catalog.model_dump(mode="json"))
            return catalog
        payload = {
            "characters": [value.model_dump(mode="json") for value in named]
        }
        self._check_text_budget("manga_character_catalog", payload)
        try:
            raw = await self._generate_text(
                "manga_character_catalog",
                payload,
                NamedCharacterCatalogPayload,
            )
        except StructuredOutputError as error:
            store.write_json(paths.character_catalog, {})
            raise IncompleteStageError("character_catalog", store.root, [0]) from error
        catalog = clean_character_catalog(raw)
        store.write_json(paths.character_catalog, catalog.model_dump(mode="json"))
        return catalog

    async def _character_assignments(
        self,
        store: DatasetArtifactStore,
        paths: MangaPaths,
        observations: list[CharacterObservation],
        catalog: NamedCharacterCatalog,
    ) -> list[CharacterAssignment]:
        if not observations:
            store.write_jsonl(paths.character_assignments, [])
            return []
        if not catalog.characters:
            result = CharacterAssignmentsResult(
                assignments=[
                    CharacterAssignment(
                        batch_index=value.batch_index,
                        local_character_index=value.local_character_index,
                        name=None,
                    )
                    for value in observations
                ]
            )
            store.write_jsonl(
                paths.character_assignments,
                [
                    self._record(
                        {"batch_indexes": sorted({value.batch_index for value in observations})},
                        "programmatic-empty-catalog",
                        self.text_model_client.provider,
                        self.config.text_model,
                        result,
                    )
                ],
            )
            return result.assignments

        groups = self._assignment_groups(observations, catalog)
        inputs = [
            {
                "path": "work/manga/aligned_extractions.jsonl"
                if self.config.ocr_enabled
                else "work/manga/visual_extractions.jsonl",
                "batch_indexes": sorted({value.batch_index for value in group}),
                "catalog_path": "work/manga/character_catalog.json",
            }
            for group in groups
        ]
        prompt = self.prompts.load("manga_character_assignment")

        async def generate(index: int) -> CharacterAssignmentsResult:
            group = groups[index]
            payload = {
                "catalog": catalog.model_dump(mode="json"),
                "characters": [value.model_dump(mode="json") for value in group],
            }

            def validate(value: CharacterAssignmentPayload) -> CharacterAssignmentPayload:
                validate_character_assignments(value, group, catalog)
                return value

            raw = await self._generate_text(
                "manga_character_assignment",
                payload,
                CharacterAssignmentPayload,
                validator=validate,
            )
            return CharacterAssignmentsResult(
                assignments=validate_character_assignments(raw, group, catalog)
            )

        results = await self._run_jsonl_stage(
            store,
            stage="character_assignment",
            path=paths.character_assignments,
            inputs=inputs,
            schema=CharacterAssignmentsResult,
            prompt=prompt,
            client=self.text_model_client,
            model=self.config.text_model,
            generator=generate,
        )
        return [value for result in results for value in result.assignments]

    async def _dialogue_revisions(
        self,
        store: DatasetArtifactStore,
        paths: MangaPaths,
        chapters: MangaChaptersDocument,
        plots: PlotsDocument,
        catalog: NamedCharacterCatalog,
    ) -> list[DialogueRevisionResult]:
        inputs = [
            {"path": MANGA_PLOTS_PATH, "plot_index": plot.index}
            for plot in plots.plots
        ]
        prompt = self.prompts.load("manga_dialogue_revision")
        aliases = self._catalog_aliases(catalog)

        async def generate(index: int) -> DialogueRevisionResult:
            chapter = chapters.chapters[index]
            if not chapter.utterances:
                return DialogueRevisionResult(utterances=[])
            plot = plots.plots[index]
            payload = {
                "plot": "\n".join(chunk.text for chunk in plot.chunks),
                "characters": [
                    {"name": value.name, "aliases": value.aliases}
                    for value in catalog.characters
                ],
                "utterances": [
                    {
                        "speaker": value.speaker,
                        "content": value.content,
                    }
                    for value in chapter.utterances
                ],
            }
            self._check_text_budget("manga_dialogue_revision", payload)

            def validate(value: DialogueRevisionResult) -> DialogueRevisionResult:
                return self._normalize_revision(value, aliases)

            result = await self._generate_text(
                "manga_dialogue_revision",
                payload,
                DialogueRevisionResult,
                validator=validate,
            )
            return self._normalize_revision(result, aliases)

        return await self._run_jsonl_stage(
            store,
            stage="dialogue_revision",
            path=paths.dialogue_revisions,
            inputs=inputs,
            schema=DialogueRevisionResult,
            prompt=prompt,
            client=self.text_model_client,
            model=self.config.text_model,
            generator=generate,
        )

    async def _character_profiles(
        self,
        store: DatasetArtifactStore,
        paths: MangaPaths,
        observations: list[CharacterObservation],
        assignments: list[CharacterAssignment],
        catalog: NamedCharacterCatalog,
        revisions: list[DialogueRevisionResult],
    ) -> list[MangaCharacterProfileResult]:
        used = {
            line.speaker
            for revision in revisions
            for line in revision.utterances
            if line.speaker != "Unknown"
        }
        descriptions = aggregate_character_descriptions(observations, assignments)
        characters = [value for value in catalog.characters if value.name in used]
        inputs = [
            {
                "path": "work/manga/character_catalog.json",
                "name": character.name,
            }
            for character in characters
        ]
        prompt = self.prompts.load("manga_character_profile")

        async def generate(index: int) -> MangaCharacterProfileResult:
            character = characters[index]
            payload = {
                "character": {
                    "name": character.name,
                    "aliases": character.aliases,
                    "description": descriptions.get(
                        character.name,
                        character.description,
                    ),
                }
            }
            self._check_text_budget("manga_character_profile", payload)
            return await self._generate_text(
                "manga_character_profile",
                payload,
                MangaCharacterProfileResult,
            )

        return await self._run_jsonl_stage(
            store,
            stage="character_profile",
            path=paths.character_profiles,
            inputs=inputs,
            schema=MangaCharacterProfileResult,
            prompt=prompt,
            client=self.text_model_client,
            model=self.config.text_model,
            generator=generate,
        )

    def _assemble_profiles(
        self,
        store: DatasetArtifactStore,
        paths: MangaPaths,
        catalog: NamedCharacterCatalog,
        revisions: list[DialogueRevisionResult],
    ) -> list[CharacterProfile]:
        records = store.read_jsonl(paths.character_profiles)
        generated = self._read_stage_results(
            store,
            paths.character_profiles,
            MangaCharacterProfileResult,
        )
        catalog_by_name = {value.name: value for value in catalog.characters}
        profiles: list[CharacterProfile] = []
        for record, result in zip(records, generated, strict=True):
            name = str(record["input"]["name"])
            character = catalog_by_name[name]
            profiles.append(
                CharacterProfile(
                    name=name,
                    aliases=character.aliases,
                    profile=result.profile,
                    plot_refs=[
                        PlotRef(path=MANGA_PLOTS_PATH, index=index)
                        for index, revision in enumerate(revisions)
                        if any(line.speaker == name for line in revision.utterances)
                    ],
                )
            )
        return profiles

    async def _generate_text(
        self,
        stage: str,
        payload: JsonObject,
        schema: type[SchemaT],
        *,
        validator: Callable[[SchemaT], SchemaT] | None = None,
    ) -> SchemaT:
        prompt = self.prompts.load(stage)
        rendered = prompt.render()
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        content = (
            f"{rendered.user_prefix}\n{serialized}"
            if rendered.user_prefix
            else serialized
        )
        return await StructuredModelGateway(
            max_attempts=self.config.text_max_attempts
        ).generate(
            self.text_model_client,
            ModelRequest(
                model=self.config.text_model,
                messages=[
                    ModelMessage(role="system", content=rendered.system),
                    ModelMessage(role="user", content=content),
                ],
                temperature=0,
            ),
            schema,
            validator=validator,
            scheduler=self.scheduler,
            workload_id=self.scheduler_workload_id,
            workload_limit=self.config.max_concurrency,
        )

    async def _run_jsonl_stage(
        self,
        store: DatasetArtifactStore,
        *,
        stage: str,
        path: Path,
        inputs: list[JsonObject],
        schema: type[SchemaT],
        prompt: PromptTemplate,
        client: ModelClient,
        model: str,
        generator: Callable[[int], Awaitable[SchemaT]],
    ) -> list[SchemaT]:
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
                try:
                    if record.get("result") == {}:
                        raise ValueError("empty result")
                    schema.model_validate(record.get("result"))
                except (TypeError, ValueError):
                    record["result"] = {}
                    pending.append(index)
                records.append(record)
            else:
                records.append(
                    self._record(
                        input_value,
                        prompt.filename,
                        client.provider,
                        model,
                        None,
                    )
                )
                pending.append(index)
        store.write_jsonl(path, records)
        lock = asyncio.Lock()

        async def execute(index: int) -> None:
            try:
                value = await generator(index)
                records[index]["result"] = value.model_dump(mode="json")
                records[index].pop("error", None)
            except StructuredOutputError as error:
                records[index]["result"] = {}
                records[index]["error"] = str(error)
            async with lock:
                store.write_jsonl(path, records)

        await asyncio.gather(*(execute(index) for index in pending))
        failed = [
            index for index, record in enumerate(records) if record.get("result") == {}
        ]
        if failed:
            raise IncompleteStageError(stage, store.root, failed)
        return [schema.model_validate(record["result"]) for record in records]

    @staticmethod
    def _read_stage_results(
        store: DatasetArtifactStore,
        path: Path,
        schema: type[SchemaT],
    ) -> list[SchemaT]:
        results: list[SchemaT] = []
        for index, record in enumerate(store.read_jsonl(path)):
            if record.get("result") == {}:
                raise ValueError(f"{path.name} result is empty at index {index}")
            results.append(schema.model_validate(record.get("result")))
        return results

    def _assignment_groups(
        self,
        observations: list[CharacterObservation],
        catalog: NamedCharacterCatalog,
    ) -> list[list[CharacterObservation]]:
        by_batch: dict[int, list[CharacterObservation]] = {}
        for observation in observations:
            by_batch.setdefault(observation.batch_index, []).append(observation)
        groups: list[list[CharacterObservation]] = []
        current: list[CharacterObservation] = []
        for batch_index in sorted(by_batch):
            candidate = [*current, *by_batch[batch_index]]
            payload = {
                "catalog": catalog.model_dump(mode="json"),
                "characters": [value.model_dump(mode="json") for value in candidate],
            }
            if self._fits_text_budget("manga_character_assignment", payload):
                current = candidate
                continue
            if not current:
                raise ValueError(
                    f"character catalog plus batch {batch_index} exceeds text context"
                )
            groups.append(current)
            current = list(by_batch[batch_index])
            single = {
                "catalog": catalog.model_dump(mode="json"),
                "characters": [value.model_dump(mode="json") for value in current],
            }
            if not self._fits_text_budget("manga_character_assignment", single):
                raise ValueError(
                    f"character catalog plus batch {batch_index} exceeds text context"
                )
        if current:
            groups.append(current)
        return groups

    def _check_text_budget(self, stage: str, payload: JsonObject) -> None:
        if not self._fits_text_budget(stage, payload):
            raise ValueError(f"{stage} input exceeds selected text model context")

    def _fits_text_budget(self, stage: str, payload: JsonObject) -> bool:
        prompt = self.prompts.load(stage).render()
        text = prompt.system + prompt.user_prefix + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return self.tokenizer.count(text) <= (
            self.config.text_context_tokens - self.config.reserved_output_tokens
        )

    @staticmethod
    def _observations(
        results: list[VisualExtractionResult],
    ) -> list[CharacterObservation]:
        return [
            CharacterObservation(
                batch_index=batch_index,
                local_character_index=character.index,
                names=character.names,
                description=character.description,
            )
            for batch_index, result in enumerate(results)
            for character in result.characters
        ]

    def _plots(self, chapters: MangaChaptersDocument) -> PlotsDocument:
        return PlotsDocument(
            plots=[
                Plot(
                    index=chapter.index,
                    meta={
                        "chapter_index": chapter.index,
                        "title": chapter.title,
                        "page_indexes": chapter.page_indexes,
                    },
                    character_refs=[],
                    chunks=[
                        PlotChunk(
                            index=0,
                            text=chapter.plot,
                            token_count=self.tokenizer.count(chapter.plot),
                            source_refs=[],
                        )
                    ],
                )
                for chapter in chapters.chapters
            ]
        )

    @staticmethod
    def _conversations(
        revisions: list[DialogueRevisionResult],
    ) -> list[Conversation]:
        conversations: list[Conversation] = []
        for plot_index, revision in enumerate(revisions):
            if not revision.utterances:
                continue
            ref = PlotChunkRef(
                path=MANGA_PLOTS_PATH,
                plot_index=plot_index,
                chunk_index=0,
            )
            conversations.append(
                Conversation(
                    plot_ref=PlotRef(path=MANGA_PLOTS_PATH, index=plot_index),
                    utterances=[
                        Utterance(
                            index=index,
                            speaker=line.speaker,
                            content=line.content,
                            source_refs=[ref],
                        )
                        for index, line in enumerate(revision.utterances)
                    ],
                )
            )
        return conversations

    @staticmethod
    def _attach_character_refs(
        plots: PlotsDocument,
        profiles: list[CharacterProfile],
        conversations: list[Conversation],
    ) -> PlotsDocument:
        speakers_by_plot = {
            conversation.plot_ref.index: {
                line.speaker for line in conversation.utterances
            }
            for conversation in conversations
        }
        return PlotsDocument(
            plots=[
                plot.model_copy(
                    update={
                        "character_refs": [
                            CharacterRef(
                                path=MANGA_CHARACTER_PROFILES_PATH,
                                index=index,
                            )
                            for index, profile in enumerate(profiles)
                            if profile.name in speakers_by_plot.get(plot.index, set())
                        ]
                    }
                )
                for plot in plots.plots
            ]
        )

    @staticmethod
    def _catalog_aliases(catalog: NamedCharacterCatalog) -> dict[str, str]:
        aliases: dict[str, set[str]] = {}
        for character in catalog.characters:
            for value in [character.name, *character.aliases]:
                aliases.setdefault(value, set()).add(character.name)
        return {
            value: next(iter(names))
            for value, names in aliases.items()
            if len(names) == 1
        }

    @staticmethod
    def _normalize_revision(
        result: DialogueRevisionResult,
        aliases: dict[str, str],
    ) -> DialogueRevisionResult:
        values = []
        for line in result.utterances:
            if line.speaker == "Unknown":
                values.append(line)
                continue
            name = aliases.get(line.speaker)
            if name is None:
                raise ValueError(
                    f"dialogue speaker is not a unique catalog character: {line.speaker}"
                )
            values.append(line.model_copy(update={"speaker": name}))
        return DialogueRevisionResult(utterances=values)

    @staticmethod
    def _manga_resource(manifest: InputManifest) -> Any:
        resources = sorted(manifest.resources, key=lambda value: value.narrative_order)
        if len(resources) != 1 or resources[0].resource_type != "manga":
            raise ValueError("Manga Workflow requires exactly one manga folder resource")
        return resources[0]

    @staticmethod
    def _record(
        input_value: JsonObject,
        prompt_file: str,
        provider: str,
        model: str,
        result: BaseModel | None,
    ) -> JsonObject:
        return {
            "input": input_value,
            "prompt_file": prompt_file,
            "provider": provider,
            "model": model,
            "result": result.model_dump(mode="json") if result is not None else {},
        }

    async def _stage_event(
        self,
        stage: str,
        phase: StagePhase,
        artifact: Path,
    ) -> None:
        if self.stage_callback is not None:
            await self.stage_callback(stage, phase, artifact)

    @staticmethod
    def _write_character_files(
        paths: MangaPaths,
        profiles: list[CharacterProfile],
    ) -> None:
        for index, profile in enumerate(profiles):
            safe_name = _UNSAFE_FILE_CHARS.sub("-", profile.name).strip(" .-")
            path = paths.characters / f"{index:04d}-{safe_name or 'character'}.txt"
            path.write_text(profile.profile + "\n", encoding="utf-8", newline="\n")

    @staticmethod
    def _write_report(
        store: DatasetArtifactStore,
        paths: MangaPaths,
        scan: MangaScan,
        visual_results: list[VisualExtractionResult],
        ocr_pages: list[OcrPageResult],
        catalog: NamedCharacterCatalog,
        assignments: list[CharacterAssignment],
        revisions: list[DialogueRevisionResult],
        profiles: list[CharacterProfile],
        export_report: ShareGPTExportReport,
    ) -> None:
        empty_batches = sum(
            not result.utterances
            and not result.characters
            and not result.chapter_starts
            and not result.plot
            for result in visual_results
        )
        unknown_dialogue = sum(
            line.speaker == "Unknown"
            for revision in revisions
            for line in revision.utterances
        )
        warnings: list[str] = []
        if not catalog.characters:
            warnings.append("character catalog is empty")
        if any(value.name is None for value in assignments):
            warnings.append("some local characters remain unnamed")
        if unknown_dialogue:
            warnings.append("some dialogue speakers remain Unknown")
        if empty_batches:
            warnings.append("some visual batches contain no story result")
        failed_ocr = sum(page.error is not None for page in ocr_pages)
        if failed_ocr:
            warnings.append("some OCR pages failed and used VLM text")
        if export_report.sample_count == 0:
            warnings.append("ShareGPT export contains zero samples")
        store.write_json(
            paths.report,
            {
                "counts": {
                    "pages": len(scan.pages),
                    "batches": len(scan.batches),
                    "chapters": len(revisions),
                    "characters": len(profiles),
                    "conversations": sum(bool(value.utterances) for value in revisions),
                    "sharegpt_samples": export_report.sample_count,
                    "empty_visual_batches": empty_batches,
                    "unknown_dialogue": unknown_dialogue,
                    "failed_ocr_pages": failed_ocr,
                },
                "warnings": warnings,
            },
        )
