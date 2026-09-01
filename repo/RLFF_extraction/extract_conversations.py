#!/usr/bin/env python3
"""Extract ordered role-play utterances from rebuilt plot chunks."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import re
import sys
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import aiohttp
from dotenv import load_dotenv
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LLM.create_llm import ModelType, create_llm  # noqa: E402
from LLM.types import AIMessage, BaseMessage, LLMBase  # noqa: E402

JsonObject = dict[str, Any]
AttemptStatus = Literal[
    "success",
    "length",
    "validation_error",
    "response_error",
    "request_error",
]
DebugStreamKind = Literal["thinking", "content"]

ENVIRONMENT_SPEAKER = "Environment"
FUZZY_THRESHOLD = 80.0
PLOT_OPENING_PLUGIN = (
    "5. 此次对话提取中第一个对话内容必须是旁白用以开场，要求简短不能过长。"
)
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
PROMPT_SEPARATOR_RE = re.compile(r"^----------$", re.MULTILINE)
SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
SENTENCE_DELIMITERS = set("\n。！？!?，,；;：:.…「」")
LENGTH_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}


@dataclass(frozen=True, slots=True)
class ConversationCharacter:
    names: list[str]
    description: str


@dataclass(frozen=True, slots=True)
class Utterance:
    speaker: str
    content: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationResult:
    utterances: list[Utterance]


@dataclass(frozen=True, slots=True)
class PromptParts:
    system: str
    user_prefix: str


@dataclass(frozen=True, slots=True)
class ConversationJob:
    input_index: int
    plot_index: int
    source_chunk_start: int
    source_chunk_end: int
    source_token_count: int
    job_id: str
    volume: str | None
    chapter: str | None
    characters: list[ConversationCharacter]
    text: str


@dataclass(frozen=True, slots=True)
class LoadedJobs:
    title: str
    jobs: list[ConversationJob]
    plot_count: int
    source_chunk_count: int
    character_count: int


@dataclass(frozen=True, slots=True)
class Args:
    input_json: Path
    output_jsonl: Path
    prompt: Path
    provider: ModelType
    model: str
    base_url: str | None
    concurrency: int
    requests_per_second: float
    retries: int
    retry_delay_ms: int
    timeout_ms: int
    max_tokens: int
    temperature: float
    chunks_per_input: int
    start: int
    limit: int | None
    fuzzy_threshold: float
    overwrite: bool
    dry_run: bool
    debug: bool
    stream: bool


@dataclass(frozen=True, slots=True)
class LLMOutputInfo:
    content: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class FuzzyMatch:
    score: float
    alignment_score: float
    start: int
    end: int
    content: str


class ConversationValidationError(ValueError):
    def __init__(self, errors: Sequence[str], normalized_output: JsonObject) -> None:
        super().__init__("\n".join(errors))
        self.errors = tuple(errors)
        self.normalized_output = normalized_output


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def safe_filename_part(value: str) -> str:
    sanitized = SAFE_FILENAME_RE.sub("_", value).rstrip(" .")
    return sanitized or "conversation"


def job_key(plot_index: int, source_chunk_start: int, source_chunk_end: int) -> str:
    return f"{plot_index}:{source_chunk_start}-{source_chunk_end}"


def debug_log_path(input_json: Path, job_id: str, root: Path = PROJECT_ROOT) -> Path:
    return (
        root
        / "debug"
        / "conversations"
        / safe_filename_part(input_json.stem)
        / f"{safe_filename_part(job_id)}-log.txt"
    )


class DebugLogger:
    def __init__(self, path: Path, streamed: bool) -> None:
        self.path = path
        self.streamed = streamed
        self.current_section: DebugStreamKind | None = None

    def _write(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def start(
        self,
        attempt: int,
        total_attempts: int,
        started_at: str,
        messages: Sequence[BaseMessage],
        parameters: JsonObject,
    ) -> None:
        self.current_section = None
        self._write(
            "\n".join(
                [
                    f"=== ATTEMPT {attempt}/{total_attempts} ===",
                    f"started_at: {started_at}",
                    "",
                    "--- MESSAGES ---",
                    json_text(messages, indent=2),
                    "",
                    "--- ATTEMPT PARAMETERS ---",
                    json_text(parameters, indent=2),
                    "",
                    "",
                ]
            )
        )

    def stream(self, kind: DebugStreamKind, content: str) -> None:
        if not content:
            return
        if self.current_section != kind:
            separator = "" if self.current_section is None else "\n\n"
            heading = (
                "--- THINKING (STREAM) ---"
                if kind == "thinking"
                else "--- OUTPUT (STREAM) ---\ncontent:"
            )
            self._write(f"{separator}{heading}\n")
            self.current_section = kind
        self._write(content)

    def done(
        self,
        *,
        attempt: int,
        total_attempts: int,
        finished_at: str,
        duration_ms: int,
        thinking: str | None,
        output: LLMOutputInfo,
        status: AttemptStatus,
        error: str | None = None,
        parsed_json: JsonObject | None = None,
        validated_result: ConversationResult | None = None,
    ) -> None:
        output_data = asdict(output)
        content = output_data.pop("content")
        if not self.streamed:
            self._write(
                "\n".join(
                    [
                        "--- THINKING ---",
                        thinking or "(no thinking content returned)",
                        "",
                        "--- OUTPUT ---",
                        "content:",
                        content or "(no assistant output text returned)",
                        "",
                    ]
                )
            )
        elif self.current_section != "content":
            prefix = "" if self.current_section is None else "\n\n"
            self._write(f"{prefix}--- OUTPUT ---\n")

        lines = [
            "metadata:",
            json_text(output_data, indent=2),
            "",
            f"=== ATTEMPT {attempt}/{total_attempts} DONE ===",
            f"finished_at: {finished_at}",
            f"duration_ms: {duration_ms}",
            "",
            "--- PROCESSING RESULT ---",
            f"status: {status}",
        ]
        if error is not None:
            lines.append(f"error: {error}")
        if parsed_json is not None:
            lines.extend(["parsed_json:", json_text(parsed_json, indent=2)])
        if validated_result is not None:
            lines.extend(["validated_result:", json_text(result_dict(validated_result), indent=2)])
        prefix = "\n\n" if self.streamed else ""
        body = "\n".join(lines)
        self._write(f"{prefix}{body}\n\n")


def initialize_debug_log(
    path: Path,
    title: str,
    job: ConversationJob,
    parameters: JsonObject,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "RLFF conversation extraction debug log",
                "",
                "=== CHUNK ===",
                f"source_title: {title}",
                f"input_index: {job.input_index}",
                f"plot_index: {job.plot_index}",
                f"source_chunk_start: {job.source_chunk_start}",
                f"source_chunk_end: {job.source_chunk_end}",
                f"source_token_count: {job.source_token_count}",
                f"job_id: {job.job_id}",
                f"volume: {job.volume or ''}",
                f"chapter: {job.chapter or ''}",
                "",
                "=== CALL PARAMETERS ===",
                json_text(parameters, indent=2),
                "",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def split_prompt(prompt: str) -> PromptParts:
    normalized = prompt.replace("\r\n", "\n").replace("\r", "\n")
    delimiters = list(PROMPT_SEPARATOR_RE.finditer(normalized))
    if not delimiters:
        raise ValueError("Prompt has no standalone ---------- separator.")
    delimiter = delimiters[-1]
    system = normalized[: delimiter.start()].strip()
    user_prefix = normalized[delimiter.end() :].strip()
    if not system:
        raise ValueError("Prompt system section is empty.")
    return PromptParts(system=system, user_prefix=user_prefix)


def exact_fields(value: Mapping[str, Any], expected: Iterable[str], field: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{field} fields are invalid.")


def nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def string_or_none(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null.")
    return value


def merge_plot_characters(raw_plots: Sequence[Any]) -> list[ConversationCharacter]:
    characters: list[ConversationCharacter] = []
    for plot_index, raw_plot in enumerate(raw_plots):
        if not isinstance(raw_plot, dict):
            raise ValueError(f"plots[{plot_index}] must be an object.")
        raw_characters = raw_plot.get("characters")
        if not isinstance(raw_characters, list):
            raise ValueError(f"plots[{plot_index}].characters must be an array.")
        for character_index, raw_character in enumerate(raw_characters):
            field = f"plots[{plot_index}].characters[{character_index}]"
            if not isinstance(raw_character, dict):
                raise ValueError(f"{field} must be an object.")
            exact_fields(raw_character, ["name", "description"], field)
            raw_names = raw_character.get("name")
            if (
                not isinstance(raw_names, list)
                or not raw_names
                or any(not isinstance(name, str) or not name.strip() for name in raw_names)
            ):
                raise ValueError(f"{field}.name must be a non-empty string array.")
            names = cast(list[str], raw_names)
            description = nonempty_string(raw_character.get("description"), f"{field}.description")
            aliases = set(names)
            overlapping = [
                index
                for index, existing in enumerate(characters)
                if any(name in aliases for name in existing.names)
            ]
            if not overlapping:
                characters.append(ConversationCharacter(list(names), description))
                continue

            combined = [characters[index] for index in overlapping]
            combined.append(ConversationCharacter(list(names), description))
            merged_names = list(dict.fromkeys(name for item in combined for name in item.names))
            merged_description = max(
                (item.description for item in combined),
                key=len,
            )
            characters[overlapping[0]] = ConversationCharacter(
                merged_names,
                merged_description,
            )
            for index in reversed(overlapping[1:]):
                characters.pop(index)

    alias_owners: dict[str, str] = {}
    for character in characters:
        canonical = character.names[0]
        for name in character.names:
            owner = alias_owners.get(name)
            if owner is not None and owner != canonical:
                raise ValueError(f"Character alias {name} belongs to both {owner} and {canonical}.")
            alias_owners[name] = canonical
    return characters


def build_jobs(root: JsonObject, fallback_title: str, chunks_per_input: int = 4) -> LoadedJobs:
    raw_plots = root.get("plots")
    if not isinstance(raw_plots, list):
        raise ValueError("Input JSON must contain plots.")
    if chunks_per_input < 1:
        raise ValueError("chunks_per_input must be a positive integer.")

    characters = merge_plot_characters(raw_plots)
    jobs: list[ConversationJob] = []
    source_chunk_count = 0
    for plot_index, raw_plot in enumerate(raw_plots):
        if not isinstance(raw_plot, dict):
            raise ValueError(f"plots[{plot_index}] must be an object.")
        raw_chunks = raw_plot.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError(f"plots[{plot_index}].chunks must be an array.")
        volume = string_or_none(raw_plot.get("volume"), f"plots[{plot_index}].volume")
        chapter = string_or_none(raw_plot.get("chapter"), f"plots[{plot_index}].chapter")
        chunks: list[tuple[str, int]] = []
        for chunk_index, raw_chunk in enumerate(raw_chunks):
            field = f"plots[{plot_index}].chunks[{chunk_index}]"
            if not isinstance(raw_chunk, dict):
                raise ValueError(f"{field} must be an object.")
            text = raw_chunk.get("text")
            token = raw_chunk.get("token")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{field}.text is invalid.")
            if type(token) is not int or token < 0:
                raise ValueError(f"{field}.token is invalid.")
            chunks.append((text, token))
        source_chunk_count += len(chunks)

        for source_chunk_start in range(0, len(chunks), chunks_per_input):
            source_chunk_end = min(source_chunk_start + chunks_per_input, len(chunks)) - 1
            selected = chunks[source_chunk_start : source_chunk_end + 1]
            input_index = len(jobs)
            jobs.append(
                ConversationJob(
                    input_index=input_index,
                    plot_index=plot_index,
                    source_chunk_start=source_chunk_start,
                    source_chunk_end=source_chunk_end,
                    source_token_count=sum(token for _, token in selected),
                    job_id=(
                        f"input-{input_index:06d}-plot-{plot_index:06d}"
                        f"-chunks-{source_chunk_start:04d}-{source_chunk_end:04d}"
                    ),
                    volume=volume,
                    chapter=chapter,
                    characters=characters,
                    text="".join(text for text, _ in selected),
                )
            )
    title = root.get("title") if isinstance(root.get("title"), str) else fallback_title
    return LoadedJobs(
        title=cast(str, title),
        jobs=jobs,
        plot_count=len(raw_plots),
        source_chunk_count=source_chunk_count,
        character_count=len(characters),
    )


def load_jobs(path: Path, chunks_per_input: int = 4) -> LoadedJobs:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(root, dict):
        raise ValueError("Input JSON must be an object.")
    return build_jobs(root, path.stem, chunks_per_input)


def build_messages(
    prompt: PromptParts,
    input_text: str,
    characters: Sequence[ConversationCharacter],
    retry_feedback: str | None = None,
    retry_output: str | None = None,
    *,
    is_plot_start: bool = False,
) -> list[BaseMessage]:
    input_json = json_text(
        {
            "characters": [asdict(character) for character in characters],
            "input": input_text,
        },
        indent=2,
    )
    user_content = f"{prompt.user_prefix}\n{input_json}" if prompt.user_prefix else input_json
    system_content = prompt.system.replace(
        "{plugin_a}",
        PLOT_OPENING_PLUGIN if is_plot_start else "",
    )
    messages: list[BaseMessage] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    if retry_feedback is not None:
        if retry_output is not None:
            messages.append({"role": "ai", "content": retry_output})
        messages.append({"role": "user", "content": retry_feedback})
    return messages


def parse_json_object(text: str) -> JsonObject:
    source = text.strip()
    fenced = JSON_FENCE_RE.fullmatch(source)
    if fenced is not None:
        source = fenced.group(1).strip()
    parsed = json.loads(source)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object.")
    return parsed


def is_environment_narration(content: str) -> bool:
    return content.startswith("*(") and content.endswith(")*") and bool(content[2:-2].strip())


def expand_to_sentence(
    source: str,
    start: int,
    end: int,
    lower_bound: int,
) -> tuple[int, int]:
    expanded_start = start
    if source[start] not in SENTENCE_DELIMITERS:
        while (
            expanded_start > lower_bound and source[expanded_start - 1] not in SENTENCE_DELIMITERS
        ):
            expanded_start -= 1

    expanded_end = end
    if source[end - 1] not in SENTENCE_DELIMITERS:
        while expanded_end < len(source) and source[expanded_end] not in SENTENCE_DELIMITERS:
            expanded_end += 1
        if expanded_end < len(source) and source[expanded_end] != "\n":
            expanded_end += 1
    return expanded_start, expanded_end


def locate_utterance(
    utterance: str,
    source: str,
    cursor: int,
    threshold: float = FUZZY_THRESHOLD,
) -> FuzzyMatch | None:
    search_text = utterance
    if utterance.startswith("*(") and utterance.endswith(")*"):
        search_text = utterance[2:-2]
    if not search_text or cursor >= len(source):
        return None

    exact_start = source.find(search_text, cursor)
    if exact_start >= 0:
        alignment_score = 100.0
        start = exact_start
        end = exact_start + len(search_text)
    else:
        alignment = fuzz.partial_ratio_alignment(
            search_text,
            source[cursor:],
            score_cutoff=threshold,
        )
        if alignment is None:
            return None
        alignment_score = float(alignment.score)
        start = cursor + int(alignment.dest_start)
        end = cursor + int(alignment.dest_end)
        if alignment_score < threshold or not cursor <= start < end <= len(source):
            return None

    expanded_start, expanded_end = expand_to_sentence(source, start, end, cursor)
    expanded_content = source[expanded_start:expanded_end]
    return FuzzyMatch(
        score=alignment_score,
        alignment_score=alignment_score,
        start=expanded_start,
        end=expanded_end,
        content=expanded_content,
    )


def _normalized_validation_output(raw: JsonObject) -> JsonObject:
    return copy.deepcopy(raw)


def validate_result(
    raw: JsonObject,
    input_text: str,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    *,
    allow_stored_source: bool = False,
) -> ConversationResult:
    errors: list[str] = []
    normalized_output = _normalized_validation_output(raw)
    try:
        exact_fields(raw, ["utterances"], "Output")
    except ValueError as error:
        errors.append(str(error))

    raw_utterances = raw.get("utterances")
    if not isinstance(raw_utterances, list):
        errors.append("Output utterances must be an array.")
        raise ConversationValidationError(errors, normalized_output)
    normalized_raw_utterances = normalized_output.get("utterances")
    if not isinstance(normalized_raw_utterances, list):
        normalized_raw_utterances = copy.deepcopy(raw_utterances)
        normalized_output["utterances"] = normalized_raw_utterances

    source_cursor = 0
    utterances: list[Utterance] = []
    for index, value in enumerate(raw_utterances):
        if not isinstance(value, dict):
            errors.append(f"utterances[{index}] must be an object.")
            continue
        try:
            expected_fields = ["speaker", "content"]
            if allow_stored_source and "source" in value:
                expected_fields.append("source")
            exact_fields(value, expected_fields, f"utterances[{index}]")
        except ValueError as error:
            errors.append(str(error))

        speaker_value = value.get("speaker")
        content_value = value.get("content")
        speaker = (
            speaker_value if isinstance(speaker_value, str) and speaker_value.strip() else None
        )
        content = (
            content_value if isinstance(content_value, str) and content_value.strip() else None
        )
        if speaker is None:
            errors.append(f"utterances[{index}].speaker must be a non-empty string.")
        if content is None:
            errors.append(f"utterances[{index}].content must be a non-empty string.")
        if speaker is None or content is None:
            continue

        normalized_content = content
        matched_source: str | None = None
        if speaker == ENVIRONMENT_SPEAKER:
            if not is_environment_narration(content):
                errors.append(f"utterances[{index}].content must use *(...)* for Environment.")
            if allow_stored_source and "source" in value:
                errors.append(
                    f"utterances[{index}].source is only allowed for a fuzzy-matched dialogue."
                )
        else:
            match = locate_utterance(content, input_text, source_cursor, fuzzy_threshold)
            if match is None:
                errors.append(
                    f"utterances[{index}].content fuzzy match score is below "
                    f"{fuzzy_threshold:g} or has no valid ordered sentence match."
                )
            else:
                if match.score < 100.0:
                    matched_source = match.content
                    if allow_stored_source:
                        source_value = value.get("source")
                        if not isinstance(source_value, str) or not source_value:
                            errors.append(
                                f"utterances[{index}].source must contain the matched original "
                                "text when content is not an exact match."
                            )
                        elif source_value != matched_source:
                            errors.append(
                                f"utterances[{index}].source does not match the located "
                                "original text."
                            )
                else:
                    normalized_content = match.content
                    if allow_stored_source and "source" in value:
                        errors.append(
                            f"utterances[{index}].source must be omitted when content is an "
                            "exact match."
                        )
                source_cursor = match.end
                normalized_value = normalized_raw_utterances[index]
                if isinstance(normalized_value, dict) and match.score == 100.0:
                    normalized_value["content"] = normalized_content
        utterances.append(
            Utterance(
                speaker=speaker,
                content=normalized_content,
                source=matched_source,
            )
        )

    if errors:
        raise ConversationValidationError(errors, normalized_output)
    return ConversationResult(utterances=utterances)


def validate_stored_result(
    raw: JsonObject,
    input_text: str,
    model: Any,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
) -> ConversationResult:
    compatible_result = {"utterances": raw.get("utterances")} if "characters" in raw else raw
    if model != "sft-v2-migration":
        return validate_result(
            compatible_result,
            input_text,
            fuzzy_threshold,
            allow_stored_source=True,
        )

    exact_fields(compatible_result, ["utterances"], "Stored output")
    raw_utterances = compatible_result.get("utterances")
    if not isinstance(raw_utterances, list):
        raise ValueError("Stored output utterances must be an array.")
    utterances: list[Utterance] = []
    for index, value in enumerate(raw_utterances):
        if not isinstance(value, dict):
            raise ValueError(f"utterances[{index}] must be an object.")
        exact_fields(value, ["speaker", "content"], f"utterances[{index}]")
        speaker = value.get("speaker")
        if not isinstance(speaker, str):
            raise ValueError(f"utterances[{index}].speaker must be a string.")
        utterances.append(
            Utterance(
                speaker=speaker,
                content=nonempty_string(value.get("content"), f"utterances[{index}].content"),
            )
        )
    return ConversationResult(utterances=utterances)


def result_dict(result: ConversationResult) -> JsonObject:
    utterances: list[JsonObject] = []
    for utterance in result.utterances:
        value: JsonObject = {
            "speaker": utterance.speaker,
            "content": utterance.content,
        }
        if utterance.source is not None:
            value["source"] = utterance.source
        utterances.append(value)
    return {"utterances": utterances}


def invalid_json_retry_feedback(content: str) -> str:
    return "\n".join(
        [
            "[Error]结果不是合法的JSON格式，请检查输出结果：",
            content,
            "请重新生成完整输出，并且仅返回合法JSON。",
        ]
    )


def validation_retry_feedback(error: Exception, result: JsonObject) -> str:
    return "\n".join(
        [
            f"[Error]对话提取结果校验失败：{error}",
            "请检查结果：",
            json_text(result, indent=2),
            "请重新生成完整输出，并且仅返回修复后的合法JSON。",
        ]
    )


def nested_value(value: Any, *path: str | int) -> Any:
    current = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list) or not 0 <= part < len(current):
                return None
            current = current[part]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
    return current


def first_string(*values: Any) -> str | None:
    return next((value for value in values if isinstance(value, str)), None)


def finite_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return int(value)
    return None


def assistant_text(response: Any) -> str:
    content = first_string(
        nested_value(response, "choices", 0, "message", "content"),
        nested_value(response, "output", "choices", 0, "message", "content"),
        nested_value(response, "output", "text"),
        nested_value(response, "content"),
    )
    if content is None:
        raise ValueError("LLM response contains no assistant text.")
    return content


def assistant_thinking(response: Any) -> str:
    content = first_string(
        nested_value(response, "choices", 0, "message", "reasoning_content"),
        nested_value(response, "choices", 0, "message", "thinking"),
        nested_value(response, "output", "choices", 0, "message", "reasoning_content"),
        nested_value(response, "output", "choices", 0, "message", "thinking"),
        nested_value(response, "output", "message", "reasoning_content"),
        nested_value(response, "output", "message", "thinking"),
        nested_value(response, "reasoning_content"),
        nested_value(response, "thinking"),
    )
    return content.strip() if content else ""


def extract_llm_output_info(response: Any, content: str | None) -> LLMOutputInfo:
    usage = nested_value(response, "usage") or nested_value(response, "output", "usage") or {}
    return LLMOutputInfo(
        content=content,
        finish_reason=first_string(
            nested_value(response, "choices", 0, "finish_reason"),
            nested_value(response, "output", "choices", 0, "finish_reason"),
            nested_value(response, "finish_reason"),
            nested_value(response, "stop_reason"),
        ),
        input_tokens=finite_int(
            nested_value(usage, "input_tokens"),
            nested_value(usage, "prompt_tokens"),
        ),
        output_tokens=finite_int(
            nested_value(usage, "output_tokens"),
            nested_value(usage, "completion_tokens"),
        ),
        total_tokens=finite_int(nested_value(usage, "total_tokens")),
    )


async def collect_assistant_stream(
    source: AsyncIterator[AIMessage],
    debug_logger: DebugLogger | None = None,
) -> tuple[str, str, LLMOutputInfo]:
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    metadata: JsonObject = {}
    async for event in source:
        metadata.update(event.args)
        if event.type == "thinking" and event.content:
            thinking_parts.append(event.content)
            if debug_logger is not None:
                debug_logger.stream("thinking", event.content)
        elif event.type == "content" and event.content:
            content_parts.append(event.content)
            if debug_logger is not None:
                debug_logger.stream("content", event.content)
    thinking = "".join(thinking_parts).strip()
    content = "".join(content_parts)
    return thinking, content, extract_llm_output_info(metadata, content or None)


def is_length_finish_reason(reason: str | None) -> bool:
    return reason is not None and reason.lower() in LENGTH_FINISH_REASONS


def is_timeout_error(error: Exception) -> bool:
    return isinstance(error, (TimeoutError, asyncio.TimeoutError)) or (
        isinstance(error, aiohttp.ClientError) and "timeout" in str(error).lower()
    )


class RateGate:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1 / requests_per_second
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = asyncio.get_running_loop().time()
            wait_seconds = max(0.0, self.next_start - now)
            if wait_seconds:
                await asyncio.sleep(wait_seconds)
            self.next_start = asyncio.get_running_loop().time() + self.interval


async def extract_one(
    llm: LLMBase,
    limiter: RateGate,
    job: ConversationJob,
    prompt: PromptParts,
    *,
    retries: int,
    retry_delay_ms: int,
    streamed: bool,
    provider: ModelType,
    generation_parameters: JsonObject,
    fuzzy_threshold: float,
    debug_logger: DebugLogger | None = None,
) -> ConversationResult:
    total_attempts = retries + 1
    reasoning_effort: Literal["max", "high"] = "max"
    retry_feedback: str | None = None
    retry_output: str | None = None

    for attempt in range(total_attempts):
        attempt_number = attempt + 1
        messages = build_messages(
            prompt,
            job.text,
            job.characters,
            retry_feedback,
            retry_output,
            is_plot_start=job.source_chunk_start == 0,
        )
        runtime_parameters = {
            **generation_parameters,
            **({"reasoning_effort": reasoning_effort} if provider == "DeepSeek" else {}),
        }
        await limiter.wait()
        started_at = now_iso()
        started_clock = time.perf_counter()
        if debug_logger is not None:
            debug_logger.start(
                attempt_number,
                total_attempts,
                started_at,
                messages,
                {"stream": streamed, **runtime_parameters},
            )

        received = False
        thinking = ""
        content = ""
        output = LLMOutputInfo()

        def finish(
            status: AttemptStatus,
            error: Exception | None = None,
            parsed_json: JsonObject | None = None,
            validated_result: ConversationResult | None = None,
            *,
            current_attempt: int = attempt_number,
            current_started_clock: float = started_clock,
            current_thinking: str,
            current_output: LLMOutputInfo,
        ) -> None:
            if debug_logger is None:
                return
            debug_logger.done(
                attempt=current_attempt,
                total_attempts=total_attempts,
                finished_at=now_iso(),
                duration_ms=round((time.perf_counter() - current_started_clock) * 1000),
                thinking=current_thinking or None,
                output=current_output,
                status=status,
                error=str(error) if error is not None else None,
                parsed_json=parsed_json,
                validated_result=validated_result,
            )

        try:
            if streamed:
                stream = llm.stream(messages, runtime_parameters)
                received = True
                thinking, content, output = await collect_assistant_stream(
                    stream,
                    debug_logger,
                )
                if not content:
                    raise ValueError("LLM response contains no assistant text.")
            else:
                response = await llm.invoke(messages, runtime_parameters)
                received = True
                thinking = assistant_thinking(response)
                output = extract_llm_output_info(response, None)
                content = assistant_text(response)
                output = LLMOutputInfo(**{**asdict(output), "content": content})
        except Exception as error:
            length = is_length_finish_reason(output.finish_reason)
            finish(
                "length" if length else "response_error" if received else "request_error",
                error,
                current_thinking=thinking,
                current_output=output,
            )
            if (length or is_timeout_error(error)) and attempt < retries:
                if length and provider == "DeepSeek":
                    reasoning_effort = "high"
                await asyncio.sleep(retry_delay_ms * attempt_number / 1000)
                continue
            raise

        if is_length_finish_reason(output.finish_reason):
            length_error = RuntimeError(
                f"LLM stopped because finish_reason={output.finish_reason}."
            )
            finish(
                "length",
                length_error,
                current_thinking=thinking,
                current_output=output,
            )
            if attempt < retries:
                if provider == "DeepSeek":
                    reasoning_effort = "high"
                await asyncio.sleep(retry_delay_ms * attempt_number / 1000)
                continue
            raise length_error

        try:
            parsed_json = parse_json_object(content)
        except (json.JSONDecodeError, ValueError) as error:
            retry_error = ValueError(invalid_json_retry_feedback(content))
            finish(
                "validation_error",
                retry_error,
                current_thinking=thinking,
                current_output=output,
            )
            if attempt < retries:
                retry_feedback = str(retry_error)
                retry_output = content
                await asyncio.sleep(retry_delay_ms * attempt_number / 1000)
                continue
            raise retry_error from error

        try:
            result = validate_result(parsed_json, job.text, fuzzy_threshold)
            finish(
                "success",
                parsed_json=parsed_json,
                validated_result=result,
                current_thinking=thinking,
                current_output=output,
            )
            return result
        except ConversationValidationError as error:
            normalized_output = error.normalized_output
            retry_error = ValueError(validation_retry_feedback(error, normalized_output))
            finish(
                "validation_error",
                retry_error,
                parsed_json=parsed_json,
                current_thinking=thinking,
                current_output=output,
            )
            if attempt < retries:
                retry_feedback = str(retry_error)
                retry_output = json_text(normalized_output, indent=2)
                await asyncio.sleep(retry_delay_ms * attempt_number / 1000)
                continue
            raise retry_error from error

    raise RuntimeError("Conversation extraction retry loop exhausted.")


def completed_results(
    path: Path,
    jobs: Sequence[ConversationJob],
    fuzzy_threshold: float,
) -> dict[str, ConversationResult]:
    if not path.exists():
        return {}
    job_by_key = {
        job_key(job.plot_index, job.source_chunk_start, job.source_chunk_end): job for job in jobs
    }
    results: dict[str, ConversationResult] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Existing conversation JSONL line {line_number} is invalid.")
        indexes = [
            row.get("input_index"),
            row.get("plot_index"),
            row.get("source_chunk_start"),
            row.get("source_chunk_end"),
        ]
        if any(type(value) is not int for value in indexes):
            raise ValueError("Existing conversation JSONL contains an invalid job index.")
        key = job_key(
            cast(int, row["plot_index"]),
            cast(int, row["source_chunk_start"]),
            cast(int, row["source_chunk_end"]),
        )
        job = job_by_key.get(key)
        if job is None:
            raise ValueError(f"Existing conversation JSONL contains unknown job {key}.")
        if key in results:
            raise ValueError(f"Existing conversation JSONL duplicates job {key}.")
        if row["input_index"] != job.input_index:
            raise ValueError(
                f"Existing conversation JSONL has the wrong input_index for job {key}."
            )
        raw_result = row.get("result")
        if not isinstance(raw_result, dict):
            raise ValueError(f"Existing conversation JSONL has no result for job {key}.")
        try:
            results[key] = validate_stored_result(
                raw_result,
                job.text,
                row.get("model"),
                fuzzy_threshold,
            )
        except ValueError as error:
            raise ValueError(
                f"Existing conversation JSONL line {line_number}, job {key}, "
                f"failed validation:\n{error}"
            ) from error
    return results


def output_record(
    title: str,
    job: ConversationJob,
    result: ConversationResult,
    model: str,
) -> JsonObject:
    return {
        "model": model,
        "source_title": title,
        "input_index": job.input_index,
        "plot_index": job.plot_index,
        "source_chunk_start": job.source_chunk_start,
        "source_chunk_end": job.source_chunk_end,
        "source_token_count": job.source_token_count,
        "volume": job.volume,
        "chapter": job.chapter,
        "result": result_dict(result),
    }


def select_jobs(
    jobs: Sequence[ConversationJob],
    completed: set[str],
    start: int,
    limit: int | None = None,
) -> list[ConversationJob]:
    pending = [
        job
        for job in jobs
        if job.input_index >= start
        and job_key(job.plot_index, job.source_chunk_start, job.source_chunk_end) not in completed
    ]
    return pending if limit is None else pending[:limit]


def provider_value(value: str) -> ModelType:
    providers: dict[str, ModelType] = {
        "deepseek": "DeepSeek",
        "glm": "GLM",
        "qwen": "Qwen",
    }
    try:
        return providers[value.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"Unsupported provider: {value}") from error


def parse_args(argv: Sequence[str] | None = None) -> Args:
    configured_model = (
        os.getenv("RLFF_CONVERSATION_MODEL")
        or os.getenv("RLFF_PLOT_MODEL")
        or os.getenv("RLFF_VERIFIER_MODEL")
    )
    configured_provider = (
        os.getenv("RLFF_CONVERSATION_PROVIDER") or os.getenv("RLFF_PLOT_PROVIDER") or "DeepSeek"
    )
    parser = argparse.ArgumentParser(
        description="Extract ordered conversations from rebuilt plot JSON.",
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "conversation_extraction.txt")
    parser.add_argument(
        "--provider",
        type=provider_value,
        default=provider_value(configured_provider),
    )
    parser.add_argument("--model", default=configured_model or "")
    parser.add_argument("--base-url")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests-per-second", type=float, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-ms", type=int, default=2000)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--max-tokens", type=int, default=25_000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--chunks-per-input", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fuzzy-threshold", type=float, default=FUZZY_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stream", action="store_true")
    namespace = parser.parse_args(argv)

    for name, minimum in [
        ("concurrency", 1),
        ("retries", 0),
        ("retry_delay_ms", 0),
        ("timeout_ms", 1),
        ("max_tokens", 1),
        ("chunks_per_input", 1),
        ("start", 0),
    ]:
        value = getattr(namespace, name)
        if value < minimum:
            parser.error(f"{name} must be at least {minimum}")
    if namespace.requests_per_second <= 0:
        parser.error("requests_per_second must be positive")
    if not 0 <= namespace.temperature <= 2:
        parser.error("temperature must be between 0 and 2")
    if namespace.limit is not None and namespace.limit < 1:
        parser.error("limit must be a positive integer")
    if not 0 <= namespace.fuzzy_threshold <= 100:
        parser.error("fuzzy_threshold must be between 0 and 100")
    if not namespace.model:
        if namespace.provider == "DeepSeek":
            namespace.model = "deepseek-v4-pro"
        else:
            parser.error("--model or RLFF_CONVERSATION_MODEL is required for GLM and Qwen")
    if namespace.base_url is None:
        namespace.base_url = {
            "DeepSeek": os.getenv("DEEPSEEK_BASE_URL"),
            "GLM": os.getenv("GLM_BASE_URL"),
            "Qwen": os.getenv("DASHSCOPE_BASE_URL"),
        }[namespace.provider]
    return Args(**vars(namespace))


async def run(args: Args) -> None:
    loaded = load_jobs(args.input_json, args.chunks_per_input)
    prompt = split_prompt(args.prompt.read_text(encoding="utf-8-sig"))
    results = (
        {}
        if args.overwrite
        else completed_results(args.output_jsonl, loaded.jobs, args.fuzzy_threshold)
    )
    jobs = select_jobs(loaded.jobs, set(results), args.start, args.limit)
    if args.overwrite and not args.dry_run and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    summary = {
        "provider": args.provider,
        "model": args.model,
        "source_title": loaded.title,
        "plot_count": loaded.plot_count,
        "source_chunk_count": loaded.source_chunk_count,
        "input_count": len(loaded.jobs),
        "character_count": loaded.character_count,
        "chunks_per_input": args.chunks_per_input,
        "selected_count": len(jobs),
        "existing_count": len(results),
        "start": args.start,
        "concurrency": args.concurrency,
        "fuzzy_threshold": args.fuzzy_threshold,
        "debug": args.debug,
        "stream": args.stream,
    }
    print(json_text(summary, indent=2))
    if args.dry_run:
        if jobs:
            print(
                json_text(
                    build_messages(
                        prompt,
                        jobs[0].text,
                        jobs[0].characters,
                        is_plot_start=jobs[0].source_chunk_start == 0,
                    ),
                    indent=2,
                )
            )
        return
    if not jobs:
        return

    provider_parameters: JsonObject = (
        {"reasoning_effort": "max"} if args.provider == "DeepSeek" else {"enable_thinking": True}
    )
    client_config: JsonObject = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_ms": args.timeout_ms,
        "max_retries": 0,
        **({"base_url": args.base_url} if args.base_url else {}),
        **provider_parameters,
    }
    generation_parameters: JsonObject = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        **provider_parameters,
    }
    debug_call_parameters: JsonObject = {
        "provider": args.provider,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout_ms": args.timeout_ms,
        "retries": args.retries,
        "retry_delay_ms": args.retry_delay_ms,
        "requests_per_second": args.requests_per_second,
        "concurrency": args.concurrency,
        "fuzzy_threshold": args.fuzzy_threshold,
        "stream": args.stream,
        **({"base_url": args.base_url} if args.base_url else {}),
        **provider_parameters,
    }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[ConversationJob] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    limiter = RateGate(args.requests_per_second)
    failures: list[JsonObject] = []
    completed = 0

    async with create_llm(args.provider, client_config) as llm:

        async def worker() -> None:
            nonlocal completed
            while True:
                try:
                    job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    log_path = debug_log_path(args.input_json, job.job_id) if args.debug else None
                    debug_logger = None
                    if log_path is not None:
                        initialize_debug_log(
                            log_path,
                            loaded.title,
                            job,
                            debug_call_parameters,
                        )
                        debug_logger = DebugLogger(log_path, args.stream)
                    result = await extract_one(
                        llm,
                        limiter,
                        job,
                        prompt,
                        retries=args.retries,
                        retry_delay_ms=args.retry_delay_ms,
                        streamed=args.stream,
                        provider=args.provider,
                        generation_parameters=generation_parameters,
                        fuzzy_threshold=args.fuzzy_threshold,
                        debug_logger=debug_logger,
                    )
                    key = job_key(
                        job.plot_index,
                        job.source_chunk_start,
                        job.source_chunk_end,
                    )
                    results[key] = result
                    with args.output_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(
                            json_text(output_record(loaded.title, job, result, args.model)) + "\n"
                        )
                    completed += 1
                    print(f"[{completed}/{len(jobs)}] {job.job_id}")
                except Exception as error:
                    failures.append({"job_id": job.job_id, "error": str(error)})
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(min(args.concurrency, len(jobs)))))

    print(
        json_text(
            {"completed": completed, "failed": len(failures), "failures": failures},
            indent=2,
        )
    )
    if failures:
        raise RuntimeError(f"{len(failures)} conversation extraction job(s) failed.")


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
