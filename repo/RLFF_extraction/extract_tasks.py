#!/usr/bin/env python3
"""Generate detailed plot descriptions and per-character tasks from rebuilt plots."""

from __future__ import annotations

import argparse
import asyncio
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
ReasoningEffort = Literal["low", "medium", "high", "max"]

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
PROMPT_SEPARATOR_RE = re.compile(r"^----------$", re.MULTILINE)
SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
LENGTH_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}
MIN_PLOT_TOKENS = 512


@dataclass(frozen=True, slots=True)
class PlotCharacter:
    names: list[str]
    description: str


@dataclass(frozen=True, slots=True)
class CharacterTask:
    name: str
    values: list[str]


@dataclass(frozen=True, slots=True)
class TaskGenerationResult:
    plot: str
    tasks: list[CharacterTask]


@dataclass(frozen=True, slots=True)
class PromptParts:
    system: str
    user_prefix: str


@dataclass(frozen=True, slots=True)
class PlotJob:
    plot_index: int
    job_id: str
    volume: str | None
    chapter: str | None
    source_chunk_count: int
    source_token_count: int
    characters: list[PlotCharacter]
    text: str


@dataclass(frozen=True, slots=True)
class LoadedJobs:
    title: str
    jobs: list[PlotJob]
    source_chunk_count: int
    character_reference_count: int


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
    reasoning_effort: ReasoningEffort
    start: int
    limit: int | None
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def safe_filename_part(value: str) -> str:
    sanitized = SAFE_FILENAME_RE.sub("_", value).rstrip(" .")
    return sanitized or "tasks"


def debug_log_path(input_json: Path, job_id: str) -> Path:
    return (
        PROJECT_ROOT
        / "debug"
        / "tasks"
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
        messages: Sequence[BaseMessage],
        parameters: JsonObject,
    ) -> float:
        started_clock = time.perf_counter()
        self.current_section = None
        self._write(
            "\n".join(
                [
                    f"=== ATTEMPT {attempt}/{total_attempts} ===",
                    f"started_at: {now_iso()}",
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
        return started_clock

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
        started_clock: float,
        thinking: str | None,
        output: LLMOutputInfo,
        status: AttemptStatus,
        error: str | None = None,
        parsed_json: JsonObject | None = None,
        validated_result: TaskGenerationResult | None = None,
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
            f"finished_at: {now_iso()}",
            f"duration_ms: {round((time.perf_counter() - started_clock) * 1000)}",
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
    job: PlotJob,
    parameters: JsonObject,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "RLFF task generation debug log",
                "",
                "=== PLOT ===",
                f"source_title: {title}",
                f"plot_index: {job.plot_index}",
                f"job_id: {job.job_id}",
                f"volume: {job.volume or ''}",
                f"chapter: {job.chapter or ''}",
                f"source_chunk_count: {job.source_chunk_count}",
                f"source_token_count: {job.source_token_count}",
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
        raise ValueError(f"{field} fields must be exactly {sorted(expected)}.")


def nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def string_or_none(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null.")
    return value


def load_characters(raw: Any, field: str) -> list[PlotCharacter]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be an array.")
    characters: list[PlotCharacter] = []
    alias_owner: dict[str, int] = {}
    for index, value in enumerate(raw):
        character_field = f"{field}[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{character_field} must be an object.")
        names = value.get("name")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
        ):
            raise ValueError(f"{character_field}.name must be a non-empty string array.")
        clean_names = list(dict.fromkeys(name.strip() for name in cast(list[str], names)))
        description = value.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"{character_field}.description must be a string.")
        character_index = len(characters)
        for name in clean_names:
            if name in alias_owner:
                other = alias_owner[name]
                raise ValueError(
                    f"{character_field}.name alias {name!r} also belongs to characters[{other}]."
                )
            alias_owner[name] = character_index
        characters.append(PlotCharacter(names=clean_names, description=description.strip()))
    return characters


def build_jobs(root: JsonObject, fallback_title: str) -> LoadedJobs:
    raw_plots = root.get("plots")
    if not isinstance(raw_plots, list):
        raise ValueError("Input JSON must contain a plots array.")

    jobs: list[PlotJob] = []
    source_chunk_count = 0
    character_reference_count = 0
    for plot_index, raw_plot in enumerate(raw_plots):
        field = f"plots[{plot_index}]"
        if not isinstance(raw_plot, dict):
            raise ValueError(f"{field} must be an object.")
        characters = load_characters(raw_plot.get("characters"), f"{field}.characters")
        raw_chunks = raw_plot.get("chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ValueError(f"{field}.chunks must be a non-empty array.")
        texts: list[str] = []
        token_count = 0
        for chunk_index, raw_chunk in enumerate(raw_chunks):
            chunk_field = f"{field}.chunks[{chunk_index}]"
            if not isinstance(raw_chunk, dict):
                raise ValueError(f"{chunk_field} must be an object.")
            text = raw_chunk.get("text")
            token = raw_chunk.get("token")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{chunk_field}.text must be a non-empty string.")
            if type(token) is not int or token < 0:
                raise ValueError(f"{chunk_field}.token must be a non-negative integer.")
            texts.append(text)
            token_count += token
        jobs.append(
            PlotJob(
                plot_index=plot_index,
                job_id=f"plot-{plot_index:06d}",
                volume=string_or_none(raw_plot.get("volume"), f"{field}.volume"),
                chapter=string_or_none(raw_plot.get("chapter"), f"{field}.chapter"),
                source_chunk_count=len(raw_chunks),
                source_token_count=token_count,
                characters=characters,
                text="".join(texts),
            )
        )
        source_chunk_count += len(raw_chunks)
        character_reference_count += len(characters)

    title_value = root.get("title")
    title = title_value if isinstance(title_value, str) and title_value else fallback_title
    return LoadedJobs(
        title=title,
        jobs=jobs,
        source_chunk_count=source_chunk_count,
        character_reference_count=character_reference_count,
    )


def load_jobs(path: Path) -> LoadedJobs:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(root, dict):
        raise ValueError("Input JSON must be an object.")
    return build_jobs(root, path.stem)


def build_messages(
    prompt: PromptParts,
    job: PlotJob,
    retry_feedback: str | None = None,
    retry_output: str | None = None,
) -> list[BaseMessage]:
    payload = {
        "input": job.text,
        "characters": [asdict(character) for character in job.characters],
    }
    input_json = json_text(payload, indent=2)
    user_content = f"{prompt.user_prefix}\n{input_json}" if prompt.user_prefix else input_json
    messages: list[BaseMessage] = [
        {"role": "system", "content": prompt.system},
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


def validate_result(raw: JsonObject, characters: Sequence[PlotCharacter]) -> TaskGenerationResult:
    exact_fields(raw, ["plot", "tasks"], "Output")
    plot = nonempty_string(raw.get("plot"), "Output.plot")
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("Output.tasks must be an array.")

    alias_owner = {
        alias: character.names[0] for character in characters for alias in character.names
    }
    used_characters: set[str] = set()
    tasks: list[CharacterTask] = []
    for index, value in enumerate(raw_tasks):
        field = f"Output.tasks[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be an object.")
        exact_fields(value, ["name", "values"], field)
        name = nonempty_string(value.get("name"), f"{field}.name")
        canonical = alias_owner.get(name)
        if canonical is None:
            raise ValueError(f"{field}.name {name!r} is not present in input characters.")
        if canonical in used_characters:
            raise ValueError(f"{field}.name duplicates tasks for character {canonical!r}.")
        used_characters.add(canonical)

        raw_values = value.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"{field}.values must be a non-empty string array.")
        values = [
            nonempty_string(task_value, f"{field}.values[{value_index}]")
            for value_index, task_value in enumerate(raw_values)
        ]
        if len(set(values)) != len(values):
            raise ValueError(f"{field}.values contains duplicate tasks.")
        tasks.append(CharacterTask(name=name, values=values))
    return TaskGenerationResult(plot=plot, tasks=tasks)


def result_dict(result: TaskGenerationResult) -> JsonObject:
    return {
        "plot": result.plot,
        "tasks": [asdict(task) for task in result.tasks],
    }


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
            f"[Error]任务提取结果校验失败：{error}",
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
    job: PlotJob,
    prompt: PromptParts,
    *,
    retries: int,
    retry_delay_ms: int,
    streamed: bool,
    provider: ModelType,
    generation_parameters: JsonObject,
    configured_reasoning_effort: ReasoningEffort,
    debug_logger: DebugLogger | None = None,
) -> TaskGenerationResult:
    total_attempts = retries + 1
    reasoning_effort = configured_reasoning_effort
    retry_feedback: str | None = None
    retry_output: str | None = None

    for attempt in range(total_attempts):
        attempt_number = attempt + 1
        messages = build_messages(prompt, job, retry_feedback, retry_output)
        runtime_parameters = {
            **generation_parameters,
            **({"reasoning_effort": reasoning_effort} if provider == "DeepSeek" else {}),
        }
        await limiter.wait()
        started_clock = (
            debug_logger.start(
                attempt_number,
                total_attempts,
                messages,
                {"stream": streamed, **runtime_parameters},
            )
            if debug_logger is not None
            else time.perf_counter()
        )
        received = False
        thinking = ""
        content = ""
        output = LLMOutputInfo()

        def finish(
            status: AttemptStatus,
            error: Exception | None = None,
            parsed_json: JsonObject | None = None,
            validated_result: TaskGenerationResult | None = None,
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
                started_clock=current_started_clock,
                thinking=current_thinking or None,
                output=current_output,
                status=status,
                error=str(error) if error is not None else None,
                parsed_json=parsed_json,
                validated_result=validated_result,
            )

        try:
            if streamed:
                received = True
                thinking, content, output = await collect_assistant_stream(
                    llm.stream(messages, runtime_parameters),
                    debug_logger,
                )
                if not content:
                    raise ValueError("LLM response contains no assistant text.")
            else:
                response = await llm.invoke(messages, runtime_parameters)
                received = True
                thinking = assistant_thinking(response)
                content = assistant_text(response)
                output = extract_llm_output_info(response, content)
        except Exception as error:
            length = is_length_finish_reason(output.finish_reason)
            status: AttemptStatus = (
                "length" if length else "response_error" if received else "request_error"
            )
            finish(status, error, current_thinking=thinking, current_output=output)
            if (length or is_timeout_error(error)) and attempt < retries:
                if length and provider == "DeepSeek" and reasoning_effort == "max":
                    reasoning_effort = "high"
                await asyncio.sleep(retry_delay_ms * attempt_number / 1000)
                continue
            raise

        if is_length_finish_reason(output.finish_reason):
            error = RuntimeError(f"LLM stopped because finish_reason={output.finish_reason}.")
            finish("length", error, current_thinking=thinking, current_output=output)
            if attempt < retries:
                if provider == "DeepSeek" and reasoning_effort == "max":
                    reasoning_effort = "high"
                await asyncio.sleep(retry_delay_ms * attempt_number / 1000)
                continue
            raise error

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
            result = validate_result(parsed_json, job.characters)
        except ValueError as error:
            retry_error = ValueError(validation_retry_feedback(error, parsed_json))
            finish(
                "validation_error",
                retry_error,
                parsed_json,
                current_thinking=thinking,
                current_output=output,
            )
            if attempt < retries:
                retry_feedback = str(retry_error)
                retry_output = json_text(parsed_json, indent=2)
                await asyncio.sleep(retry_delay_ms * attempt_number / 1000)
                continue
            raise retry_error from error

        finish(
            "success",
            parsed_json=parsed_json,
            validated_result=result,
            current_thinking=thinking,
            current_output=output,
        )
        return result

    raise RuntimeError("Task generation retry loop exhausted.")


def output_record(
    title: str,
    job: PlotJob,
    result: TaskGenerationResult,
    model: str,
) -> JsonObject:
    return {
        "model": model,
        "source_title": title,
        "plot_index": job.plot_index,
        "source_chunk_start": 0,
        "source_chunk_end": job.source_chunk_count - 1,
        "source_chunk_count": job.source_chunk_count,
        "source_token_count": job.source_token_count,
        "volume": job.volume,
        "chapter": job.chapter,
        "result": result_dict(result),
    }


def completed_results(
    path: Path,
    title: str,
    jobs: Sequence[PlotJob],
) -> dict[int, TaskGenerationResult]:
    if not path.exists():
        return {}
    job_by_index = {job.plot_index: job for job in jobs}
    results: dict[int, TaskGenerationResult] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Existing task JSONL line {line_number} is invalid.")
        plot_index = row.get("plot_index")
        if type(plot_index) is not int or plot_index not in job_by_index:
            raise ValueError(f"Existing task JSONL line {line_number} has an unknown plot_index.")
        if row.get("source_title") != title:
            raise ValueError(
                f"Existing task JSONL line {line_number} has a different source title."
            )
        if plot_index in results:
            raise ValueError(f"Existing task JSONL duplicates plot_index {plot_index}.")
        raw_result = row.get("result")
        if not isinstance(raw_result, dict):
            raise ValueError(f"Existing task JSONL line {line_number} has no result object.")
        results[plot_index] = validate_result(raw_result, job_by_index[plot_index].characters)
    return results


def select_jobs(
    jobs: Sequence[PlotJob],
    completed: set[int],
    start: int,
    limit: int | None,
) -> list[PlotJob]:
    pending = [
        job
        for job in jobs
        if job.source_token_count >= MIN_PLOT_TOKENS
        and job.plot_index >= start
        and job.plot_index not in completed
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


def reasoning_effort_value(value: str) -> ReasoningEffort:
    normalized = value.lower()
    if normalized not in {"low", "medium", "high", "max"}:
        raise argparse.ArgumentTypeError("reasoning effort must be one of: low, medium, high, max")
    return cast(ReasoningEffort, normalized)


def parse_args(argv: Sequence[str] | None = None) -> Args:
    configured_model = (
        os.getenv("RLFF_TASK_MODEL")
        or os.getenv("RLFF_PLOT_MODEL")
        or os.getenv("RLFF_VERIFIER_MODEL")
    )
    configured_provider = (
        os.getenv("RLFF_TASK_PROVIDER") or os.getenv("RLFF_PLOT_PROVIDER") or "DeepSeek"
    )
    parser = argparse.ArgumentParser(
        description="Generate detailed plot descriptions and character tasks per rebuilt plot.",
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "task_generation.txt")
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
    parser.add_argument(
        "--reasoning-effort",
        type=reasoning_effort_value,
        default=reasoning_effort_value(os.getenv("RLFF_TASK_REASONING_EFFORT", "max")),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
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
    if not namespace.model:
        if namespace.provider == "DeepSeek":
            namespace.model = "deepseek-v4-pro"
        else:
            parser.error("--model or RLFF_TASK_MODEL is required for GLM and Qwen")
    if namespace.base_url is None:
        namespace.base_url = {
            "DeepSeek": os.getenv("DEEPSEEK_BASE_URL"),
            "GLM": os.getenv("GLM_BASE_URL"),
            "Qwen": os.getenv("DASHSCOPE_BASE_URL"),
        }[namespace.provider]
    return Args(**vars(namespace))


async def run(args: Args) -> None:
    loaded = load_jobs(args.input_json)
    prompt = split_prompt(args.prompt.read_text(encoding="utf-8-sig"))
    eligible_plot_count = sum(job.source_token_count >= MIN_PLOT_TOKENS for job in loaded.jobs)
    results = (
        {} if args.overwrite else completed_results(args.output_jsonl, loaded.title, loaded.jobs)
    )
    jobs = select_jobs(loaded.jobs, set(results), args.start, args.limit)
    if args.overwrite and not args.dry_run and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    summary = {
        "provider": args.provider,
        "model": args.model,
        "source_title": loaded.title,
        "plot_count": len(loaded.jobs),
        "minimum_plot_tokens": MIN_PLOT_TOKENS,
        "eligible_plot_count": eligible_plot_count,
        "skipped_short_plot_count": len(loaded.jobs) - eligible_plot_count,
        "source_chunk_count": loaded.source_chunk_count,
        "character_reference_count": loaded.character_reference_count,
        "selected_plot_count": len(jobs),
        "existing_plot_count": len(results),
        "start": args.start,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "debug": args.debug,
        "stream": args.stream,
    }
    print(json_text(summary, indent=2))
    if args.dry_run:
        if jobs:
            print(json_text(build_messages(prompt, jobs[0]), indent=2))
        return
    if not jobs:
        return

    provider_parameters: JsonObject = (
        {"reasoning_effort": args.reasoning_effort}
        if args.provider == "DeepSeek"
        else {"enable_thinking": True}
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
        "stream": args.stream,
        **({"base_url": args.base_url} if args.base_url else {}),
        **provider_parameters,
    }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[PlotJob] = asyncio.Queue()
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
                        configured_reasoning_effort=args.reasoning_effort,
                        debug_logger=debug_logger,
                    )
                    results[job.plot_index] = result
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
        raise RuntimeError(f"{len(failures)} task generation job(s) failed.")


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
