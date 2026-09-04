"""Optional PaddleOCR capability probe and shared spawned worker lifecycle."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rar_agent.workflow.manga.models import ImagePage, OcrPageResult
from rar_agent.workflow.manga.ocr_worker import recognize_batch


class OcrCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    cuda: bool
    model_dir: str
    reason: str | None = None


def default_ocr_model_dir(project_root: Path) -> Path:
    override = os.environ.get("RAR_PADDLEOCR_VL_MODEL_DIR")
    return Path(override).expanduser().resolve() if override else (
        project_root / "models" / "PaddleOCR-VL-1.6"
    ).resolve()


def probe_paddleocr(project_root: Path, *, timeout: float = 30.0) -> OcrCapability:
    model_dir = default_ocr_model_dir(project_root)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "rar_agent.workflow.manga.ocr_worker",
                "--probe",
                str(model_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode != 0 or not lines:
            reason = completed.stderr.strip() or "OCR probe did not return a result"
            return OcrCapability(
                available=False,
                cuda=False,
                model_dir=str(model_dir),
                reason=reason,
            )
        return OcrCapability.model_validate(json.loads(lines[-1]))
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as error:
        return OcrCapability(
            available=False,
            cuda=False,
            model_dir=str(model_dir),
            reason=f"OCR probe failed: {error}",
        )


class PaddleOcrWorkerManager:
    """Share one single-process PaddleOCR executor across active Workflows."""

    def __init__(
        self,
        capability: OcrCapability,
        *,
        idle_seconds: float = 60.0,
    ) -> None:
        self.capability = capability
        self.idle_seconds = idle_seconds
        self._executor: ProcessPoolExecutor | None = None
        self._guard = asyncio.Lock()
        self._active_workflows = 0
        self._idle_task: asyncio.Task[None] | None = None

    async def recognize(
        self,
        project_root: Path,
        pages: list[ImagePage],
        *,
        batch_size: int,
    ) -> list[OcrPageResult]:
        if not self.capability.available:
            raise RuntimeError(self.capability.reason or "OCR is unavailable")
        if batch_size < 1:
            raise ValueError("OCR batch_size must be positive")
        await self._enter()
        try:
            results: list[OcrPageResult] = []
            for offset in range(0, len(pages), batch_size):
                batch = pages[offset : offset + batch_size]
                payload = [
                    (
                        page.page_index,
                        str((project_root / page.path).resolve()),
                    )
                    for page in batch
                ]
                try:
                    raw = await self._submit(payload)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    results.extend(
                        OcrPageResult(
                            page_index=page.page_index,
                            blocks=[],
                            error=str(error),
                        )
                        for page in batch
                    )
                    continue
                results.extend(OcrPageResult.model_validate(value) for value in raw)
            return sorted(results, key=lambda value: value.page_index)
        finally:
            await self._leave()

    async def close(self) -> None:
        async with self._guard:
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None
            self._shutdown_executor()

    async def _submit(self, payload: list[tuple[int, str]]) -> list[dict[str, Any]]:
        for attempt in range(2):
            executor = await self._ensure_executor()
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(
                    executor,
                    recognize_batch,
                    self.capability.model_dir,
                    payload,
                )
            except BrokenProcessPool:
                async with self._guard:
                    self._shutdown_executor()
                if attempt:
                    raise
        raise RuntimeError("OCR worker failed")

    async def _ensure_executor(self) -> ProcessPoolExecutor:
        async with self._guard:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            return self._executor

    async def _enter(self) -> None:
        async with self._guard:
            self._active_workflows += 1
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None

    async def _leave(self) -> None:
        async with self._guard:
            self._active_workflows -= 1
            if self._active_workflows == 0:
                self._idle_task = asyncio.create_task(self._idle_shutdown())

    async def _idle_shutdown(self) -> None:
        try:
            await asyncio.sleep(self.idle_seconds)
            async with self._guard:
                if self._active_workflows == 0:
                    self._shutdown_executor()
                self._idle_task = None
        except asyncio.CancelledError:
            return

    def _shutdown_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
