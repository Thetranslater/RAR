import subprocess
from pathlib import Path

import pytest

from rar_agent.workflow.manga.models import ImagePage
from rar_agent.workflow.manga.ocr_runtime import (
    OcrCapability,
    PaddleOcrWorkerManager,
    probe_paddleocr,
)


def test_probe_paddleocr_uses_isolated_probe_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "ocr-model"
    model_dir.mkdir()
    monkeypatch.setenv("RAR_PADDLEOCR_VL_MODEL_DIR", str(model_dir))
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"available":true,"cuda":true,"model_dir":"'
                + str(model_dir).replace("\\", "\\\\")
                + '","reason":null}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    result = probe_paddleocr(tmp_path)

    assert result.available is True
    assert calls[0][-2:] == ["--probe", str(model_dir)]


async def test_unavailable_ocr_manager_rejects_without_starting_worker(
    tmp_path: Path,
) -> None:
    manager = PaddleOcrWorkerManager(
        OcrCapability(
            available=False,
            cuda=False,
            model_dir=str(tmp_path / "missing"),
            reason="CUDA is unavailable",
        )
    )
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        await manager.recognize(
            tmp_path,
            [ImagePage(page_index=0, path="page.jpg")],
            batch_size=4,
        )
