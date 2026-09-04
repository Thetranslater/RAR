"""Spawned PaddleOCR-VL worker implementation and lightweight probe entrypoint."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_PIPELINE: Any | None = None
_PIPELINE_MODEL_DIR: str | None = None


def recognize_batch(model_dir: str, pages: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """Run one page batch inside the persistent spawned process."""
    pipeline = _pipeline(model_dir)
    raw_output = pipeline.predict([path for _, path in pages])
    if isinstance(raw_output, (dict, list)) or hasattr(raw_output, "json"):
        results = raw_output if isinstance(raw_output, list) else [raw_output]
    else:
        results = list(raw_output)
    if len(results) != len(pages):
        raise RuntimeError(
            "PaddleOCR-VL returned a result count different from input count"
        )
    records: list[dict[str, Any]] = []
    for (page_index, _), result in zip(pages, results, strict=True):
        raw = _result_dict(result)
        records.append(
            {
                "page_index": page_index,
                "blocks": [
                    {"text": item["text"], "bbox": _flat_bbox(item.get("bbox"))}
                    for item in _text_items(raw)
                    if str(item.get("text", "")).strip()
                ],
                "error": None,
            }
        )
    return records


def probe(model_dir: str) -> dict[str, Any]:
    path = Path(model_dir)
    if not path.is_dir():
        return {
            "available": False,
            "cuda": False,
            "model_dir": str(path),
            "reason": "PaddleOCR-VL model directory does not exist",
        }
    try:
        import paddle  # type: ignore[import-not-found]
        from paddleocr import PaddleOCRVL  # type: ignore[import-not-found]  # noqa: F401
    except Exception as error:
        return {
            "available": False,
            "cuda": False,
            "model_dir": str(path),
            "reason": f"OCR dependencies are unavailable: {error}",
        }
    try:
        cuda = bool(paddle.device.is_compiled_with_cuda()) and bool(
            paddle.device.cuda.device_count()
        )
    except Exception:
        cuda = False
    return {
        "available": cuda,
        "cuda": cuda,
        "model_dir": str(path),
        "reason": None if cuda else "CUDA is unavailable for PaddleOCR-VL",
    }


def _pipeline(model_dir: str) -> Any:
    global _PIPELINE, _PIPELINE_MODEL_DIR
    if _PIPELINE is not None and model_dir == _PIPELINE_MODEL_DIR:
        return _PIPELINE
    from paddleocr import PaddleOCRVL

    _PIPELINE = PaddleOCRVL(
        vl_rec_model_dir=model_dir,
        device="gpu",
        pipeline_version="v1.6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=True,
        use_ocr_for_image_block=True,
    )
    _PIPELINE_MODEL_DIR = model_dir
    return _PIPELINE


def _result_dict(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if value is None:
        to_dict = getattr(result, "to_dict", None)
        value = to_dict() if callable(to_dict) else None
    if isinstance(value, str):
        value = json.loads(value)
    value = _jsonable(value)
    if not isinstance(value, dict):
        raise TypeError("PaddleOCR-VL result is not an object")
    wrapped = value.get("res")
    return wrapped if len(value) == 1 and isinstance(wrapped, dict) else value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dicts(child)


def _text_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    parsing = result.get("parsing_res_list")
    if isinstance(parsing, list):
        values = []
        for block in parsing:
            if not isinstance(block, dict):
                continue
            text = _first(block, "block_content", "content", "text")
            if text:
                values.append(
                    {
                        "text": text,
                        "bbox": _first(block, "block_bbox", "bbox", "coordinate"),
                    }
                )
        if values:
            return values
    values = []
    for payload in _dicts(result):
        texts = payload.get("rec_texts")
        if not isinstance(texts, list):
            continue
        boxes = _first(payload, "rec_boxes", "rec_polys", "dt_polys") or []
        for index, text in enumerate(texts):
            values.append(
                {
                    "text": text,
                    "bbox": boxes[index] if index < len(boxes) else None,
                }
            )
    return values


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _flat_bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    numbers: list[float] = []

    def collect(item: Any) -> None:
        if isinstance(item, (int, float)):
            numbers.append(float(item))
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    return numbers or None


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        print(json.dumps(probe(sys.argv[2]), ensure_ascii=False))
        return
    raise SystemExit("usage: python -m rar_agent.workflow.manga.ocr_worker --probe MODEL_DIR")


if __name__ == "__main__":
    main()
