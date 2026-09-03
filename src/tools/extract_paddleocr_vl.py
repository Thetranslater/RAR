"""Extract structured OCR results from manga images with PaddleOCR-VL.

The tool uses PaddleOCR-VL's complete local pipeline: layout analysis followed
by visual-language recognition.  It does not call a separate story or
character-understanding model.  The raw PaddleOCR-VL result is preserved, and
common text/layout fields are also normalized for downstream comparison with
another VLM.

The PaddleOCR runtime is intentionally imported only after argument parsing,
so this script can be copied to a CUDA server and checked locally without
installing PaddlePaddle or PaddleOCR in the repository environment.

Examples (run from the repository root)::

    python src/tools/extract_paddleocr_vl.py \
        --model-dir models/paddleOCR_VL_for_manga \
        --image "data/manga/page.png" \
        --output outputs/paddleocr_vl_results.json

    python src/tools/extract_paddleocr_vl.py \
        --model-dir models/paddleOCR_VL_for_manga \
        --image-dir "data/manga" \
        --output outputs/paddleocr_vl_results.json

The server is expected to provide PaddlePaddle, PaddleOCR, Pillow, and the
model files.  This standalone tool does not add project dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "models" / "paddleOCR_VL_for_manga"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "paddleocr_vl_results.json"
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR-VL on manga images and save raw structured results, "
            "layout boxes, and normalized text items."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        nargs="+",
        help="One or more image paths. Quote paths containing spaces.",
    )
    input_group.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of images to process recursively, in sorted path order.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Local PaddleOCR-VL multimodal model directory, passed as "
            f"vl_rec_model_dir (default: {DEFAULT_MODEL_DIR})."
        ),
    )
    parser.add_argument(
        "--layout-model-dir",
        type=Path,
        default=None,
        help=(
            "Optional local layout model directory, passed as "
            "layout_detection_model_dir. If omitted, PaddleOCR may download "
            "the official layout model."
        ),
    )
    parser.add_argument(
        "--pipeline-version",
        choices=("v1", "v1.5", "v1.6"),
        default=None,
        help=(
            "PaddleOCR-VL pipeline version. Omit to use the installed "
            "PaddleOCR default; use v1/v1.5 when the fine-tuned model is from "
            "that pipeline version."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("paddle_static", "paddle_dynamic", "transformers"),
        default=None,
        help=(
            "Inference engine. Omit for PaddleOCR's default local engine; "
            "the transformers option is only for a Transformers-compatible model."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Paddle device string, for example gpu:0 or cpu (default: cpu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Consolidated JSON output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--save-visualizations",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Optional directory passed to each Result.save_to_img() call for "
            "PaddleOCR intermediate visualizations."
        ),
    )
    parser.add_argument(
        "--no-layout-detection",
        action="store_true",
        help="Disable layout analysis and use the VL recognition prompt directly.",
    )
    parser.add_argument(
        "--no-ocr-for-image-block",
        action="store_true",
        help="Do not run OCR inside layout regions classified as image blocks.",
    )
    return parser


def collect_image_paths(
    image_args: list[str] | None, image_dir: Path | None
) -> list[Path]:
    if image_dir is not None:
        resolved_dir = image_dir.expanduser().resolve()
        if not resolved_dir.is_dir():
            raise NotADirectoryError(f"Image directory does not exist: {resolved_dir}")
        paths = sorted(
            path
            for path in resolved_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        if not image_args:
            raise ValueError("Provide either --image or --image-dir")
        paths = [Path(path).expanduser().resolve() for path in image_args]

    if not paths:
        raise FileNotFoundError("No supported images were found")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Image does not exist: {missing[0]}")
    return paths


def load_pipeline(args: argparse.Namespace) -> Any:
    """Construct PaddleOCR-VL using the local fine-tuned VL model."""
    from paddleocr import PaddleOCRVL

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"PaddleOCR-VL model directory does not exist: {model_dir}")

    kwargs: dict[str, Any] = {
        "vl_rec_model_dir": str(model_dir),
        "device": args.device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": not args.no_layout_detection,
        "use_ocr_for_image_block": not args.no_ocr_for_image_block,
    }
    if args.layout_model_dir is not None:
        layout_dir = args.layout_model_dir.expanduser().resolve()
        if not layout_dir.is_dir():
            raise FileNotFoundError(
                f"Layout model directory does not exist: {layout_dir}"
            )
        kwargs["layout_detection_model_dir"] = str(layout_dir)
    if args.pipeline_version is not None:
        kwargs["pipeline_version"] = args.pipeline_version
    if args.engine is not None:
        kwargs["engine"] = args.engine

    return PaddleOCRVL(**kwargs)


def to_jsonable(value: Any) -> Any:
    """Convert Paddle/Numpy values into values accepted by json.dumps."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        return to_jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    raise TypeError(f"Cannot serialize PaddleOCR-VL value of type {type(value)!r}")


def result_to_dict(result: Any) -> dict[str, Any]:
    """Read the official Result.json property without relying on print output."""
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if value is None:
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
    if value is None:
        raise TypeError(
            "PaddleOCR-VL returned a result without a json property or to_dict() method"
        )
    if isinstance(value, str):
        value = json.loads(value)
    value = to_jsonable(value)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary from Result.json, got {type(value)!r}")

    # Some PaddleX display paths wrap the payload in {"res": {...}}. Keep the
    # actual structured result as the top-level raw result when that happens.
    wrapped = value.get("res")
    if len(value) == 1 and isinstance(wrapped, dict):
        return wrapped
    return value


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    """Yield nested dictionaries, useful for version-tolerant OCR extraction."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def normalize_parsing_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize PaddleOCR-VL parsing_res_list into text-like records."""
    parsing = result.get("parsing_res_list")
    if not isinstance(parsing, list):
        return []

    items: list[dict[str, Any]] = []
    for index, block in enumerate(parsing):
        if not isinstance(block, dict):
            continue
        content = first_present(block, ("block_content", "content", "text"))
        if content is None or content == "":
            continue
        items.append(
            {
                "index": index,
                "text": content,
                "label": first_present(block, ("block_label", "label")),
                "bbox": first_present(block, ("block_bbox", "bbox", "coordinate")),
                "block_id": first_present(block, ("block_id", "id")),
                "block_order": first_present(block, ("block_order", "order")),
                "score": first_present(block, ("score", "confidence")),
            }
        )
    return items


def normalize_standard_ocr_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize classic PaddleOCR rec_texts/rec_scores/rec_boxes fields."""
    items: list[dict[str, Any]] = []
    for payload in iter_dicts(result):
        texts = payload.get("rec_texts")
        if not isinstance(texts, list):
            continue
        scores = payload.get("rec_scores") or []
        boxes = first_present(payload, ("rec_boxes", "rec_polys", "dt_polys")) or []
        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else None
            bbox = boxes[index] if index < len(boxes) else None
            items.append(
                {
                    "index": index,
                    "text": text,
                    "label": "text",
                    "bbox": bbox,
                    "score": score,
                }
            )
    return items


def normalize_text_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    parsing_items = normalize_parsing_items(result)
    if parsing_items:
        return parsing_items
    return normalize_standard_ocr_items(result)


def normalize_layout_boxes(result: dict[str, Any]) -> list[dict[str, Any]]:
    layout_result = result.get("layout_det_res")
    if not isinstance(layout_result, dict):
        return []
    boxes = layout_result.get("boxes")
    if not isinstance(boxes, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        if not isinstance(box, dict):
            continue
        normalized.append(
            {
                "index": index,
                "label": box.get("label"),
                "score": box.get("score"),
                "bbox": first_present(box, ("coordinate", "bbox", "box")),
                "cls_id": box.get("cls_id"),
            }
        )
    return normalized


def run_ocr(
    pipeline: Any,
    image_paths: list[Path],
    visualization_dir: Path | None,
) -> list[dict[str, Any]]:
    # PaddleOCR-VL officially accepts a list of image paths and returns one
    # Result object per image. This also avoids reinitializing the pipeline.
    raw_output = pipeline.predict([str(path) for path in image_paths])
    if isinstance(raw_output, (dict, list)) or hasattr(raw_output, "json"):
        results = [raw_output] if not isinstance(raw_output, list) else raw_output
    else:
        results = list(raw_output)
    if len(results) != len(image_paths):
        raise RuntimeError(
            "PaddleOCR-VL returned a result count different from the input count: "
            f"{len(results)} != {len(image_paths)}"
        )

    if visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, (image_path, result) in enumerate(zip(image_paths, results)):
        raw_result = result_to_dict(result)
        record = {
            "image": str(image_path),
            "result_index": index,
            "image_size": {
                "width": raw_result.get("width"),
                "height": raw_result.get("height"),
            },
            "text_items": normalize_text_items(raw_result),
            "layout_detections": normalize_layout_boxes(raw_result),
            "raw_result": raw_result,
        }
        records.append(record)

        if visualization_dir is not None:
            result.save_to_img(save_path=str(visualization_dir))

        print(
            f"[{index + 1}/{len(image_paths)}] {image_path.name}: "
            f"{len(record['text_items'])} text item(s), "
            f"{len(record['layout_detections'])} layout box(es)"
        )
    return records


def write_output(
    output_path: Path,
    records: list[dict[str, Any]],
    model_dir: Path,
    device: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "paddleocr-vl-results-v1",
        "model_dir": str(model_dir),
        "device": device,
        "images": records,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    image_paths = collect_image_paths(args.image, args.image_dir)
    pipeline = load_pipeline(args)
    records = run_ocr(
        pipeline=pipeline,
        image_paths=image_paths,
        visualization_dir=(
            args.save_visualizations.expanduser().resolve()
            if args.save_visualizations is not None
            else None
        ),
    )
    output_path = args.output.expanduser().resolve()
    write_output(
        output_path=output_path,
        records=records,
        model_dir=args.model_dir.expanduser().resolve(),
        device=args.device,
    )
    print(f"Wrote PaddleOCR-VL results for {len(records)} image(s) to {output_path}")


if __name__ == "__main__":
    main()
