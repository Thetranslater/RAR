"""Run Magiv3 detection on one or more comic images.

This tool intentionally calls only Magiv3's detection/association API.  It does
not call OCR, captioning, or character-grounding generation.

The public Magiv3 detector returns bounding boxes for panels, text regions,
characters, and speech-bubble tails.  It does not return a segmentation mask
for the speech-bubble body, so ``tails`` should not be interpreted as complete
balloon masks.

Example (run from the repository root on a CUDA server)::

    python src/tools/extract_magiv3_detections.py \
        --image data/manga/images/page.png \
        --output outputs/page.magiv3.json \
        --draw-dir outputs/page.magiv3

    python src/tools/extract_magiv3_detections.py \
        --image-dir data/manga/images \
        --output outputs/manga.magiv3.json

The server is expected to provide the Magiv3 runtime dependencies, including
PyTorch, Transformers, and Pillow.  No project dependency is added for this
standalone server-side tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "models" / "magiv3"
DETECTION_KEYS = ("panels", "texts", "characters", "tails")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Magiv3 detection boxes for panels, text regions, "
            "characters, and speech-bubble tails."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        nargs="+",
        help="One or more image paths. Paths containing spaces must be quoted.",
    )
    input_group.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of images to process recursively, in sorted path order.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local Magiv3 directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "magiv3_detections.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--draw-dir",
        type=Path,
        default=None,
        help="Optional directory for annotated PNGs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="cuda",
        help="Inference device (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Model dtype; auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images passed to Magiv3 per inference call (default: 1).",
    )
    parser.add_argument(
        "--character-threshold",
        type=float,
        default=0.5,
        help="Character-character association threshold.",
    )
    parser.add_argument(
        "--text-character-threshold",
        type=float,
        default=0.8,
        help="Text-character association threshold.",
    )
    parser.add_argument(
        "--text-tail-threshold",
        type=float,
        default=0.8,
        help="Text-tail association threshold.",
    )
    parser.add_argument(
        "--essential-text-threshold",
        type=float,
        default=0.8,
        help="Essential-text threshold.",
    )
    return parser


def resolve_device(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Run this tool on the CUDA server or pass --device cpu."
        )
    return torch_module.device(requested)


def resolve_dtype(torch_module: Any, requested: str, device: Any) -> Any:
    if requested == "auto":
        requested = "float16" if device.type == "cuda" else "float32"
    return {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[requested]


def load_model(model_path: Path, device_name: str, dtype_name: str) -> tuple[Any, Any, Any]:
    """Load the local custom-code model and processor without downloading files."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if not model_path.is_dir():
        raise FileNotFoundError(f"Magiv3 model directory does not exist: {model_path}")

    device = resolve_device(torch, device_name)
    dtype = resolve_dtype(torch, dtype_name, device)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    return model, processor, device


def load_images(image_paths: Iterable[str]) -> tuple[list[Path], list[Any]]:
    from PIL import Image

    paths = [Path(path).expanduser().resolve() for path in image_paths]
    images: list[Any] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return paths, images


def collect_image_paths(image_args: list[str] | None, image_dir: Path | None) -> list[str]:
    if image_dir is None:
        if not image_args:
            raise ValueError("Provide either --image or --image-dir")
        return image_args

    image_dir = image_dir.expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")

    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under: {image_dir}")
    return [str(path) for path in image_paths]


def to_jsonable(value: Any) -> Any:
    """Convert tensors and numpy-like scalars without requiring numpy at import time."""
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
    raise TypeError(f"Cannot serialize Magiv3 output of type {type(value)!r}")


def run_detection(
    model: Any,
    processor: Any,
    image_paths: list[Path],
    images: list[Any],
    batch_size: int,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    records: list[dict[str, Any]] = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        raw_results = model.predict_detections_and_associations(
            batch_images,
            processor,
            character_character_association_threshold=thresholds["character"],
            text_character_association_threshold=thresholds["text_character"],
            text_tail_association_threshold=thresholds["text_tail"],
            essential_text_threshold=thresholds["essential_text"],
        )
        if len(raw_results) != len(batch_images):
            raise RuntimeError(
                "Magiv3 returned a result count different from the input image count: "
                f"{len(raw_results)} != {len(batch_images)}"
            )

        for path, image, raw_result in zip(
            image_paths[start : start + batch_size], batch_images, raw_results
        ):
            result = to_jsonable(raw_result)
            records.append(
                {
                    "image": str(path),
                    "image_size": {"width": image.width, "height": image.height},
                    "detections": {
                        key: result.get(key, []) for key in DETECTION_KEYS
                    },
                    "associations": {
                        key: result.get(key, [])
                        for key in (
                            "character_cluster_labels",
                            "text_character_associations",
                            "text_tail_associations",
                            "is_essential_text",
                        )
                    },
                }
            )
    return records


def draw_record(record: dict[str, Any], image: Any, target: Path) -> None:
    """Write a lightweight visual check without depending on Magiv3's plotting helpers."""
    from PIL import ImageDraw

    colors = {
        "panels": "#20a020",
        "texts": "#e03030",
        "characters": "#2050d0",
        "tails": "#d020c0",
    }
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for category, color in colors.items():
        for index, bbox in enumerate(record["detections"].get(category, [])):
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1 + 2, y1 + 2), f"{category[:-1]}:{index}", fill=color)

    target.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(target)


def write_outputs(
    output_path: Path,
    records: list[dict[str, Any]],
    draw_dir: Path | None,
    images: list[Any],
) -> None:
    payload = {
        "format": "magiv3-detections-v1",
        "note": (
            "Magiv3 returns bounding boxes for speech-bubble tails, not a "
            "complete speech-bubble body segmentation mask."
        ),
        "images": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if draw_dir is not None:
        for index, (record, image) in enumerate(zip(records, images)):
            filename = f"{index:04d}_{Path(record['image']).stem}.png"
            draw_record(record, image, draw_dir / filename)


def main() -> None:
    args = build_parser().parse_args()
    input_paths = collect_image_paths(args.image, args.image_dir)
    image_paths, images = load_images(input_paths)
    model, processor, _device = load_model(args.model.resolve(), args.device, args.dtype)
    records = run_detection(
        model=model,
        processor=processor,
        image_paths=image_paths,
        images=images,
        batch_size=args.batch_size,
        thresholds={
            "character": args.character_threshold,
            "text_character": args.text_character_threshold,
            "text_tail": args.text_tail_threshold,
            "essential_text": args.essential_text_threshold,
        },
    )
    write_outputs(args.output.resolve(), records, args.draw_dir, images)
    print(f"Wrote detections for {len(records)} image(s) to {args.output.resolve()}")
    if args.draw_dir is not None:
        print(f"Wrote annotated images to {args.draw_dir.resolve()}")


if __name__ == "__main__":
    main()
