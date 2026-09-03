"""Segment Magiv3-detected panels with SAM2.

The input is the JSON produced by ``extract_magiv3_detections.py``.  For each
image record, every box in ``detections.panels`` is used as a SAM2 box prompt.
The highest-scoring mask returned by SAM2 is written as a full-image binary
mask, and as an RGBA crop whose alpha channel is the mask.

This tool intentionally ignores character, text, and tail detections.

Example::

    python src/tools/segment_magiv3_panels_with_sam2.py \
        --input outputs/magiv3_detections.json \
        --sam2-checkpoint models/sam2/sam2.1_hiera_large.pt \
        --output-dir outputs/sam2_panels \
        --save-overlays

The server is expected to provide the SAM2 Python package, PyTorch, NumPy, and
Pillow.  This standalone tool does not add project dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"


def default_checkpoint_path() -> Path:
    candidates = (
        REPOSITORY_ROOT / "models" / "sam2" / SAM2_CHECKPOINT_NAME,
        REPOSITORY_ROOT / "src" / "models" / "sam2" / SAM2_CHECKPOINT_NAME,
        Path.cwd() / "models" / "sam2" / SAM2_CHECKPOINT_NAME,
        Path.cwd() / "src" / "models" / "sam2" / SAM2_CHECKPOINT_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use SAM2 box prompts to segment panels from Magiv3 detection JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Magiv3 detection JSON path.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="SAM2.1 Hiera-L checkpoint path.",
    )
    parser.add_argument(
        "--sam2-config",
        default=SAM2_CONFIG,
        help=(
            "SAM2 Hydra config name. Keep this relative when using the official "
            "sam2 package."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "sam2_panels",
        help="Directory for masks, segmented crops, overlays, and manifest JSON.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="SAM2 inference device (default: auto).",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Also save full-image mask overlays for visual inspection.",
    )
    return parser


def load_detection_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Detection JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("Input JSON must contain an 'images' array.")
    return payload["images"]


def resolve_device(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Run on the CUDA server or pass --device cpu."
        )
    return torch_module.device(requested)


def load_predictor(checkpoint: Path, config: str, requested_device: str) -> tuple[Any, Any]:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")

    device = resolve_device(torch, requested_device)
    sam2_model = build_sam2(config, str(checkpoint), device=str(device))
    return SAM2ImagePredictor(sam2_model), device


def as_box(value: Any, image_width: int, image_height: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Panel box must contain four coordinates: {value!r}")
    x1, y1, x2, y2 = (float(item) for item in value)
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Panel box has no positive area after clipping: {value!r}")
    return [x1, y1, x2, y2]


def crop_bounds(box: list[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = max(0, min(image_width, int(x1)))
    top = max(0, min(image_height, int(y1)))
    right = max(left + 1, min(image_width, int(x2 + 0.999999)))
    bottom = max(top + 1, min(image_height, int(y2 + 0.999999)))
    return left, top, right, bottom


def save_panel_outputs(
    image: Any,
    mask: Any,
    box: list[float],
    output_dir: Path,
    image_index: int,
    panel_index: int,
    source_stem: str,
    score: float,
    save_overlay: bool,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    image_array = np.asarray(image.convert("RGB"))
    mask_array = np.asarray(mask).astype(bool)
    if mask_array.shape != image_array.shape[:2]:
        raise RuntimeError(
            "SAM2 returned a mask with a different size from the source image: "
            f"{mask_array.shape} != {image_array.shape[:2]}"
        )

    # Keep the extracted content inside the Magiv3 panel proposal even if the
    # SAM2 mask slightly spills across the box boundary.
    left, top, right, bottom = crop_bounds(box, image_array.shape[1], image_array.shape[0])
    box_region = np.zeros_like(mask_array, dtype=bool)
    box_region[top:bottom, left:right] = True
    mask_array &= box_region

    name = f"image_{image_index:05d}_{source_stem}_panel_{panel_index:03d}"
    mask_path = output_dir / "masks" / f"{name}.png"
    segment_path = output_dir / "segments" / f"{name}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    segment_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray((mask_array * 255).astype(np.uint8), mode="L").save(mask_path)

    crop = image_array[top:bottom, left:right]
    crop_mask = mask_array[top:bottom, left:right]
    alpha = (crop_mask * 255).astype(np.uint8)
    rgba_crop = np.concatenate([crop, alpha[..., None]], axis=-1)
    Image.fromarray(rgba_crop, mode="RGBA").save(segment_path)

    overlay_path: Path | None = None
    if save_overlay:
        overlay_path = output_dir / "overlays" / f"{name}.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = image_array.copy()
        overlay[mask_array] = (
            0.65 * overlay[mask_array] + 0.35 * np.array([40, 180, 255])
        ).astype(np.uint8)
        Image.fromarray(overlay, mode="RGB").save(overlay_path)

    return {
        "panel_index": panel_index,
        "box": box,
        "mask_score": score,
        "mask_path": str(mask_path),
        "segment_path": str(segment_path),
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
        "crop_box": [left, top, right, bottom],
    }


def process_images(
    image_records: list[dict[str, Any]],
    predictor: Any,
    output_dir: Path,
    save_overlay: bool,
) -> list[dict[str, Any]]:
    from PIL import Image

    manifest: list[dict[str, Any]] = []
    for image_index, record in enumerate(image_records):
        image_path_value = record.get("image")
        if not isinstance(image_path_value, str):
            raise ValueError(f"images[{image_index}].image must be a string")
        image_path = Path(image_path_value).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image does not exist: {image_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        image_width, image_height = image.size
        panels = record.get("detections", {}).get("panels", [])
        if not isinstance(panels, list):
            raise ValueError(f"images[{image_index}].detections.panels must be an array")

        predictor.set_image(image)
        image_result: dict[str, Any] = {
            "image": str(image_path.resolve()),
            "image_size": {"width": image_width, "height": image_height},
            "panels": [],
        }
        source_stem = image_path.stem.replace(" ", "_")

        for panel_index, raw_box in enumerate(panels):
            box = as_box(raw_box, image_width, image_height)
            masks, scores, _logits = predictor.predict(
                box=box,
                multimask_output=True,
            )
            if len(masks) == 0 or len(scores) == 0:
                raise RuntimeError(
                    f"SAM2 returned no mask for image {image_index}, panel {panel_index}"
                )
            best_index = int(scores.argmax())
            panel_result = save_panel_outputs(
                image=image,
                mask=masks[best_index],
                box=box,
                output_dir=output_dir,
                image_index=image_index,
                panel_index=panel_index,
                source_stem=source_stem,
                score=float(scores[best_index]),
                save_overlay=save_overlay,
            )
            image_result["panels"].append(panel_result)

        manifest.append(image_result)
        print(
            f"[{image_index + 1}/{len(image_records)}] {image_path.name}: "
            f"{len(panels)} panel(s)"
        )
    return manifest


def write_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sam2_panel_segments.json"
    payload = {
        "format": "sam2-panel-segments-v1",
        "source": "Magiv3 detections, detections.panels",
        "images": manifest,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()
    image_records = load_detection_json(args.input.expanduser().resolve())
    predictor, _device = load_predictor(
        checkpoint=args.sam2_checkpoint,
        config=args.sam2_config,
        requested_device=args.device,
    )
    manifest = process_images(
        image_records=image_records,
        predictor=predictor,
        output_dir=args.output_dir.resolve(),
        save_overlay=args.save_overlays,
    )
    manifest_path = write_manifest(args.output_dir.resolve(), manifest)
    print(f"Wrote SAM2 panel manifest to {manifest_path}")


if __name__ == "__main__":
    main()
