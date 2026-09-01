"""Join LLaMA-Factory predictions with the RLFF evaluation mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} must contain a JSON object")
        rows.append(value)
    return rows


def merge(predictions_path: Path, mapping_path: Path, output_path: Path) -> int:
    predictions = _read_jsonl(predictions_path)
    mappings = _read_jsonl(mapping_path)
    if len(predictions) != len(mappings):
        raise ValueError(
            f"Prediction/mapping length mismatch: {len(predictions)} != {len(mappings)}"
        )

    output: list[dict[str, Any]] = []
    label_mismatches: list[int] = []
    for line_number, (prediction, mapping) in enumerate(
        zip(predictions, mappings, strict=True), 1
    ):
        predicted = prediction.get("predict")
        if not isinstance(predicted, str) or not predicted.strip():
            raise ValueError(
                f"{predictions_path} line {line_number} has no non-empty predict field"
            )
        label = prediction.get("label")
        reference = mapping.get("reference_response")
        if (
            isinstance(label, str)
            and isinstance(reference, str)
            and label.strip() != reference.strip()
        ):
            label_mismatches.append(line_number)
        validation = mapping.get("validation")
        history = mapping.get("history")
        if not isinstance(validation, list) or not validation:
            raise ValueError(f"{mapping_path} line {line_number} has no validation facts")
        if not isinstance(history, str):
            raise ValueError(f"{mapping_path} line {line_number} has invalid history")
        output.append(
            {
                "sample_id": mapping.get("sample_id"),
                "title": mapping.get("title"),
                "plot_index": mapping.get("plot_index"),
                "message_index": mapping.get("message_index"),
                "character": mapping.get("character"),
                "states": validation,
                "history": history,
                "input": predicted,
                "reference_response": reference,
            }
        )

    if label_mismatches:
        preview = ", ".join(map(str, label_mismatches[:10]))
        raise ValueError(
            "LLaMA-Factory labels do not align with the mapping at lines "
            f"{preview}. Check that predictions use the same evaluation.jsonl order."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in output
        ),
        encoding="utf-8",
    )
    return len(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="LLaMA-Factory generated_predictions.jsonl")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("dataset/evaluation/roleplay_reward_model/rlff_mapping.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/evaluation/roleplay_reward_model/reward_inputs.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = merge(args.predictions.resolve(), args.mapping.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "count": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
