"""Refresh the self-contained RLFF cloud-transfer bundle.

The script copies only reproducible project inputs. It never touches model weights,
secrets, or training outputs already placed in the deployment directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = PROJECT_ROOT / "deploy" / "rlff_phase_d"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "episodes" / "rlff_200" / "episodes.jsonl"


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required bundle source does not exist: {source}")
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"required bundle source directory does not exist: {source}")
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".ruff_cache",
            "test_build_episode_dataset.py",
        ),
    )


def sync_bundle(bundle: Path, dataset: Path) -> None:
    bundle.mkdir(parents=True, exist_ok=True)

    _copy_tree(PROJECT_ROOT / "src" / "rlff", bundle / "src" / "rlff")
    _copy_file(
        PROJECT_ROOT / "src" / "script" / "__init__.py",
        bundle / "src" / "script" / "__init__.py",
    )
    _copy_file(
        PROJECT_ROOT / "src" / "script" / "system_prompt_renderer.py",
        bundle / "src" / "script" / "system_prompt_renderer.py",
    )
    _copy_tree(PROJECT_ROOT / "tests" / "tests_rlff", bundle / "tests" / "tests_rlff")

    for filename in ("pyproject.toml", "RLFF.md"):
        _copy_file(PROJECT_ROOT / filename, bundle / filename)
    for filename in (
        "sft_system_v1.txt",
        "sft_system_v2.txt",
        "sft_system_v3.txt",
        "completion_reward_system.txt",
        "completion_reward_system_v2.txt",
        "trajectory_reward_system.txt",
    ):
        _copy_file(PROJECT_ROOT / filename, bundle / "prompts" / filename)

    _copy_file(dataset, bundle / "data" / "episodes.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sync_bundle(args.bundle.resolve(), args.dataset.resolve())
    print(f"Cloud bundle refreshed: {args.bundle.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
