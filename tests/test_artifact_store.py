from pathlib import Path

from rar_agent.storage.artifacts import DatasetArtifactStore


def test_dataset_store_creates_fixed_layout_without_overwriting(tmp_path: Path) -> None:
    first = DatasetArtifactStore.create(tmp_path, "My Book")
    second = DatasetArtifactStore.create(tmp_path, "My Book")

    assert first.root.relative_to(tmp_path).as_posix() == "datasets/my-book"
    assert second.root.relative_to(tmp_path).as_posix() == "datasets/my-book-2"
    assert first.paths.work.is_dir()
    assert first.paths.characters.is_dir()
    assert first.paths.exports.is_dir()
    assert first.paths.report.is_dir()


def test_stage_recovery_only_returns_missing_or_empty_results(tmp_path: Path) -> None:
    store = DatasetArtifactStore.create(tmp_path, "Novel")
    expected_inputs = [
        {"path": "work/text_chunks.jsonl", "index": 0},
        {"path": "work/text_chunks.jsonl", "index": 1},
        {"path": "work/text_chunks.jsonl", "index": 2},
    ]
    store.write_jsonl(
        store.paths.plot_extractions,
        [
            {
                "input": expected_inputs[0],
                "result": {"plots": [], "characters": [], "state": "finished"},
            },
            {"input": expected_inputs[1], "result": {}},
        ],
    )

    assert store.pending_unit_indexes(store.paths.plot_extractions, expected_inputs) == [1, 2]
