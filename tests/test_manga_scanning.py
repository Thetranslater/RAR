from pathlib import Path

import pytest
from PIL import Image

from rar_agent.workflow.manga.scanning import MangaScanError, scan_manga_folder


def _image(path: Path, *, color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def test_scan_manga_folder_naturally_sorts_pages_and_keeps_batches_in_directories(
    tmp_path: Path,
) -> None:
    _image(tmp_path / "manga" / "chapter-2" / "10.jpg")
    _image(tmp_path / "manga" / "chapter-2" / "2.jpg")
    _image(tmp_path / "manga" / "chapter-10" / "1.png")
    _image(tmp_path / "manga" / ".preview" / "1.jpg")
    _image(tmp_path / "manga" / "__MACOSX" / "2.jpg")
    (tmp_path / "manga" / "chapter-2" / "notes.txt").write_text("ignore")

    scan = scan_manga_folder(tmp_path, "manga", batch_size=2)

    assert [page.path for page in scan.pages] == [
        "manga/chapter-2/2.jpg",
        "manga/chapter-2/10.jpg",
        "manga/chapter-10/1.png",
    ]
    assert [page.page_index for page in scan.pages] == [0, 1, 2]
    assert [page.meta for page in scan.pages] == [
        {"directory": "chapter-2"},
        {"directory": "chapter-2"},
        {"directory": "chapter-10"},
    ]
    assert [batch.model_dump() for batch in scan.batches] == [
        {"batch_index": 0, "page_indexes": [0, 1]},
        {"batch_index": 1, "page_indexes": [2]},
    ]


def test_scan_manga_folder_rejects_corrupt_supported_images(tmp_path: Path) -> None:
    corrupt = tmp_path / "manga" / "1.webp"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"not an image")

    with pytest.raises(MangaScanError, match="cannot be decoded"):
        scan_manga_folder(tmp_path, "manga")


def test_scan_manga_folder_rejects_empty_inputs_and_project_escape(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(MangaScanError, match="no supported images"):
        scan_manga_folder(tmp_path, "empty")
    with pytest.raises(MangaScanError, match="workspace-relative"):
        scan_manga_folder(tmp_path, "../outside")


def test_scan_manga_folder_can_limit_debug_to_five_batches(tmp_path: Path) -> None:
    for index in range(12):
        _image(tmp_path / "manga" / f"{index}.jpg")

    scan = scan_manga_folder(tmp_path, "manga", batch_size=2, max_batches=5)

    assert len(scan.batches) == 5
    assert [page.page_index for page in scan.pages] == list(range(10))
