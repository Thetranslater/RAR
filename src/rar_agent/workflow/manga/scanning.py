"""Discover, validate and naturally order one local manga image tree."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath, PureWindowsPath

from PIL import Image, UnidentifiedImageError

from rar_agent.workflow.manga.models import (
    ImageBatch,
    ImagePage,
    MangaPreview,
    MangaPreviewIssue,
    MangaScan,
)

SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_DIGITS = re.compile(r"(\d+)")


class MangaScanError(ValueError):
    pass


def scan_manga_folder(
    project_root: Path,
    resource_path: str,
    *,
    batch_size: int = 5,
    max_batches: int | None = None,
) -> MangaScan:
    if not 1 <= batch_size <= 10:
        raise MangaScanError("batch_size must be between 1 and 10")
    if max_batches is not None and max_batches < 1:
        raise MangaScanError("max_batches must be positive")

    normalized = _relative_path(resource_path)
    project = project_root.resolve()
    resource = (project / normalized).resolve()
    try:
        resource.relative_to(project)
    except ValueError as error:
        raise MangaScanError("resource path must be workspace-relative") from error
    if not resource.is_dir():
        raise MangaScanError(f"manga resource is not a directory: {normalized}")

    image_paths = _discover_images(resource)
    if not image_paths:
        raise MangaScanError("manga folder contains no supported images")
    image_paths.sort(key=lambda value: _natural_key(value.relative_to(resource)))
    for image_path in image_paths:
        _validate_image(image_path, project)

    all_pages = [
        ImagePage(
            page_index=index,
            path=image_path.relative_to(project).as_posix(),
            meta={"directory": _directory_meta(image_path.parent, resource)},
        )
        for index, image_path in enumerate(image_paths)
    ]
    all_batches = _build_batches(all_pages, batch_size)
    if max_batches is not None:
        all_batches = all_batches[:max_batches]
        kept = {index for batch in all_batches for index in batch.page_indexes}
        all_pages = [page for page in all_pages if page.page_index in kept]

    return MangaScan(
        resource_path=normalized,
        pages=all_pages,
        batches=all_batches,
    )


def preview_manga_folder(
    project_root: Path,
    resource_path: str,
    *,
    batch_size: int = 5,
) -> MangaPreview:
    if not 1 <= batch_size <= 10:
        raise MangaScanError("batch_size must be between 1 and 10")
    normalized = _relative_path(resource_path)
    project = project_root.resolve()
    resource = (project / normalized).resolve()
    try:
        resource.relative_to(project)
    except ValueError as error:
        raise MangaScanError("resource path must be workspace-relative") from error
    if not resource.is_dir():
        raise MangaScanError(f"manga resource is not a directory: {normalized}")

    skipped: list[MangaPreviewIssue] = []
    image_paths = _discover_images(resource, skipped=skipped, project_root=project)
    image_paths.sort(key=lambda value: _natural_key(value.relative_to(resource)))
    valid_paths: list[Path] = []
    errors: list[MangaPreviewIssue] = []
    for image_path in image_paths:
        try:
            _validate_image(image_path, project)
        except MangaScanError as error:
            errors.append(
                MangaPreviewIssue(
                    path=image_path.relative_to(project).as_posix(),
                    reason=str(error),
                )
            )
        else:
            valid_paths.append(image_path)
    pages = [
        ImagePage(
            page_index=index,
            path=image_path.relative_to(project).as_posix(),
            meta={"directory": _directory_meta(image_path.parent, resource)},
        )
        for index, image_path in enumerate(valid_paths)
    ]
    return MangaPreview(
        path=normalized,
        image_count=len(pages),
        batch_count=len(_build_batches(pages, batch_size)),
        first_paths=[page.path for page in pages[:10]],
        skipped=skipped,
        errors=errors,
    )


def _relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if (
        not normalized
        or PurePath(normalized).is_absolute()
        or PureWindowsPath(normalized).is_absolute()
        or ".." in PurePath(normalized).parts
    ):
        raise MangaScanError("resource path must be workspace-relative")
    return normalized


def _discover_images(
    resource: Path,
    *,
    skipped: list[MangaPreviewIssue] | None = None,
    project_root: Path | None = None,
) -> list[Path]:
    values: list[Path] = []
    for root, directories, filenames in os.walk(resource, followlinks=False):
        root_path = Path(root)
        kept_directories: list[str] = []
        for name in directories:
            candidate = root_path / name
            reason = None
            if name.startswith(".") or name == "__MACOSX":
                reason = "hidden or metadata directory"
            elif candidate.is_symlink():
                reason = "symbolic link"
            if reason is None:
                kept_directories.append(name)
            elif skipped is not None and project_root is not None:
                skipped.append(
                    MangaPreviewIssue(
                        path=candidate.relative_to(project_root).as_posix(),
                        reason=reason,
                    )
                )
        directories[:] = kept_directories
        for name in filenames:
            candidate = root_path / name
            reason = None
            if name.startswith("."):
                reason = "hidden file"
            elif candidate.is_symlink():
                reason = "symbolic link"
            elif candidate.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                reason = "unsupported file type"
            if reason is None:
                values.append(candidate)
            elif skipped is not None and project_root is not None:
                skipped.append(
                    MangaPreviewIssue(
                        path=candidate.relative_to(project_root).as_posix(),
                        reason=reason,
                    )
                )
    return values


def _natural_key(value: Path) -> tuple[tuple[tuple[int, object], ...], ...]:
    return tuple(
        tuple(
            (1, int(part)) if part.isdigit() else (0, part.casefold())
            for part in _DIGITS.split(component)
            if part
        )
        for component in value.parts
    )


def _validate_image(path: Path, project_root: Path) -> None:
    try:
        path.resolve().relative_to(project_root)
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        relative = path.relative_to(project_root).as_posix()
        raise MangaScanError(f"image cannot be decoded: {relative}") from error


def _directory_meta(directory: Path, resource: Path) -> str:
    relative = directory.relative_to(resource).as_posix()
    return "" if relative == "." else relative


def _build_batches(pages: list[ImagePage], batch_size: int) -> list[ImageBatch]:
    batches: list[ImageBatch] = []
    current_directory: object = object()
    current: list[int] = []
    for page in pages:
        directory = page.meta.get("directory", "")
        if current and (directory != current_directory or len(current) >= batch_size):
            batches.append(
                ImageBatch(batch_index=len(batches), page_indexes=current)
            )
            current = []
        current_directory = directory
        current.append(page.page_index)
    if current:
        batches.append(ImageBatch(batch_index=len(batches), page_indexes=current))
    return batches
