from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def normalize_relative_path(path: Path) -> str:
    """Return a stable POSIX-style relative-path representation."""
    return path.as_posix().strip()


def make_sample_id(relative_path: str) -> str:
    """
    Build a deterministic sample identifier from the input-relative path.

    The ID is intentionally independent from generated output names so every
    pipeline stage can use it as the canonical join key.
    """
    normalized = relative_path.replace("\\", "/").strip()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"sample_{digest}"


def iter_input_images(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_manifest(input_dir: Path, manifest_csv: Path) -> list[dict[str, str]]:
    """Create the canonical input manifest used by all downstream stages."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Không thấy input folder:\n{input_dir}")

    image_paths = list(iter_input_images(input_dir))

    if not image_paths:
        raise RuntimeError(f"Không có ảnh trong:\n{input_dir}")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for image_path in image_paths:
        relative_path = normalize_relative_path(image_path.relative_to(input_dir))
        sample_id = make_sample_id(relative_path)

        if sample_id in seen_ids:
            raise RuntimeError(
                "Phát hiện sample_id collision. Điều này rất hiếm; "
                f"hãy tăng độ dài hash. sample_id={sample_id}"
            )

        seen_ids.add(sample_id)

        rows.append(
            {
                "sample_id": sample_id,
                "source_filename": image_path.name,
                "relative_path": relative_path,
                "source_extension": image_path.suffix.lower(),
            }
        )

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "source_filename",
                "relative_path",
                "source_extension",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def load_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Không thấy manifest:\n{manifest_csv}")

    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    required = {"sample_id", "source_filename", "relative_path"}

    if rows:
        missing = required - set(rows[0])
        if missing:
            raise ValueError(
                "Manifest thiếu cột bắt buộc: " + ", ".join(sorted(missing))
            )

    return rows


def sample_id_from_generated_filename(filename: str | None) -> str | None:
    if not filename:
        return None

    stem = Path(str(filename)).stem
    return stem or None
