from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_META_FILENAME = ".checkpoint_meta.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def fingerprint_input_files(
    directories: Iterable[Path],
    extensions: set[str] | None = None,
) -> str:
    """
    Build a fast upstream fingerprint from file path + size + mtime_ns.

    This intentionally avoids hashing every image byte. It is strong enough to
    invalidate OCR checkpoints when an upstream crop/preprocess stage rewrites
    or changes its generated files, without re-reading a large image dataset.
    """
    digest = hashlib.sha256()

    for directory in sorted((Path(path) for path in directories), key=str):
        digest.update(str(directory.resolve()).encode("utf-8"))

        if not directory.exists():
            digest.update(b"<missing>")
            continue

        files = sorted(path for path in directory.rglob("*") if path.is_file())

        for path in files:
            if extensions and path.suffix.lower() not in extensions:
                continue

            stat = path.stat()
            relative = path.relative_to(directory).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))

    return digest.hexdigest()


def build_stage_fingerprint(
    stage_name: str,
    code_paths: Iterable[Path],
    input_directories: Iterable[Path],
    input_extensions: set[str] | None = None,
    extra: dict[str, Any] | None = None,
    packages: Iterable[str] = (),
) -> tuple[str, dict[str, Any]]:
    code_hashes: dict[str, str | None] = {}

    for path in sorted((Path(path) for path in code_paths), key=str):
        code_hashes[str(path.resolve())] = sha256_file(path) if path.exists() else None

    payload = {
        "stage_name": stage_name,
        "code_hashes": code_hashes,
        "input_fingerprint": fingerprint_input_files(
            input_directories,
            extensions=input_extensions,
        ),
        "package_versions": package_versions(packages),
        "extra": extra or {},
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    fingerprint = hashlib.sha256(serialized).hexdigest()
    return fingerprint, payload


def _atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        file.write(text)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, path)


def atomic_write_json(path: Path, value: Any, indent: int = 2) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=indent)
    _atomic_replace_text(path, text)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8-sig",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, path)


def append_jsonl(path: Path, record: dict[str, Any], durable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

    with path.open("a", encoding="utf-8") as file:
        file.write(line)
        file.flush()
        if durable:
            os.fsync(file.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    for index, line in enumerate(lines):
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            # A crash can leave only the final appended line incomplete. Ignore
            # that one line; corruption earlier in the file should not be hidden.
            if index == len(lines) - 1:
                break
            raise

        if isinstance(item, dict):
            records.append(item)

    return records


def dedupe_records(
    records: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for record in records:
        value = record.get(key)
        if value is None:
            continue
        normalized = str(value)
        if normalized not in by_key:
            order.append(normalized)
        by_key[normalized] = record

    return [by_key[value] for value in order]


def prepare_stage_checkpoint(
    output_dir: Path,
    stage_name: str,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    checkpoint_paths: Iterable[Path],
) -> bool:
    """
    Return True when an existing checkpoint is compatible and can be resumed.

    If the stage/input/environment fingerprint changed, only this stage's
    checkpoint/output files are invalidated. Upstream expensive artifacts are
    untouched.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / CHECKPOINT_META_FILENAME
    checkpoint_paths = [Path(path) for path in checkpoint_paths]

    previous: dict[str, Any] | None = None
    if meta_path.exists():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None

    compatible = bool(previous and previous.get("fingerprint") == fingerprint)

    if not compatible:
        had_old_state = bool(previous) or any(path.exists() for path in checkpoint_paths)

        for path in checkpoint_paths:
            if path.exists():
                path.unlink()

        meta = {
            "stage_name": stage_name,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "invalidated_previous_fingerprint": (
                previous.get("fingerprint") if previous else None
            ),
        }
        atomic_write_json(meta_path, meta)

        if had_old_state:
            print(
                f"Checkpoint {stage_name}: fingerprint thay đổi -> "
                "checkpoint cũ đã được invalidated."
            )
        return False

    return True
