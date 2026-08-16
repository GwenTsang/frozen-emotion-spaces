#!/usr/bin/env python3
"""Fail closed on common mistakes before the first public Git commit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10_000_000
MAX_SINGLE_FILE_BYTES = 5_000_000
REQUIRED = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "DATA_AVAILABILITY.md",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "artifacts/artifacts.lock.json",
    "splits/SHA256SUMS",
}
FORBIDDEN_SUFFIXES = {".zip", ".npy", ".npz", ".pyc", ".aux", ".log", ".out"}
SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
LOCAL_PATH_PATTERNS = (b"/" + b"root/", b"/home/" + b"gwen/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    process = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / part.decode() for part in process.stdout.split(b"\0") if part]


def validate_splits() -> None:
    manifest = ROOT / "splits/SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = filename.removeprefix("*")
        path = ROOT / "splits" / filename
        if sha256_file(path) != digest:
            raise ValueError(f"split checksum mismatch: {filename}")


def validate_artifact_lock() -> tuple[int, int]:
    lock = json.loads(
        (ROOT / "artifacts/artifacts.lock.json").read_text(encoding="utf-8")
    )
    if (
        lock.get("schema_version") != 1
        or lock.get("bucket_private") is not False
        or lock.get("bucket_is_mutable_transport_not_archive") is not True
    ):
        raise ValueError("artifact lock release or Bucket status is invalid")
    logical_paths: set[str] = set()
    objects: dict[str, tuple[str, int]] = {}
    total = 0
    for row in lock.get("artifacts", []):
        logical = str(row["logical_path"])
        logical_path = PurePosixPath(logical)
        if logical_path.is_absolute() or ".." in logical_path.parts:
            raise ValueError(f"unsafe artifact logical path: {logical}")
        if logical in logical_paths:
            raise ValueError(f"duplicate artifact logical path: {logical}")
        logical_paths.add(logical)
        digest = str(row["sha256"])
        size = int(row["bytes"])
        expected = PurePosixPath(
            "objects", "sha256", digest[:2], digest, logical_path.name
        ).as_posix()
        if len(digest) != 64 or row["remote_path"] != expected or size <= 0:
            raise ValueError(f"invalid content-addressed artifact: {logical}")
        identity = (digest, size)
        previous = objects.setdefault(str(row["remote_path"]), identity)
        if previous != identity:
            raise ValueError(f"remote artifact collision: {row['remote_path']}")
        total += size
    if not logical_paths:
        raise ValueError("artifact lock is empty")
    return len(logical_paths), len(objects)


def main() -> int:
    files = tracked_files()
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    missing = sorted(REQUIRED - relative)
    if missing:
        raise ValueError(f"required release files are untracked: {missing}")
    forbidden = sorted(
        path for path in relative if PurePosixPath(path).suffix in FORBIDDEN_SUFFIXES
    )
    if forbidden:
        raise ValueError(f"generated or heavy files are tracked: {forbidden}")
    total = sum(path.stat().st_size for path in files)
    largest = max(files, key=lambda path: path.stat().st_size)
    if total > MAX_TRACKED_BYTES:
        raise ValueError(f"tracked release is too large: {total} bytes")
    if largest.stat().st_size > MAX_SINGLE_FILE_BYTES:
        raise ValueError(f"tracked file is too large: {largest}")
    for path in files:
        if path.stat().st_size > 2_000_000:
            continue
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise ValueError(f"possible credential in tracked file: {path}")
        if any(pattern in content for pattern in LOCAL_PATH_PATTERNS):
            raise ValueError(f"machine-local path in tracked file: {path}")
    validate_splits()
    artifact_count, object_count = validate_artifact_lock()
    print(
        json.dumps(
            {
                "tracked_files": len(files),
                "tracked_bytes": total,
                "largest_file": largest.relative_to(ROOT).as_posix(),
                "largest_file_bytes": largest.stat().st_size,
                "artifact_paths": artifact_count,
                "unique_bucket_objects": object_count,
                "status": "ok",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
