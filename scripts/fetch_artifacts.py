#!/usr/bin/env python3
"""Fetch Bucket objects named in the Git-locked artifact registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_relative_path(value: str) -> Path:
    logical = PurePosixPath(value)
    if logical.is_absolute() or not logical.parts or ".." in logical.parts:
        raise ValueError(f"unsafe logical path in lock: {value!r}")
    return Path(*logical.parts)


def load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or not isinstance(lock.get("artifacts"), list):
        raise ValueError("unsupported artifact lock")
    if not isinstance(lock.get("bucket_id"), str) or not lock["bucket_id"]:
        raise ValueError("artifact lock lacks a bucket identity")
    return lock


def verify(path: Path, *, expected_bytes: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_bytes
        and sha256_file(path) == expected_sha256
    )


def fetch_one(
    *,
    bucket_id: str,
    remote_path: str,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.part-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    encoded_path = quote(remote_path, safe="/")
    url = f"https://huggingface.co/buckets/{bucket_id}/resolve/{encoded_path}"
    try:
        request = Request(url, headers={"User-Agent": "frozen-emotion-spaces-fetch/1"})
        with urlopen(request) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if not verify(
            temporary,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        ):
            raise ValueError(f"downloaded object failed verification: {url}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("artifacts/artifacts.lock.json"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("artifacts/downloads/replication-20260816"),
    )
    parser.add_argument(
        "--role",
        action="append",
        choices=("expensive_cache", "golden_experiment_output", "validation_output"),
        help="fetch only selected roles; may be repeated",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    lock = load_lock(arguments.lock)
    selected_roles = set(arguments.role or ())
    rows = [
        row
        for row in lock["artifacts"]
        if not selected_roles or row.get("role") in selected_roles
    ]
    fetched = skipped = failed = 0
    for row in rows:
        relative = checked_relative_path(str(row["logical_path"]))
        destination = arguments.destination / relative
        expected_bytes = int(row["bytes"])
        expected_sha256 = str(row["sha256"])
        if verify(
            destination,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        ):
            skipped += 1
            continue
        if arguments.verify_only:
            print(f"MISSING_OR_INVALID {relative}")
            failed += 1
            continue
        fetch_one(
            bucket_id=str(lock["bucket_id"]),
            remote_path=str(row["remote_path"]),
            destination=destination,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        fetched += 1
    print(
        json.dumps(
            {"selected": len(rows), "fetched": fetched, "valid": skipped, "failed": failed},
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
