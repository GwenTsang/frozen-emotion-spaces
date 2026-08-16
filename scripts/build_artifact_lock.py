#!/usr/bin/env python3
"""Build a content-addressed lock and optional Bucket upload staging tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath


INCLUDED_ROOTS = (
    "analysis",
    "cache-fast",
    "counterfactual-pilot",
    "geometry-diagnostics",
    "manifests",
    "matched-nulls",
    "observed-vs-counterfactual",
    "results",
    "results-conditional",
)
EXCLUDED_SUFFIXES = {".aux", ".log", ".out"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role_for(path: PurePosixPath) -> str:
    top = path.parts[0]
    if top == "cache-fast":
        return "expensive_cache"
    if top in {"manifests", "analysis", "geometry-diagnostics"}:
        return "validation_output"
    return "golden_experiment_output"


def discover(source: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in INCLUDED_ROOTS:
        root = source / root_name
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES
        )
    return sorted(files, key=lambda path: path.relative_to(source).as_posix())


def safely_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size:
            raise FileExistsError(f"staging collision: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_lock(
    source: Path,
    *,
    bucket_id: str,
    bucket_private: bool,
    release_id: str,
    staging: Path | None,
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for path in discover(source):
        relative = PurePosixPath(path.relative_to(source).as_posix())
        digest = sha256_file(path)
        remote = PurePosixPath(
            "objects", "sha256", digest[:2], digest, relative.name
        )
        artifacts.append(
            {
                "logical_path": relative.as_posix(),
                "remote_path": remote.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "role": role_for(relative),
            }
        )
        if staging is not None:
            safely_link_or_copy(path, staging / Path(remote.as_posix()))

    return {
        "schema_version": 1,
        "release_id": release_id,
        "bucket_id": bucket_id,
        "bucket_private": bool(bucket_private),
        "bucket_is_mutable_transport_not_archive": True,
        "source_status": "new_replication_not_historical_recovery",
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path)
    parser.add_argument(
        "--bucket-id",
        default="GwendalTsang/frozen-emotion-spaces-replication",
    )
    parser.add_argument("--bucket-private", action="store_true")
    parser.add_argument("--release-id", default="replication-2026-08-16-r1")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    source = arguments.source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    lock = build_lock(
        source,
        bucket_id=arguments.bucket_id,
        bucket_private=arguments.bucket_private,
        release_id=arguments.release_id,
        staging=arguments.staging,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    total_bytes = sum(int(row["bytes"]) for row in lock["artifacts"])
    unique = {row["remote_path"] for row in lock["artifacts"]}
    print(
        json.dumps(
            {
                "artifacts": len(lock["artifacts"]),
                "unique_objects": len(unique),
                "logical_bytes": total_bytes,
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
