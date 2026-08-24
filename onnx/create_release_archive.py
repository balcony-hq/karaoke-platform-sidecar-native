#!/usr/bin/env python3
"""Create a deterministic, self-contained ONNX sidecar tar.gz archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


MAX_FILE_BYTES = 2 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def safe_relative(path: str) -> str:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in path or "\x00" in path:
        raise ValueError(f"unsafe bundle path: {path}")
    return relative.as_posix()


def main() -> None:
    options = parse_args()
    bundle = options.bundle.resolve()
    output = options.output.resolve()
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"bundle manifest is missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 3 or manifest.get("runtime") != "vocalarc-onnx":
        raise ValueError(f"invalid sidecar manifest: {manifest_path}")
    expected = {safe_relative("manifest.json")}
    for item in manifest.get("files", []):
        relative = safe_relative(item["path"])
        if relative in expected:
            raise ValueError(f"duplicate archive path: {relative}")
        expected.add(relative)

    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    actual = {safe_relative(path.relative_to(bundle).as_posix()) for path in files}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"bundle contents do not match manifest; missing={missing}, unexpected={unexpected}")
    for path in files:
        if path.stat().st_size >= MAX_FILE_BYTES:
            raise ValueError(f"release file exceeds GitHub's per-file limit: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for path in files:
                    relative = safe_relative(path.relative_to(bundle).as_posix())
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as source:
                        archive.addfile(info, source)

    size = output.stat().st_size
    if size >= MAX_FILE_BYTES:
        raise ValueError(f"archive exceeds GitHub's per-file limit: {output} ({size} bytes)")
    print(json.dumps({"archive": str(output), "bytes": size, "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
