#!/usr/bin/env python3
"""Stage self-contained native binaries and emit a release manifest.

This uses only the Python standard library. It does not build or bundle a
Python runtime; it verifies that each executable already contains the exact
FP32 model pack and copies it to the platform release name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import zipfile
from pathlib import Path


TRAILER = struct.Struct("<8sQ")
MAGIC = b"VSCEMB01"
MAX_ARTIFACT_BYTES = 2 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_embedded_model(executable: Path, model: Path, model_digest: str) -> None:
    executable_size = executable.stat().st_size
    if executable_size >= MAX_ARTIFACT_BYTES:
        raise ValueError(f"{executable} exceeds the 2 GiB release limit")
    if executable_size < TRAILER.size:
        raise ValueError(f"{executable} is too small to contain an embedded model")
    with executable.open("rb") as handle:
        handle.seek(-TRAILER.size, 2)
        magic, model_bytes = TRAILER.unpack(handle.read(TRAILER.size))
        if magic != MAGIC or model_bytes <= 0 or model_bytes > executable_size - TRAILER.size:
            raise ValueError(f"{executable} has no valid embedded model trailer")
        handle.seek(-TRAILER.size - model_bytes, 2)
        digest = hashlib.sha256()
        remaining = model_bytes
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{executable} embedded model is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    if model_bytes != model.stat().st_size or digest.hexdigest() != model_digest:
        raise ValueError(f"{executable} does not contain the bundled model.f32")


def artifact_name(target: str, gpu: bool) -> str:
    prefix = "vocalarc-native-gpu" if gpu else "vocalarc-native"
    suffix = ".exe" if target.startswith("win32-") else ""
    return f"{prefix}-{target}{suffix}"


def verify_bundle_size(paths: list[Path]) -> int:
    total = sum(path.stat().st_size for path in paths)
    if total >= MAX_ARTIFACT_BYTES:
        raise ValueError(f"release bundle exceeds the 2 GiB limit: {total} bytes")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--model", type=Path, default=Path("assets/model.f32"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-runtime-dll",
        type=Path,
        action="append",
        default=[],
        help="DLL to bundle beside the Windows GPU executable; repeat for transitive dependencies",
    )
    parser.add_argument(
        "--gpu-archive",
        action="store_true",
        help="Package the Windows GPU executable and runtime DLLs into a ZIP archive",
    )
    args = parser.parse_args()

    model = args.model.resolve()
    model_digest = sha256(model)
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}
    staged_paths: list[Path] = []
    for provider, source in (("cpu", args.cpu), ("gpu", args.gpu)):
        if source is None:
            continue
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        verify_embedded_model(source, model, model_digest)
        name = artifact_name(args.target, provider == "gpu")
        destination = args.output / name
        shutil.copy2(source, destination)
        staged_paths.append(destination)
        artifacts[provider] = {
            "file": name,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "embeddedModelBytes": model.stat().st_size,
        }

    if args.gpu_archive and (args.gpu is None or not args.target.startswith("win32-")):
        raise ValueError("--gpu-archive is only supported for Windows GPU releases")

    if args.gpu is not None and args.target.startswith("win32-"):
        if not args.gpu_runtime_dll:
            raise ValueError(
                "Windows GPU releases require --gpu-runtime-dll for every non-system CUDA DLL"
            )
        dependencies: list[dict[str, object]] = []
        seen_names: set[str] = set()
        for source in args.gpu_runtime_dll:
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            name = source.name
            if name.lower() in seen_names:
                raise ValueError(f"duplicate GPU runtime dependency: {name}")
            seen_names.add(name.lower())
            destination = args.output / name
            shutil.copy2(source, destination)
            staged_paths.append(destination)
            dependencies.append({
                "file": name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            })
        artifacts["gpu"]["runtimeDependencies"] = dependencies

        if args.gpu_archive:
            gpu_files = [args.output / artifacts["gpu"]["file"]]
            gpu_files.extend(args.output / dependency["file"] for dependency in dependencies)
            archive_name = f"{Path(artifacts['gpu']['file']).stem}.zip"
            archive_path = args.output / archive_name
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                for path in gpu_files:
                    archive.write(path, arcname=path.name)
            for path in gpu_files:
                path.unlink()
            staged_paths[:] = [path for path in staged_paths if path not in gpu_files]
            staged_paths.append(archive_path)
            artifacts["gpu"]["archive"] = {
                "file": archive_name,
                "bytes": archive_path.stat().st_size,
                "sha256": sha256(archive_path),
            }

    model_manifest_path = model.parent / "manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text()) if model_manifest_path.is_file() else {}
    manifest_path = args.output / "vocalarc-native-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schemaVersion") != 2
            or manifest.get("runtime") != "vocalarc-native"
            or manifest.get("model", {}).get("sha256") != model_digest
        ):
            raise ValueError(f"existing release manifest is incompatible: {manifest_path}")
    else:
        manifest = {
            "schemaVersion": 2,
            "runtime": "vocalarc-native",
            "embeddedModel": True,
            "model": {
                "name": model_manifest.get("model", "bs_leap_xe_voc"),
                "bytes": model.stat().st_size,
                "sha256": model_digest,
                "precision": model_manifest.get("precision", "fp32"),
            },
            "artifacts": {},
        }
    manifest["artifacts"][args.target] = artifacts
    bundle_bytes = verify_bundle_size(staged_paths)
    for artifact in artifacts.values():
        artifact["bundleBytes"] = bundle_bytes
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    verify_bundle_size(staged_paths + [manifest_path])
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
