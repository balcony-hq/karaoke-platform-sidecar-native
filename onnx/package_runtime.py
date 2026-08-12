#!/usr/bin/env python3
"""Stage a small, self-contained ONNX Runtime sidecar bundle.

The release contains only the executable, one graph, the ORT libraries needed
by that provider, and an integrity manifest.  Headers, CMake files, Python,
checkpoints, and duplicate graphs never enter a release bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path


MAX_FILE_BYTES = 2 * 1024**3
PROVIDER_LIBS = {
    "cpu": [],
    "cuda": ["libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so"],
    "tensorrt": ["libonnxruntime_providers_tensorrt.so", "libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so"],
    # DirectML is compiled into the official Microsoft.ML.OnnxRuntime.DirectML
    # onnxruntime.dll; only the shared provider loader is a separate file.
    "directml": ["onnxruntime_providers_shared.dll"],
    # CoreML is built into the official macOS ONNX Runtime binary.
    "coreml": [],
    "openvino": ["libonnxruntime_providers_openvino.so", "libonnxruntime_providers_shared.so"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if os.name != "nt" and target.parent.name == "runtime":
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ort-root", type=Path, required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDER_LIBS), required=True)
    parser.add_argument("--target", required=True, help="release target, e.g. linux-x64 or win32-x64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-version", default="vocalarc-onnx-dev")
    parser.add_argument("--dependency", type=Path, action="append", default=[], help="additional accelerator runtime library")
    parser.add_argument("--provider-library", type=Path, action="append", default=[], help="override/add provider library")
    return parser.parse_args()


def main() -> None:
    options = args()
    output = options.output.resolve()
    runtime = output / "runtime"
    models = output / "models"
    output.mkdir(parents=True, exist_ok=True)
    copy_file(options.sidecar.resolve(), runtime / options.sidecar.name)
    copy_file(options.model.resolve(), models / "model.onnx")

    ort_lib = options.ort_root.resolve() / "lib"
    if not ort_lib.is_dir():
        raise FileNotFoundError(f"ORT lib directory not found: {ort_lib}")
    core_candidates = [
        "onnxruntime.dll",
        "libonnxruntime.so.1",
        "libonnxruntime.so",
        "libonnxruntime.dylib",
    ]
    core = next((ort_lib / name for name in core_candidates if (ort_lib / name).exists()), None)
    if core is None:
        raise FileNotFoundError(f"could not find the ORT core library in {ort_lib}")
    copy_file(core, runtime / core.name)

    provider_sources = list(options.provider_library)
    if not provider_sources:
        provider_sources = [ort_lib / name for name in PROVIDER_LIBS[options.provider]]
    for source in provider_sources:
        source = source.resolve()
        if source.exists():
            copy_file(source, runtime / source.name)
        elif options.provider != "cpu":
            raise FileNotFoundError(source)

    for dependency in options.dependency:
        copy_file(dependency.resolve(), runtime / dependency.name)

    files = []
    total_bytes = 0
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size >= MAX_FILE_BYTES:
            raise ValueError(f"release file exceeds GitHub's per-file limit: {path} ({size} bytes)")
        relative = path.relative_to(output).as_posix()
        files.append({"path": relative, "bytes": size, "sha256": sha256(path)})
        total_bytes += size
    manifest = {
        "schemaVersion": 3,
        "runtime": "vocalarc-onnx",
        "version": options.release_version,
        "target": options.target,
        "provider": options.provider,
        "executable": f"runtime/{options.sidecar.name}",
        "model": "models/model.onnx",
        "files": files,
        "totalBytes": total_bytes,
        "limits": {"maxFileBytes": MAX_FILE_BYTES, "maxReleaseFileBytes": MAX_FILE_BYTES},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "provider": options.provider, "files": files, "totalBytes": total_bytes}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"package_runtime: {error}", file=sys.stderr)
        raise
