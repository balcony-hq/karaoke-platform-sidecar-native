#!/usr/bin/env python3
"""Build the provider-specific ONNX graphs shipped with the C++ sidecar.

The model weights are identical for every profile.  Only graph operators and
arithmetic precision vary so that each ONNX Runtime execution provider gets a
graph it can execute efficiently:

* CUDA/TensorRT: FP16 graph with the ONNX ``Attention`` operator.
* DirectML/CoreML/OpenVINO: FP16 graph with query-blocked MatMul attention.
* CPU: FP32 graph with the portable blocked attention fallback.

The FP16 graphs retain FP32 graph inputs.  This matches the upstream CUDA AMP
path, where the STFT is produced in FP32 and autocast controls model kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import SeparationConfig  # type: ignore[import-not-found]
    from manual_export import build_model  # type: ignore[import-not-found]
    from model import load_reference_model  # type: ignore[import-not-found]
else:
    from .config import SeparationConfig
    from .manual_export import build_model
    from .model import load_reference_model


PROFILES = {
    "cuda": {"precision": "fp16", "inputPrecision": "fp32", "opset": 24, "attention": "attention"},
    "tensorrt": {"precision": "fp16", "inputPrecision": "fp32", "opset": 24, "attention": "attention"},
    "directml": {"precision": "fp16", "inputPrecision": "fp32", "opset": 20, "attention": "blocked"},
    "coreml": {"precision": "fp16", "inputPrecision": "fp32", "opset": 20, "attention": "blocked"},
    "openvino": {"precision": "fp16", "inputPrecision": "fp32", "opset": 20, "attention": "blocked"},
    "cpu": {"precision": "fp32", "inputPrecision": "fp32", "opset": 20, "attention": "blocked"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), action="append")
    source_root = Path(__file__).resolve().parents[2] / "Music-Source-Separation-Training"
    parser.add_argument("--config", type=Path, default=source_root / "configs/leap_xe_config_voc.yaml")
    parser.add_argument("--checkpoint", type=Path, default=source_root / "checkpoints/bs_leap_xe_voc.ckpt")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/profiles"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else (Path.cwd() / args.checkpoint).resolve()
    output_root = args.output_root if args.output_root.is_absolute() else (Path.cwd() / args.output_root).resolve()
    profiles = args.profile or ["cuda", "directml", "coreml", "openvino", "cpu"]
    unknown = set(profiles) - set(PROFILES)
    if unknown:
        raise ValueError(f"unknown profiles: {sorted(unknown)}")

    separation = SeparationConfig()
    frames = args.frames or separation.chunk_frames
    model, checkpoint_metadata = load_reference_model(config_path, checkpoint_path, device="cpu")
    for profile in profiles:
        settings = PROFILES[profile]
        profile_root = output_root / profile
        model_path = profile_root / "model.onnx"
        print(f"Exporting {profile} -> {model_path}", flush=True)
        graph_stats = build_model(
            model,
            separation,
            args.batch,
            frames,
            model_path,
            opset=settings["opset"],
            attention_mode=settings["attention"],
            precision=settings["precision"],
            input_precision=settings["inputPrecision"],
        )
        manifest = {
            "schemaVersion": 3,
            "runtime": "vocalarc-onnx",
            "profile": profile,
            "provider": profile,
            "model": "bs_leap_xe_voc",
            "graph": "spectral-core",
            "modelFile": "model.onnx",
            "modelSha256": sha256(model_path),
            "modelBytes": model_path.stat().st_size,
            "precision": settings["precision"],
            "inputPrecision": settings["inputPrecision"],
            "opset": settings["opset"],
            "attention": settings["attention"],
            "batch": args.batch,
            "frames": frames,
            "sourceConfig": str(config_path),
            "sourceCheckpoint": str(checkpoint_path),
            "checkpointMetadata": str(checkpoint_metadata),
            "graphStats": graph_stats,
            "audio": separation.as_dict(),
        }
        (profile_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"profile": profile, "bytes": model_path.stat().st_size, "sha256": manifest["modelSha256"]}), flush=True)


if __name__ == "__main__":
    main()
