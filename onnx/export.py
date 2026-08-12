#!/usr/bin/env python3
"""Export the upstream BS-RoFormer spectral core to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audio import waveform_to_stft  # type: ignore[import-not-found]
    from config import SeparationConfig  # type: ignore[import-not-found]
    from model import SpectralCore, load_reference_model, set_math_attention  # type: ignore[import-not-found]
else:
    from .audio import waveform_to_stft
    from .config import SeparationConfig
    from .model import SpectralCore, load_reference_model, set_math_attention


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_root = Path(__file__).resolve().parents[2] / "Music-Source-Separation-Training"
    parser.add_argument(
        "--config",
        type=Path,
        default=source_root / "configs/leap_xe_config_voc.yaml",
        help="upstream model YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=source_root / "checkpoints/bs_leap_xe_voc.ckpt",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/leap_xe"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=0, help="0 uses the reference chunk frame count")
    parser.add_argument("--opset", type=int, default=24)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument(
        "--input-precision",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="graph input type; auto keeps STFT values in FP32 for FP16 AMP parity",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("attention", "blocked"),
        default="attention",
        help="standard ONNX Attention kernel, or a portable query-blocked MatMul fallback",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="use the TorchScript ONNX exporter directly; required by the current PyTorch 2.13 graph decomposer",
    )
    parser.add_argument(
        "--torch-trace",
        action="store_true",
        help="use torch.onnx.export instead of the memory-bounded direct ONNX graph builder",
    )
    parser.add_argument(
        "--keep-fused-attention",
        action="store_true",
        help="try exporting PyTorch fused attention; portable math attention is the default",
    )
    parser.add_argument("--verify", action="store_true", help="run ONNX checker and a tensor parity check")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else (Path.cwd() / args.checkpoint).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir).resolve()
    separation = SeparationConfig()
    frames = args.frames or separation.chunk_frames
    if args.batch <= 0 or frames <= 0:
        raise ValueError("batch and frames must be positive")

    model, _ = load_reference_model(config_path, checkpoint_path, device="cpu")
    core = SpectralCore(model).eval()
    if not args.keep_fused_attention:
        set_math_attention(core)

    dummy_waveform = torch.zeros(args.batch, separation.channels, separation.chunk_size)
    dummy_stft = waveform_to_stft(dummy_waveform, separation)
    if dummy_stft.shape[2] != frames:
        if args.frames:
            dummy_stft = torch.zeros(
                args.batch,
                separation.spectral_channels,
                frames,
                2,
                dtype=torch.float32,
            )
        else:
            raise AssertionError(f"unexpected reference frame count: {dummy_stft.shape}")
    export_stft = dummy_stft.half() if args.precision == "fp16" else dummy_stft
    input_precision = ("fp32" if args.precision == "fp16" else "fp32") if args.input_precision == "auto" else args.input_precision
    export_stft = dummy_stft.half() if input_precision == "fp16" else dummy_stft

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"bs_roformer_leap_xe_spectral_b{args.batch}_f{frames}.onnx"
    print(f"Exporting {model_path}")
    manual_stats: dict[str, object] | None = None
    if not args.torch_trace:
        if args.keep_fused_attention:
            raise ValueError("the direct exporter only supports portable math attention")
        from manual_export import build_model

        manual_stats = build_model(
            model,
            separation,
            args.batch,
            frames,
            model_path,
            opset=args.opset,
            attention_mode=args.attention_mode,
            precision=args.precision,
            input_precision=input_precision,
        )
    else:
        wrapper = core.half() if args.precision == "fp16" else core
        export_kwargs = dict(
            input_names=["stft"],
            output_names=["mask"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=not args.legacy,
            optimize=True,
            verify=False,
            external_data=False,
        )
        if args.legacy:
            export_kwargs.pop("optimize")
            export_kwargs.pop("verify")
            export_kwargs.pop("external_data")
        try:
            torch.onnx.export(wrapper, (export_stft,), model_path, **export_kwargs)
        except Exception as error:
            if args.keep_fused_attention or args.legacy:
                raise
            print(f"dynamo export failed with math attention: {error}", file=sys.stderr)
            print("Retrying with the legacy exporter.", file=sys.stderr)
            legacy_kwargs = dict(export_kwargs)
            legacy_kwargs["dynamo"] = False
            legacy_kwargs.pop("optimize", None)
            legacy_kwargs.pop("verify", None)
            legacy_kwargs.pop("external_data", None)
            torch.onnx.export(wrapper, (export_stft,), model_path, **legacy_kwargs)

    # Loading a session performs protobuf parsing and provider-independent
    # graph validation without importing the third-party ``onnx`` module here
    # (the source directory itself is named ``onnx``).
    import onnxruntime as ort

    validation_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    del validation_session
    metadata = {
        "schemaVersion": 1,
        "runtime": "onnxruntime",
        "model": "bs_leap_xe_voc",
        "graph": "spectral-core",
        "modelFile": model_path.name,
        "modelSha256": _sha256(model_path),
        "precision": args.precision,
        "inputPrecision": input_precision,
        "opset": args.opset,
        "sourceConfig": str(config_path),
        "sourceCheckpoint": str(checkpoint_path),
        "attention": "fused" if args.keep_fused_attention else args.attention_mode,
        "exporter": "direct-onnx" if not args.torch_trace else "torch.onnx",
        "graphStats": manual_stats,
        "input": {"name": "stft", "shape": list(dummy_stft.shape), "layout": "B,F_times_C,T,real_imag"},
        "output": {"name": "mask", "shape": [args.batch, separation.num_stems, frames, separation.spectral_channels, 2]},
        "audio": separation.as_dict(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.verify:
        verification_providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            verification_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=verification_providers)
        rng = np.random.default_rng(0xA11CE)
        values = rng.standard_normal(dummy_stft.shape, dtype=np.float32)
        verification_values = values.astype(np.float16) if input_precision == "fp16" else values
        reference_precision = "fp32"
        if args.precision == "fp16" and torch.cuda.is_available():
            # The upstream configuration enables CUDA AMP.  RMSNorm and
            # other numerically sensitive operations stay in FP32 while
            # eligible matrix operations use FP16.
            reference_precision = "cuda_amp"
            reference_core = core.to("cuda")
            torch_input = torch.from_numpy(values).cuda()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                torch_output = reference_core(torch_input).float().cpu().numpy()
        else:
            if args.precision == "fp16":
                # There is no CUDA AMP reference on a CPU-only export host;
                # retain a useful graph smoke test and label its arithmetic.
                reference_precision = "fp16_cpu_fallback"
                reference_core = core.half()
                torch_input = torch.from_numpy(values).half() if input_precision == "fp16" else torch.from_numpy(values)
            else:
                reference_core = core
                torch_input = torch.from_numpy(values)
            with torch.inference_mode():
                torch_output = reference_core(torch_input).float().numpy()
        onnx_output = session.run(["mask"], {"stft": verification_values})[0].astype(np.float32)
        difference = np.abs(torch_output - onnx_output)
        print(json.dumps({
            "verification": {
                "referencePrecision": reference_precision,
                "provider": session.get_providers()[0],
                "maxAbs": float(difference.max()),
                "meanAbs": float(difference.mean()),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            },
            "providers": ort.get_available_providers(),
        }, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
