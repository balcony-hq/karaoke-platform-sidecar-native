#!/usr/bin/env python3
"""Compare PyTorch and ONNX Runtime speed and waveform parity."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audio import apply_tta, bigshifts, predict_with_core  # type: ignore[import-not-found]
    from config import SeparationConfig  # type: ignore[import-not-found]
    from model import SpectralCore, load_reference_model, set_math_attention  # type: ignore[import-not-found]
    from runtime import create_session, read_manifest  # type: ignore[import-not-found]
else:
    from .audio import apply_tta, bigshifts, predict_with_core
    from .config import SeparationConfig
    from .model import SpectralCore, load_reference_model, set_math_attention
    from .runtime import create_session, read_manifest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="exported spectral ONNX model")
    parser.add_argument("--input", type=Path, help="optional stereo WAV/FLAC; otherwise deterministic noise is used")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--torch-attention", choices=("flash", "math"), default="flash")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto", help="ONNX input precision")
    parser.add_argument(
        "--torch-precision",
        choices=("auto", "amp", "fp32", "fp16"),
        default="amp",
        help="reference precision: CUDA AMP (the upstream default), full FP32, or all-FP16",
    )
    parser.add_argument("--bigshifts", type=int, default=1)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    return parser.parse_args()


def _input(args: argparse.Namespace, config: SeparationConfig) -> np.ndarray:
    if args.input:
        data, rate = sf.read(args.input, dtype="float32", always_2d=True)
        if rate != config.sample_rate:
            raise ValueError(f"input must be {config.sample_rate} Hz, got {rate}")
        data = data.T
        if data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)
        if data.shape[0] != 2:
            raise ValueError("benchmark input must be mono or stereo")
        return data
    samples = max(1, round(args.seconds * config.sample_rate))
    rng = np.random.default_rng(0x20260810)
    return rng.normal(0.0, 0.03, size=(config.channels, samples)).astype(np.float32)


def _metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    difference = estimate.astype(np.float64) - reference.astype(np.float64)
    reference_flat = reference.astype(np.float64).reshape(-1)
    estimate_flat = estimate.astype(np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    ref_rms = float(np.sqrt(np.mean(np.square(reference_flat))))
    corr = float(np.corrcoef(reference_flat, estimate_flat)[0, 1]) if ref_rms > 0 else 1.0
    return {
        "maxAbs": float(np.max(np.abs(difference))),
        "meanAbs": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "relativeRmse": rmse / max(ref_rms, 1e-12),
        "correlation": corr,
        "referenceRms": ref_rms,
        "estimateRms": float(np.sqrt(np.mean(np.square(estimate_flat)))),
    }


def _timed(function, warmups: int, iterations: int) -> tuple[object, list[float]]:
    output = None
    for _ in range(max(0, warmups)):
        output = function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(max(1, iterations)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    return output, timings


def main() -> None:
    args = _args()
    config = SeparationConfig()
    mix = _input(args, config)
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()

    manifest = read_manifest(args.model.resolve()) or {}
    model_precision = args.precision
    if model_precision == "auto":
        model_precision = str(manifest.get("precision", "fp32"))
    torch_precision = ("amp" if model_precision == "fp16" else model_precision) if args.torch_precision == "auto" else args.torch_precision

    # Keep the two runtimes out of GPU memory at the same time.  On a 16 GB
    # card, loading both the reference model and ORT can evict the attention
    # workspace and make a valid benchmark fail with an artificial OOM.
    model, _ = load_reference_model(config_path, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")
    if args.torch_attention == "math":
        set_math_attention(model)
    core = SpectralCore(model).eval()
    if torch_precision == "fp16":
        core = core.half()

    device = next(model.parameters()).device

    def pytorch_predict(batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if torch_precision == "amp" and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return predict_with_core(core, batch.to(device), config).cpu()
            return predict_with_core(core, batch.to(device), config).cpu()

    def separate(predict):
        base = bigshifts(predict, mix, config, args.bigshifts)
        return apply_tta(predict, mix, base, config, args.bigshifts) if args.tta else base

    # Core parity uses the actual chunk path, including STFT and ISTFT.
    reference_vocals = separate(pytorch_predict)
    _, pytorch_timings = _timed(lambda: separate(pytorch_predict), args.warmups, args.iterations)
    torch_device = str(device)
    del core, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    session, provider = create_session(args.model.resolve(), provider=args.provider, intra_op_threads=args.threads)
    onnx_predict = lambda batch: predict_with_core(session, batch, config)
    onnx_vocals = separate(onnx_predict)
    _, onnx_timings = _timed(
        lambda: separate(onnx_predict),
        args.warmups,
        args.iterations,
    )

    pytorch_median = statistics.median(pytorch_timings)
    onnx_median = statistics.median(onnx_timings)
    duration = mix.shape[-1] / config.sample_rate
    result = {
        "schemaVersion": 1,
        "inputSeconds": duration,
        "inputSamples": int(mix.shape[-1]),
        "bigshifts": args.bigshifts,
        "tta": bool(args.tta),
        "precision": model_precision,
        "torchPrecision": torch_precision,
        "torch": {
            "device": torch_device,
            "attention": args.torch_attention,
            "timingsSeconds": pytorch_timings,
            "medianSeconds": pytorch_median,
            "realTimeFactor": duration / max(pytorch_median, 1e-12),
        },
        "onnx": {
            "provider": provider,
            "availableProviders": __import__("onnxruntime").get_available_providers(),
            "timingsSeconds": onnx_timings,
            "medianSeconds": onnx_median,
            "realTimeFactor": duration / max(onnx_median, 1e-12),
            "speedupOverPyTorch": pytorch_median / max(onnx_median, 1e-12),
        },
        "parity": _metrics(reference_vocals, onnx_vocals),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
