#!/usr/bin/env python3
"""Benchmark the persistent C++ ONNX sidecar against PyTorch CUDA AMP.

The benchmark deliberately keeps the two runtimes sequential.  This avoids
turning a GPU-memory competition between PyTorch and ONNX Runtime into a
runtime comparison.  The native process is started once, pinged once, and
then receives all warmup and measured separation requests over its JSONL
stdin.  The C++ ``profile.elapsedSeconds`` value is the primary native timing:
it covers the native STFT, model execution, ISTFT, overlap aggregation,
bigshifts, and TTA, but not WAV file I/O or process startup.

Production defaults are a 19.99 second input, ``bigshifts=2``, and TTA on.
The input can be replaced with ``--input``.  The report also includes native
request wall time so file-I/O effects remain visible without contaminating the
inference speedup calculation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import benchmark as benchmark_utils  # type: ignore[import-not-found]
    from config import SeparationConfig  # type: ignore[import-not-found]
    from model import SpectralCore, load_reference_model, set_math_attention  # type: ignore[import-not-found]
    from runtime import read_manifest  # type: ignore[import-not-found]
else:
    from . import benchmark as benchmark_utils
    from .config import SeparationConfig
    from .model import SpectralCore, load_reference_model, set_math_attention
    from .runtime import read_manifest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite number zero or greater")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True, help="compiled vocalarc-onnx-sidecar executable")
    parser.add_argument("--model", type=Path, required=True, help="provider-compatible spectral ONNX model")
    parser.add_argument("--config", type=Path, required=True, help="reference leap_xe YAML configuration")
    parser.add_argument("--checkpoint", type=Path, required=True, help="reference PyTorch checkpoint")
    parser.add_argument("--input", type=Path, help="stereo/mono 44.1 kHz input; otherwise benchmark.py deterministic noise is used")
    parser.add_argument(
        "--seconds",
        type=_positive_float,
        default=19.99,
        help="duration of deterministic input when --input is omitted (default: 19.99)",
    )
    parser.add_argument(
        "--bigshifts",
        type=_positive_int,
        default=2,
        help="circular-shift count (production default: 2)",
    )
    tta = parser.add_mutually_exclusive_group()
    tta.add_argument("--tta", dest="tta", action="store_true", help="enable channel/polarity TTA (default)")
    tta.add_argument("--no-tta", dest="tta", action="store_false", help="disable TTA for diagnostic runs")
    parser.set_defaults(tta=True)
    parser.add_argument("--provider", default="cuda", help="provider passed to the native process (default: cuda)")
    parser.add_argument("--threads", type=_positive_int, default=1, help="native ORT intra-op threads")
    parser.add_argument("--bundle-root", type=Path, help="optional native bundle root passed to the sidecar")
    parser.add_argument("--runtime-dir", type=Path, help="optional directory prepended to the native loader path")
    parser.add_argument("--engine-cache", type=Path, help="optional TensorRT engine-cache directory")
    parser.add_argument("--torch-attention", choices=("flash", "math"), default="flash")
    parser.add_argument("--warmups", type=_nonnegative_int, default=1, help="warmup requests per runtime")
    parser.add_argument(
        "--repeats",
        "--iterations",
        dest="repeats",
        type=_positive_int,
        default=3,
        help="measured requests per runtime (default: 3; --iterations is an alias)",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark-native.json"), help="JSON report path")
    parser.add_argument(
        "--allow-cpu-reference",
        action="store_true",
        help="allow a non-CUDA PyTorch fallback; the default requires native CUDA AMP",
    )
    sync = parser.add_mutually_exclusive_group()
    sync.add_argument(
        "--cuda-sync",
        dest="cuda_sync",
        action="store_true",
        help="call torch.cuda.synchronize around native requests (default when CUDA is available)",
    )
    sync.add_argument(
        "--no-cuda-sync",
        dest="cuda_sync",
        action="store_false",
        help="disable the explicit benchmark-side CUDA synchronization",
    )
    parser.set_defaults(cuda_sync=None)
    parser.add_argument(
        "--max-abs-tol",
        type=_nonnegative_float,
        default=0.10,
        help="maximum permitted absolute waveform error (default: 0.10)",
    )
    parser.add_argument(
        "--rmse-tol",
        type=_nonnegative_float,
        default=0.02,
        help="maximum permitted waveform RMSE (default: 0.02)",
    )
    parser.add_argument(
        "--relative-rmse-tol",
        type=_nonnegative_float,
        default=0.05,
        help="maximum permitted RMSE/reference-RMS (default: 0.05)",
    )
    parser.add_argument(
        "--correlation-tol",
        type=_nonnegative_float,
        default=0.001,
        help="maximum permitted correlation deficit from one (default: 0.001)",
    )
    parser.add_argument(
        "--no-fail-on-parity",
        dest="fail_on_parity",
        action="store_false",
        help="write/report parity failures but exit successfully",
    )
    parser.set_defaults(fail_on_parity=True)
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _input(args: argparse.Namespace, config: SeparationConfig) -> np.ndarray:
    # Keep deterministic-input behavior and sample validation identical to the
    # existing Python benchmark rather than creating a second input utility.
    return benchmark_utils._input(args, config)


def _write_float_wav(path: Path, mix: np.ndarray, config: SeparationConfig) -> None:
    if mix.ndim != 2 or mix.shape[0] != config.channels:
        raise ValueError(f"expected [{config.channels}, samples] input, got {mix.shape}")
    sf.write(
        path,
        np.asarray(mix.T, dtype=np.float32),
        config.sample_rate,
        format="WAV",
        subtype="FLOAT",
    )


def _read_float_wav(path: Path, config: SeparationConfig) -> np.ndarray:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != config.sample_rate:
        raise ValueError(f"native output must be {config.sample_rate} Hz, got {sample_rate}")
    if data.shape[1] == 1:
        data = np.repeat(data, config.channels, axis=1)
    if data.shape[1] != config.channels:
        raise ValueError(f"native output must have {config.channels} channels, got {data.shape[1]}")
    return np.asarray(data.T, dtype=np.float32)


def _synchronize_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reference_benchmark(
    args: argparse.Namespace,
    config: SeparationConfig,
    mix: np.ndarray,
) -> tuple[np.ndarray, list[float], dict[str, object]]:
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.allow_cpu_reference:
        raise RuntimeError(
            "native PyTorch AMP reference requires CUDA; pass --allow-cpu-reference "
            "only for a non-AMP diagnostic benchmark"
        )
    device_name = "cuda" if has_cuda else "cpu"
    model, _ = load_reference_model(args.config, args.checkpoint, device=device_name)
    if args.torch_attention == "math":
        set_math_attention(model)
    core = SpectralCore(model).eval()
    device = next(model.parameters()).device

    def pytorch_predict(batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return benchmark_utils.predict_with_core(core, batch.to(device), config).cpu()
            return benchmark_utils.predict_with_core(core, batch.to(device), config).cpu()

    def separate() -> np.ndarray:
        base = benchmark_utils.bigshifts(pytorch_predict, mix, config, args.bigshifts)
        if args.tta:
            return benchmark_utils.apply_tta(pytorch_predict, mix, base, config, args.bigshifts)
        return base

    # benchmark._timed performs CUDA synchronization around every measured
    # call and is the same timing utility used by onnx/benchmark.py.
    reference_output, timings = benchmark_utils._timed(separate, args.warmups, args.repeats)
    reference = np.asarray(reference_output, dtype=np.float32)
    timing_values = [float(value) for value in timings]
    info: dict[str, object] = {
        "device": str(device),
        "precision": "cuda_amp" if device.type == "cuda" else "cpu_fallback",
        "attention": args.torch_attention,
        "torchVersion": torch.__version__,
        "cudaAvailable": has_cuda,
    }
    if has_cuda:
        info["cudaDevice"] = torch.cuda.get_device_name(device)

    # Do not let the reference closure retain the model while the native
    # process creates its own CUDA context and ORT session.
    del separate, pytorch_predict, core, model
    if has_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return reference, timing_values, info


def _loader_environment(runtime_dir: Path | None) -> dict[str, str] | None:
    if runtime_dir is None:
        return None
    resolved = runtime_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"native runtime directory does not exist: {resolved}")
    environment = os.environ.copy()
    if os.name == "nt":
        variable = "PATH"
    elif sys.platform == "darwin":
        variable = "DYLD_LIBRARY_PATH"
    else:
        variable = "LD_LIBRARY_PATH"
    existing = environment.get(variable, "")
    environment[variable] = str(resolved) + (os.pathsep + existing if existing else "")
    return environment


@dataclass
class NativeCall:
    """One completed sidecar request and its two useful timing views."""

    response: dict[str, Any]
    output: np.ndarray
    profile_seconds: float
    wall_seconds: float


class NativeSidecar:
    """Small synchronous client for the C++ sidecar's persistent JSONL API."""

    def __init__(
        self,
        executable: Path,
        model: Path,
        config: SeparationConfig,
        *,
        provider: str,
        threads: int,
        bundle_root: Path | None = None,
        runtime_dir: Path | None = None,
        engine_cache: Path | None = None,
    ) -> None:
        self.executable = executable
        self.model = model
        self.config = config
        self.provider = provider
        self.threads = threads
        self.bundle_root = bundle_root
        self.runtime_dir = runtime_dir
        self.engine_cache = engine_cache
        self.process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self.request_count = 0
        self.separation_count = 0
        self.startup_seconds = 0.0
        self.ping: dict[str, Any] = {}

    def __enter__(self) -> "NativeSidecar":
        command = [
            os.fspath(self.executable),
            "--model",
            os.fspath(self.model),
            "--provider",
            self.provider,
            "--threads",
            str(self.threads),
        ]
        if self.bundle_root is not None:
            command.extend(("--bundle-root", os.fspath(self.bundle_root)))
        if self.engine_cache is not None:
            command.extend(("--engine-cache", os.fspath(self.engine_cache)))
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_loader_environment(self.runtime_dir),
        )
        try:
            self.ping = self._request({"type": "ping"})
            self.startup_seconds = time.perf_counter() - started
            return self
        except BaseException:
            self.close()
            raise

    def _diagnostics(self) -> str:
        if self.process is None or self.process.stderr is None:
            return ""
        try:
            return self.process.stderr.read().strip()
        except OSError:
            return ""

    def _request(self, payload: dict[str, object]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("native sidecar is not running")
        if self.process.poll() is not None:
            diagnostics = self._diagnostics()
            raise RuntimeError(f"native sidecar exited with code {self.process.returncode}: {diagnostics}")
        self._request_id += 1
        request = {"id": self._request_id, **payload}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            diagnostics = self._diagnostics()
            code = self.process.poll()
            raise RuntimeError(f"native sidecar closed stdout (exit code {code}): {diagnostics}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"native sidecar returned invalid JSON: {line!r}") from error
        if not isinstance(response, dict):
            raise RuntimeError(f"native sidecar response is not an object: {response!r}")
        self.request_count += 1
        if response.get("ok") is not True:
            raise RuntimeError(f"native sidecar request failed: {json.dumps(response, sort_keys=True)}")
        return response

    def separate(self, input_path: Path, vocals_path: Path, instrumental_path: Path, *, bigshifts: int, tta: bool) -> NativeCall:
        started = time.perf_counter()
        response = self._request(
            {
                "type": "separate",
                "inputPath": os.fspath(input_path),
                "vocalsPath": os.fspath(vocals_path),
                "instrumentalPath": os.fspath(instrumental_path),
                "bigshifts": bigshifts,
                "tta": tta,
            }
        )
        wall_seconds = time.perf_counter() - started
        self.separation_count += 1
        output = _read_float_wav(vocals_path, self.config)
        profile = response.get("profile")
        try:
            profile_seconds = float(profile.get("elapsedSeconds", wall_seconds)) if isinstance(profile, dict) else wall_seconds
        except (TypeError, ValueError):
            profile_seconds = wall_seconds
        if not math.isfinite(profile_seconds) or profile_seconds <= 0.0:
            profile_seconds = wall_seconds
        return NativeCall(response, output, profile_seconds, wall_seconds)

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self._request({"type": "shutdown"})
                except (OSError, RuntimeError):
                    pass
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            self.process = None


def _native_benchmark(
    args: argparse.Namespace,
    config: SeparationConfig,
    input_wav: Path,
    work_dir: Path,
    synchronize: bool,
) -> tuple[np.ndarray, list[float], list[float], dict[str, object]]:
    output = None
    profile_timings: list[float] = []
    wall_timings: list[float] = []
    with NativeSidecar(
        _require_file(args.sidecar, "sidecar executable"),
        _require_file(args.model, "ONNX model"),
        config,
        provider=args.provider,
        threads=args.threads,
        bundle_root=args.bundle_root.expanduser().resolve() if args.bundle_root else None,
        runtime_dir=args.runtime_dir,
        engine_cache=args.engine_cache,
    ) as sidecar:
        call_index = 0

        def request() -> NativeCall:
            nonlocal call_index
            call_index += 1
            vocals_path = work_dir / f"native-vocals-{call_index:04d}.wav"
            instrumental_path = work_dir / f"native-instrumental-{call_index:04d}.wav"
            return sidecar.separate(
                input_wav,
                vocals_path,
                instrumental_path,
                bigshifts=args.bigshifts,
                tta=args.tta,
            )

        for _ in range(args.warmups):
            call = request()
            _synchronize_cuda(synchronize)
            output = call.output

        for _ in range(args.repeats):
            _synchronize_cuda(synchronize)
            call = request()
            _synchronize_cuda(synchronize)
            profile_timings.append(call.profile_seconds)
            wall_timings.append(call.wall_seconds)
            output = call.output

        if output is None:
            raise RuntimeError("native benchmark produced no output")
        info: dict[str, object] = {
            "persistent": True,
            "pid": sidecar.process.pid if sidecar.process is not None else None,
            "startupSeconds": sidecar.startup_seconds,
            "requests": sidecar.request_count,
            "separations": sidecar.separation_count,
            "ping": sidecar.ping,
            "providerRequested": args.provider,
            "cudaSynchronization": "torch.cuda.synchronize" if synchronize and torch.cuda.is_available() else "sidecar protocol completion only",
        }
    return output, profile_timings, wall_timings, info


def _parity(metrics: dict[str, float], args: argparse.Namespace) -> dict[str, object]:
    correlation = metrics["correlation"]
    correlation_error = 1.0 - correlation if math.isfinite(correlation) else None
    checks = {
        "maxAbs": math.isfinite(metrics["maxAbs"]) and metrics["maxAbs"] <= args.max_abs_tol,
        "rmse": math.isfinite(metrics["rmse"]) and metrics["rmse"] <= args.rmse_tol,
        "relativeRmse": math.isfinite(metrics["relativeRmse"]) and metrics["relativeRmse"] <= args.relative_rmse_tol,
        "correlation": correlation_error is not None and correlation_error <= args.correlation_tol,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "correlationError": correlation_error,
        "tolerances": {
            "maxAbs": args.max_abs_tol,
            "rmse": args.rmse_tol,
            "relativeRmse": args.relative_rmse_tol,
            "correlationDeficit": args.correlation_tol,
        },
    }


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate a median for an empty timing list")
    return float(statistics.median(values))


def _main() -> int:
    args = _arguments()
    config = SeparationConfig()
    args.config = _require_file(args.config, "reference config")
    args.checkpoint = _require_file(args.checkpoint, "reference checkpoint")
    args.model = _require_file(args.model, "ONNX model")
    if args.input is not None:
        args.input = _require_file(args.input, "input audio")

    mix = _input(args, config)
    duration = mix.shape[-1] / config.sample_rate
    manifest = read_manifest(args.model.expanduser().resolve()) or {}
    model_manifest = {
        key: manifest[key]
        for key in ("profile", "provider", "precision", "inputPrecision", "opset", "batch", "modelSha256")
        if key in manifest
    }

    reference_output, torch_timings, torch_info = _reference_benchmark(args, config, mix)
    # The reference model has been released before this temporary workspace is
    # created, so only one inference runtime owns the GPU at a time.
    with tempfile.TemporaryDirectory(prefix="vocalarc-native-benchmark-") as temporary:
        work_dir = Path(temporary)
        input_wav = work_dir / "input.wav"
        _write_float_wav(input_wav, mix, config)
        native_output, native_timings, native_wall_timings, native_info = _native_benchmark(
            args,
            config,
            input_wav,
            work_dir,
            synchronize=torch.cuda.is_available() if args.cuda_sync is None else bool(args.cuda_sync),
        )

    metrics = benchmark_utils._metrics(reference_output, native_output)
    parity = _parity(metrics, args)
    serializable_metrics = {
        key: (float(value) if math.isfinite(value) else None)
        for key, value in metrics.items()
    }
    torch_median = _median(torch_timings)
    native_median = _median(native_timings)
    wall_median = _median(native_wall_timings)
    report: dict[str, object] = {
        "schemaVersion": 1,
        "input": {
            "path": str(args.input) if args.input is not None else None,
            "generatedBy": "onnx.benchmark._input" if args.input is None else None,
            "seconds": duration,
            "samples": int(mix.shape[-1]),
            "sampleRate": config.sample_rate,
            "channels": config.channels,
        },
        "settings": {
            "bigshifts": args.bigshifts,
            "tta": bool(args.tta),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "threads": args.threads,
            "productionDefaults": args.bigshifts == config.default_bigshifts and bool(args.tta) == config.default_tta,
        },
        "model": model_manifest,
        "torch": {
            **torch_info,
            "timingsSeconds": torch_timings,
            "medianSeconds": torch_median,
            "realTimeFactor": duration / max(torch_median, 1.0e-12),
        },
        "native": {
            **native_info,
            "timingDefinition": "C++ profile.elapsedSeconds; excludes WAV I/O and process startup",
            "timingsSeconds": native_timings,
            "medianSeconds": native_median,
            "realTimeFactor": duration / max(native_median, 1.0e-12),
            "speedupOverPyTorch": torch_median / max(native_median, 1.0e-12),
            "requestWallTimingsSeconds": native_wall_timings,
            "requestWallMedianSeconds": wall_median,
            "requestWallRealTimeFactor": duration / max(wall_median, 1.0e-12),
            "requestWallSpeedupOverPyTorch": torch_median / max(wall_median, 1.0e-12),
        },
        "parity": {
            **serializable_metrics,
            **parity,
        },
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if parity["passed"] or not args.fail_on_parity else 2


if __name__ == "__main__":
    raise SystemExit(_main())
