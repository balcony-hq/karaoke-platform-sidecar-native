"""ONNX Runtime provider selection and waveform separation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .audio import apply_tta, bigshifts, predict_with_core
    from .config import SeparationConfig
except ImportError:  # direct script imports from the onnx directory
    from audio import apply_tta, bigshifts, predict_with_core  # type: ignore[no-redef]
    from config import SeparationConfig  # type: ignore[no-redef]


def available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def create_session(
    model_path: Path,
    *,
    provider: str = "auto",
    intra_op_threads: int = 1,
    engine_cache: Path | None = None,
) -> tuple[Any, str]:
    """Create an ORT session, trying the requested accelerator order."""

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if provider == "auto":
        candidates = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        candidates += ["CoreMLExecutionProvider", "DmlExecutionProvider", "OpenVINOExecutionProvider"]
        candidates += ["XNNPACKExecutionProvider", "CPUExecutionProvider"]
    else:
        aliases = {
            "tensorrt": "TensorrtExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
            "directml": "DmlExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
            "xnnpack": "XNNPACKExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }
        candidates = [aliases.get(provider.lower(), provider)]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.inter_op_num_threads = 1
    session_options.intra_op_num_threads = max(1, int(intra_op_threads))
    if engine_cache is not None:
        engine_cache.mkdir(parents=True, exist_ok=True)
        session_options.add_session_config_entry("session.set_denormal_as_zero", "1")

    errors: list[str] = []
    for candidate in candidates:
        if candidate not in available:
            continue
        try:
            options: dict[str, object] = {}
            if candidate == "TensorrtExecutionProvider" and engine_cache is not None:
                options = {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(engine_cache),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(engine_cache / "timing.cache"),
                    "trt_fp16_enable": False,
                }
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=[(candidate, options)] if options else [candidate],
            )
            actual = session.get_providers()[0] if session.get_providers() else candidate
            return session, actual
        except Exception as error:  # provider installation can be incomplete
            errors.append(f"{candidate}: {error}")

    raise RuntimeError(
        f"no usable ONNX Runtime provider for {model_path}; available={sorted(available)}; "
        + " | ".join(errors)
    )


class OnnxSeparator:
    def __init__(
        self,
        model_path: Path,
        *,
        config: SeparationConfig | None = None,
        provider: str = "auto",
        intra_op_threads: int = 1,
        engine_cache: Path | None = None,
    ) -> None:
        self.config = config or SeparationConfig()
        self.model_path = model_path
        manifest = read_manifest(model_path)
        if manifest is not None:
            manifest_audio = manifest.get("audio")
            if isinstance(manifest_audio, dict):
                required = ("sample_rate", "channels", "n_fft", "hop_length", "win_length", "chunk_size", "num_overlap")
                mismatches = {
                    key: (manifest_audio.get(key), getattr(self.config, key))
                    for key in required
                    if manifest_audio.get(key) != getattr(self.config, key)
                }
                if mismatches:
                    raise ValueError(f"model audio settings do not match runtime config: {mismatches}")
        self.session, self.provider = create_session(
            model_path,
            provider=provider,
            intra_op_threads=intra_op_threads,
            engine_cache=engine_cache,
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("the spectral model must have exactly one input and one output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def predict(self, waveform: torch.Tensor) -> torch.Tensor:
        return predict_with_core(self.session, waveform, self.config)

    def separate(
        self,
        mix: np.ndarray,
        *,
        bigshift_count: int | None = None,
        tta: bool | None = None,
        progress: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        started = time.perf_counter()
        bigshift_count = self.config.default_bigshifts if bigshift_count is None else max(1, int(bigshift_count))
        tta = self.config.default_tta if tta is None else bool(tta)
        vocals = bigshifts(self.predict, mix, self.config, bigshift_count, progress=progress)
        if tta:
            vocals = apply_tta(self.predict, mix, vocals, self.config, bigshift_count, progress=progress)
        instrumental = mix - vocals
        profile = {
            "provider": self.provider,
            "model": str(self.model_path),
            "bigshifts": bigshift_count,
            "tta": tta,
            "elapsedSeconds": time.perf_counter() - started,
            "realTimeFactor": mix.shape[-1] / self.config.sample_rate / max(time.perf_counter() - started, 1e-9),
        }
        return vocals, instrumental, profile


def read_manifest(model_path: Path) -> dict[str, object] | None:
    manifest_path = model_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))
