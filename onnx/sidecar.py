#!/usr/bin/env python3
"""JSONL ONNX Runtime sidecar compatible with the desktop protocol."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import SeparationConfig  # type: ignore[import-not-found]
    from runtime import OnnxSeparator, available_providers  # type: ignore[import-not-found]
else:
    from .config import SeparationConfig
    from .runtime import OnnxSeparator, available_providers


def _response(request_id: object, **values: object) -> None:
    payload = {"id": request_id, "ok": True, **values}
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _error(request_id: object, error: Exception) -> None:
    payload = {
        "id": request_id,
        "ok": False,
        "error": str(error),
        "errorType": type(error).__name__,
    }
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _read_stereo(path: Path, config: SeparationConfig) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != config.sample_rate:
        raise ValueError(f"expected {config.sample_rate} Hz WAV, got {sample_rate} Hz")
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, config.channels, axis=0)
    if audio.shape[0] != config.channels:
        raise ValueError(f"expected mono or stereo input, got {audio.shape[0]} channels")
    return audio, sample_rate


def _write(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, sample_rate, subtype="FLOAT", format="WAV")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--engine-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SeparationConfig()
    separator: OnnxSeparator | None = None
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request: dict[str, object] = json.loads(raw_line)
        request_id = request.get("id")
        request_type = request.get("type")
        try:
            if request_type == "ping":
                _response(
                    request_id,
                    type="pong",
                    runtime="vocalarc-onnx-python-prototype",
                    availableProviders=available_providers(),
                    model=str(args.model),
                    supports={"bigshifts": True, "tta": True, "cpu": "CPUExecutionProvider" in available_providers()},
                )
            elif request_type == "load":
                separator = OnnxSeparator(
                    args.model,
                    config=config,
                    provider=str(request.get("provider", args.provider)),
                    intra_op_threads=args.threads,
                    engine_cache=args.engine_cache,
                )
                _response(request_id, type="loaded", provider=separator.provider)
            elif request_type == "separate":
                if separator is None:
                    separator = OnnxSeparator(
                        args.model,
                        config=config,
                        provider=str(request.get("provider", args.provider)),
                        intra_op_threads=args.threads,
                        engine_cache=args.engine_cache,
                    )
                input_path = Path(str(request["inputPath"]))
                vocals_path = Path(str(request["vocalsPath"]))
                instrumental_path = Path(str(request["instrumentalPath"]))
                mix, sample_rate = _read_stereo(input_path, config)

                def progress(event: dict[str, object]) -> None:
                    print(json.dumps({"id": request_id, "event": "progress", **event}, separators=(",", ":")), flush=True)

                vocals, instrumental, profile = separator.separate(
                    mix,
                    bigshift_count=int(request.get("bigshifts", config.default_bigshifts)),
                    tta=bool(request.get("tta", config.default_tta)),
                    progress=progress,
                )
                _write(vocals_path, vocals, sample_rate)
                _write(instrumental_path, instrumental, sample_rate)
                _response(request_id, type="separated", profile=profile)
            elif request_type == "shutdown":
                _response(request_id, type="shutdown")
                return
            else:
                raise ValueError(f"unsupported request type: {request_type}")
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            _error(request_id, error)


if __name__ == "__main__":
    main()
