#!/usr/bin/env python3
"""Benchmark the native sidecar without a Python inference dependency."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows has no POSIX resource module.
    resource = None


RATE = 44_100
CHANNELS = 2


def peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def fixture(seconds: float) -> list[float]:
    samples = max(1, round(seconds * RATE))
    generator = random.Random(0x20260810)
    return [generator.gauss(0.0, 0.03) for _ in range(CHANNELS * samples)]


def write_pcm16(path: Path, interleaved: list[float]) -> None:
    pcm = bytearray(len(interleaved) * 2)
    for index, value in enumerate(interleaved):
        sample = max(-32768, min(32767, round(value * 32767.0)))
        struct.pack_into("<h", pcm, index * 2, sample)
    with path.open("wb") as handle:
        data_bytes = len(pcm)
        block_align = CHANNELS * 2
        byte_rate = RATE * block_align
        handle.write(b"RIFF")
        handle.write(struct.pack("<I", 36 + data_bytes))
        handle.write(b"WAVEfmt ")
        handle.write(struct.pack("<IHHIIHH", 16, 1, CHANNELS, RATE, byte_rate, block_align, 16))
        handle.write(b"data")
        handle.write(struct.pack("<I", data_bytes))
        handle.write(pcm)


def read_float_wav(path: Path) -> list[float]:
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise RuntimeError("native output is not a RIFF/WAVE file")
    offset = 12
    channels = width = audio_format = 0
    payload: bytes | None = None
    while offset + 8 <= len(raw):
        chunk = raw[offset:offset + 4]
        size = struct.unpack_from("<I", raw, offset + 4)[0]
        body = offset + 8
        if chunk == b"fmt " and size >= 16:
            audio_format, channels = struct.unpack_from("<HH", raw, body)
            width = struct.unpack_from("<H", raw, body + 14)[0]
        elif chunk == b"data":
            payload = raw[body:body + size]
        offset = body + size + (size & 1)
    if audio_format != 3 or channels != CHANNELS or width != 32 or payload is None:
        raise RuntimeError("native output must be stereo IEEE-float32 WAV")
    if len(payload) % 4 != 0:
        raise RuntimeError("native output has an invalid float32 payload")
    return [struct.unpack_from("<f", payload, offset)[0] for offset in range(0, len(payload), 4)]


def request(process: subprocess.Popen[str], value: dict[str, object]) -> dict[str, object]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("native process pipes are unavailable")
    process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read() if process.stderr else ""
        raise RuntimeError(stderr or "native process exited without a response")
    response = json.loads(line)
    if not response.get("ok", False):
        raise RuntimeError(str(response.get("error", response)))
    return response


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / max(1, len(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--time-band-group", type=int, default=32)
    parser.add_argument("--frequency-frame-group", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audio = fixture(args.seconds)
    samples = len(audio) // CHANNELS
    measurements: list[float] = []
    response: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="vocalarc-native-") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.wav"
        output_path = directory / "vocals.wav"
        write_pcm16(input_path, audio)
        command = [str(args.native), "--model", str(args.model)]
        if args.threads > 0:
            command.extend(("--threads", str(args.threads)))
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        request(process, {"id": 1, "type": "load"})
        for _ in range(max(0, args.warmups)):
            request(
                process,
                {
                    "id": 2,
                    "type": "separate",
                    "inputPath": str(input_path),
                    "vocalsPath": str(output_path),
                    "timeBandGroup": args.time_band_group,
                    "frequencyFrameGroup": args.frequency_frame_group,
                },
            )
        for index in range(max(1, args.repetitions)):
            started = time.perf_counter()
            response = request(
                process,
                {
                    "id": 10 + index,
                    "type": "separate",
                    "inputPath": str(input_path),
                    "vocalsPath": str(output_path),
                    "timeBandGroup": args.time_band_group,
                    "frequencyFrameGroup": args.frequency_frame_group,
                },
            )
            measurements.append((time.perf_counter() - started) * 1000.0)
        request(process, {"id": 99, "type": "shutdown"})
        process.wait(timeout=5)
        if process.returncode != 0:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(stderr.strip() or f"native process exited with {process.returncode}")
        output = read_float_wav(output_path)

    median = sorted(measurements)[len(measurements) // 2]
    report: dict[str, object] = {
        "runtime": "vocalarc-native-gpu" if str(response.get("backend", "")).startswith("native-cuda") else "vocalarc-native-cpu",
        "binaryBytes": args.native.stat().st_size,
        "modelBytes": args.model.stat().st_size,
        "inputSamples": samples,
        "inputSeconds": samples / RATE,
        "threads": args.threads or os.cpu_count(),
        "timeBandGroup": response.get("timeBandGroup", args.time_band_group),
        "frequencyFrameGroup": response.get("frequencyFrameGroup", args.frequency_frame_group),
        "warmups": max(0, args.warmups),
        "repetitions": len(measurements),
        "inferenceMs": measurements,
        "medianInferenceMs": median,
        "realTimeFactor": (samples / RATE) / (median / 1000.0),
        "nativeOutputRms": rms(output),
        "nativeBackend": response.get("backend"),
        "nativeDevice": response.get("device"),
        "nativePeakRssBytes": response.get("peakRssBytes") or peak_rss_bytes(),
        "simd": response.get("simd"),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
