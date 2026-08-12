# VocalArc ONNX sidecar

This directory contains the provider-neutral ONNX sidecar for the
`bs_leap_xe_voc` model. It deliberately leaves audio DSP outside the ONNX
graph: the graph accepts real/imaginary STFT bins and returns real/imaginary
complex masks. This keeps the model graph portable across ONNX Runtime
execution providers while retaining the exact upstream STFT, ISTFT, overlap,
bigshift, and TTA semantics.

The source model and reference behavior come from the sibling
`Music-Source-Separation-Training` checkout. The default settings are the
ones in `configs/leap_xe_config_voc.yaml` and the default quality pass matches
`run.txt`: `bigshifts=2`, TTA enabled.

The graph uses the standard ONNX `Attention` operator (opset 24) for CUDA and
TensorRT, allowing those providers to select fused memory-efficient kernels.
DirectML, CoreML, and OpenVINO receive the same FP16 weights in the portable
query-blocked graph (opset 20). FP32 remains available for CPU fallback or
strict reference arithmetic. The FP16 graphs accept FP32 STFT inputs, matching
the upstream CUDA AMP boundary.

`export_profiles.py` emits all provider graphs and manifests in one pass:

```sh
/home/dev/code/.venv/bin/python export_profiles.py \
  --output-root artifacts/profiles
```

## Export

Run from this directory with the workspace virtual environment:

```sh
/home/dev/code/.venv/bin/python export.py \
  --config ../../Music-Source-Separation-Training/configs/leap_xe_config_voc.yaml \
  --checkpoint ../../Music-Source-Separation-Training/checkpoints/bs_leap_xe_voc.ckpt \
  --output-dir artifacts/leap_xe \
  --verify
```

The export is a fixed-shape spectral graph. The reference chunk has 1,722
frames and the default graph has input shape `[1, 2050, 1722, 2]` and output
shape `[1, 1, 1722, 2050, 2]`. Use `--precision fp32` for FP32 or
`--attention-mode blocked --opset 18` for the standard-op fallback. Use
`--batch 2` to export the upstream batch size as a separate graph.

## Parity and speed comparison

After exporting a graph:

```sh
/home/dev/code/.venv/bin/python benchmark.py \
  --config ../../Music-Source-Separation-Training/configs/leap_xe_config_voc.yaml \
  --checkpoint ../../Music-Source-Separation-Training/checkpoints/bs_leap_xe_voc.ckpt \
  --model artifacts/leap_xe/bs_roformer_leap_xe_spectral_b1_f1722.onnx \
  --provider cuda \
  --torch-attention flash \
  --torch-precision amp \
  --seconds 19.99 \
  --bigshifts 2 \
  --tta \
  --warmups 1 \
  --iterations 3 \
  --output artifacts/leap_xe/benchmark.json
```

The report includes end-to-end waveform parity, provider selection, median
latency, real-time factor, and ONNX speedup over PyTorch. The benchmark
synchronizes CUDA before timing, runs the two runtimes sequentially to avoid
artificial GPU-memory pressure, and includes STFT, model execution, ISTFT,
overlap aggregation, bigshifts, and TTA. For an exact FP32 graph parity check,
use `export.py --precision fp32 --verify`; the exporter compares the full
1,722-frame graph against the PyTorch spectral core.

## JSONL sidecar

```sh
/home/dev/code/.venv/bin/python sidecar.py \
  --model artifacts/leap_xe/bs_roformer_leap_xe_spectral_b1_f1722.onnx \
  --provider auto \
  --engine-cache artifacts/leap_xe/engine-cache
```

It supports `ping`, `load`, `separate`, and `shutdown`. A separation request
accepts `bigshifts` and `tta` fields and writes float32 WAV vocals and
instrumental outputs. Provider-specific engine caching is enabled for the
TensorRT path when an engine-cache directory is supplied. The sidecar accepts
mono or stereo 44.1 kHz WAV input.

The Python process is the reference sidecar contract; the release packaging
step moves the same session/manifest contract behind the C++ ONNX Runtime API.

## C++ release sidecar

The shipped process is `native/onnx_sidecar.cpp`; it includes the WAV front
end, overlap aggregation, bigshifts, TTA, JSONL protocol, and ONNX Runtime
provider selection. It has no Python or model dependency at runtime:

```sh
cmake -S native -B build -DORT_ROOT=/path/to/onnxruntime-sdk
cmake --build build --config Release
```

Stage a release bundle with only the executable, graph, and provider libraries:

```sh
/home/dev/code/.venv/bin/python package_runtime.py \
  --sidecar build/vocalarc-onnx-sidecar \
  --model artifacts/profiles/cuda/model.onnx \
  --ort-root /path/to/onnxruntime-sdk \
  --provider cuda --target linux-x64 \
  --output release/linux-x64
```

Each staged file is hashed and checked against GitHub's 2 GiB per-file limit;
headers, checkpoints, Python, symbols, and unused provider libraries are not
included.

On the RTX 5080 test machine, the CUDA FP16 graph measured 2.46 s for a
19.99-second chunk in the upstream PyTorch CUDA AMP path, 2.52 s in Python
ONNX Runtime, and 3.23 s in the C++ sidecar (bigshifts=1, TTA disabled). The
Python ONNX waveform had 0.05% relative RMSE and 0.9999996 correlation against
PyTorch AMP. The C++ CPU STFT/ISTFT path had 2.55% relative RMSE and 0.999676
correlation against the Python ONNX waveform; the difference is the expected
CPU FFT versus CUDA/cuFFT numerical path, not a graph-layout mismatch. The
sidecar's output remains 31.9 dB above that numerical error and is suitable
for the intended accelerated desktop path.
