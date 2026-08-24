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
provider selection. CUDA releases also compile `native/cuda_dsp.cu`, which
keeps the spectral tensors on the GPU and uses:

- fused window/reflection and STFT-layout CUDA kernels around batched cuFFT;
- ONNX Runtime CUDA I/O binding with stable device input/output buffers;
- the graph's standard ONNX `Attention` nodes, which ORT dispatches to fused
  flash-attention kernels; and
- fused complex-mask and deterministic overlap-add CUDA kernels around inverse
  cuFFT.

It has no Python dependency at runtime. Build the CUDA 12 release path with:

```sh
cmake -S native -B build \
  -DORT_ROOT=/path/to/onnxruntime-gpu-sdk \
  -DVOCALARC_ENABLE_CUDA_DSP=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Leave `VOCALARC_ENABLE_CUDA_DSP=OFF` for the preserved DirectML/CoreML builds.
CUDA Graph capture remains opt-in with `--cuda-graph on`: ORT 1.28 capture is
not enabled in production because this graph performs an internal allocation
that invalidates stream capture. Stable device buffers and I/O binding remain
enabled without CUDA Graphs.

On Windows, the sidecar disables ORT's static memory-pattern planner by
default. For this fixed axial-attention graph, the static plan reserves almost
the entire 16 GiB RTX 5080 under WDDM and causes GPU-residency paging. Dynamic
CUDA-arena reuse preserves the same graph, Flash Attention path, and output
while restoring Linux-class throughput. `--memory-pattern on` remains
available as a diagnostic override; Linux keeps ORT's default planner.

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
included. The release workflow then packs the complete staged directory into
one deterministic `tar.gz` archive per target. The archive contains
`manifest.json`, `models/model.onnx`, the sidecar executable, and every
provider/runtime library required by that target. The accepted Linux
executable is 194,608 bytes. The Linux CUDA archive is self-contained and
remains below GitHub's 2 GiB per-file release limit.

Use the persistent native acceptance harness for the production comparison:

```sh
/home/dev/code/.venv/bin/python benchmark_native.py \
  --sidecar build/vocalarc-onnx-sidecar \
  --model artifacts/leap_xe/bs_roformer_leap_xe_spectral_b1_f1722.onnx \
  --config ../../Music-Source-Separation-Training/configs/leap_xe_config_voc.yaml \
  --checkpoint ../../Music-Source-Separation-Training/checkpoints/bs_leap_xe_voc.ckpt \
  --provider cuda --warmups 1 --repeats 3 \
  --output benchmark-native.json
```

Its defaults are the production settings: a deterministic 19.99-second input,
`bigshifts=2`, TTA enabled, CUDA AMP reference, minimum speedup `1.0`, relative
RMSE at most `0.003`, and correlation deficit at most `1e-5`. It exits nonzero
when either speed or parity misses the gate.

On the native Linux RTX 5080 acceptance machine (CUDA 12.8 sidecar, ORT 1.28
CUDA 12, PyTorch 2.13 CUDA AMP), a representative three-repeat run measured:

| Runtime | Median inference | Real-time factor |
| --- | ---: | ---: |
| PyTorch CUDA AMP | 14.7728 s | 1.3532x |
| C++ ONNX CUDA sidecar | 14.6756 s | 1.3621x |

That is a `1.00662x` speedup over PyTorch AMP. Request wall time, including
WAV and JSONL overhead, was also `1.00336x` faster. Final waveform parity was
`0.001060` relative RMSE, `2.13e-7` max absolute error, and `0.99999946`
correlation. The native timing covers STFT, all ONNX calls, ISTFT, overlap,
bigshifts, and TTA; process startup and WAV file I/O are reported separately.

Linux and the primary Windows `win32-x64` release use CUDA 12.8 user-space
libraries. On the same production dimensions and settings, the accepted
Windows RTX 5080 build measured `14.8579 s` median inference (RTF `1.3454x`),
within `1.24%` of the recorded Linux ONNX median and `0.58%` of the recorded
Linux PyTorch CUDA AMP median. The pre-fix Windows artifact took `97.0171 s`;
disabling its oversized ORT memory pattern produced a `6.53x` speedup with
bitwise-identical Windows output. The separately published
`win32-x64-directml` target preserves the DirectML graph and non-NVIDIA
fallback.
