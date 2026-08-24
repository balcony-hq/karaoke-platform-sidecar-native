# Windows CUDA implementation

Status: implemented in the build and release pipeline on 2026-08-24; native
Windows runtime/performance validation is intentionally pending a Windows CUDA
machine.

## Release targets

The release workflow publishes two independent Windows bundles:

| Target | Provider | Purpose |
| --- | --- | --- |
| `win32-x64` | CUDA 12 | Primary NVIDIA production runtime. |
| `win32-x64-directml` | DirectML | Preserved fallback and comparison runtime. |

The bundles must remain separate. They contain incompatible builds of
`onnxruntime.dll`, different model graphs, and different provider libraries.
The desktop reads `manifest.json` and passes its provider explicitly instead
of using Windows' DML-first `auto` order. A future release lock can select the
DirectML bundle with `VOCALARC_SIDECAR_TARGET=win32-x64-directml`.

The desktop lock selects the CUDA archive for `win32-x64` and keeps the
DirectML archive as an explicit fallback target. Each archive contains its
manifest, model, executable, and all provider/runtime DLLs; the desktop
verifies the archive digest and every embedded manifest file before staging.

## CUDA implementation

Windows and Linux compile the same `cuda_dsp.cu` implementation:

- batched cuFFT R2C/C2R transforms;
- fused reflection/window/layout kernels for STFT input;
- fixed GPU-resident ONNX input and output buffers;
- ONNX Runtime CUDA I/O binding;
- fused complex-mask and deterministic overlap-add kernels; and
- FP32 STFT input with FP16 model weights/output, matching PyTorch AMP's
  precision boundary.

The model retains standard ONNX `Attention` nodes. ORT 1.28 dispatches those
nodes to its fused CUDA flash-attention implementation. The whole transformer
is not one CUDA kernel, and the release does not claim that it is.

CUDA Graph capture is available only as an experimental `--cuda-graph on`
option and is off by default. On the native Linux acceptance machine, ORT's
capture encountered an internal `cudaMalloc` and failed with
`cudaErrorStreamCaptureUnsupported`, followed by capture invalidation. The
fixed-buffer I/O-binding path meets the speed target without graph capture.

## Pinned Windows dependencies

Use the official ONNX Runtime CUDA 12 archive:

```text
onnxruntime-win-x64-gpu_cuda12-1.28.0.zip
```

Do not substitute `Microsoft.ML.OnnxRuntime.Gpu.Windows/1.28.0`: the inspected
generic NuGet package imports CUDA 13 DLL names. CUDA 12 supports a wider set
of deployed drivers and matches the Linux bundle.

The primary Windows archive contains only:

```text
runtime/vocalarc-onnx-sidecar.exe
runtime/onnxruntime.dll
runtime/onnxruntime_providers_cuda.dll
runtime/onnxruntime_providers_shared.dll
runtime/cublasLt64_12.dll
runtime/cublas64_12.dll
runtime/cudart64_12.dll
runtime/cufft64_11.dll
runtime/MSVCP140.dll
runtime/MSVCP140_1.dll
runtime/VCRUNTIME140.dll
runtime/VCRUNTIME140_1.dll
models/model.onnx
manifest.json
```

The NVIDIA display driver supplies `nvcuda.dll`; never bundle it. The inspected
ORT CUDA 12 provider does not import cuDNN or cuRAND for this graph. SDK
headers, import libraries, PDBs, checkpoints, and unused provider DLLs are not
published. Release users receive one `tar.gz` archive per target rather than
individual DLLs and binaries.

The workflow creates a pinned NVIDIA-channel CUDA 12.8 build environment on a
`windows-2022` runner (CUDA 12.8's supported MSVC generation), builds the CUDA
sidecar with MSVC/Ninja, stages the exact runtime DLLs, and verifies PE
dependencies for the CUDA executable/provider with `dumpbin`. GitHub's
Windows runner has no NVIDIA GPU, so CUDA DLL initialization, protocol, parity,
and speed are reserved for native Windows GPU acceptance. Each staged file is
hashed in schema-v3 `manifest.json` and must remain below GitHub's 2 GiB
per-file limit.

## Device compatibility

The CUDA 12.8 build emits custom-kernel code for:

```text
sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120
```

This spans Pascal through Blackwell for the small sidecar-owned kernels. Actual
support also depends on the user's NVIDIA driver, ONNX Runtime's CUDA kernels,
cuFFT, available VRAM, and the model workload. Batch 2 exceeded 16 GB during
local testing, so the production artifact remains static batch 1.

## Native Windows acceptance still required

When a native Windows CUDA host is available, run the same production gate as
Linux:

- 19.99-second deterministic input;
- `bigshifts=2`;
- TTA enabled;
- one warmup and at least three measured requests in one persistent process;
- median native inference no slower than PyTorch CUDA AMP;
- relative RMSE at most `0.003`;
- correlation deficit at most `1e-5`; and
- provider reported as `CUDAExecutionProvider` with `cudaDsp: true`.

Also test startup on a clean Windows VM, inspect every executable/DLL with
`dumpbin /DEPENDENTS`, and confirm that the DirectML target still starts with
`--provider directml`. Windows speed is not inferred from Linux and should not
be published until this test is complete.
