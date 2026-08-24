# Windows CUDA implementation

Status: implemented and accepted on a native Windows RTX 5080 on 2026-08-24.

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

Windows disables the ORT session memory-pattern planner by default. The static
plan caused ORT to reserve almost all 16 GiB of VRAM for this graph, after
which WDDM moved roughly half of the working set into shared memory. A
production request consequently took about 97 seconds. Calling
`SessionOptions::DisableMemPattern()` lets the existing CUDA arena dynamically
reuse intermediates and restores the same Flash Attention graph to Linux-class
speed. `--memory-pattern on` is retained only as a diagnostic override. Linux
keeps the ORT default because its measured path was already accepted.

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

The first Windows CUDA archive accidentally selected three ARM64 CRT DLLs from
Visual Studio's multi-architecture redistributable tree. The workflow now
selects the newest complete `x64/Microsoft.VC143.CRT` directory explicitly and
validates every staged EXE and DLL as AMD64 before creating an archive.

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

## Native Windows acceptance

The native Windows CUDA host ran the same production gate as Linux:

- 19.99-second deterministic input;
- `bigshifts=2`;
- TTA enabled;
- one warmup and at least three measured requests in one persistent process;
- median native inference within normal run-to-run variance of the recorded
  Linux CUDA baselines;
- relative RMSE at most `0.003`;
- correlation deficit at most `1e-5`; and
- provider reported as `CUDAExecutionProvider` with `cudaDsp: true`.

The accepted Windows build used an RTX 5080, driver 610.88, WDDM, CUDA 12.8.93,
and ORT 1.28.0. With one warmup and three measured requests it recorded
`14.8578632 s` median inference and RTF `1.3454x`. The checked-in Linux RTX 5080
medians are `14.6756064 s` for the ONNX sidecar and `14.7728233 s` for PyTorch
CUDA AMP, putting Windows within `1.24%` and `0.58%`, respectively. The old
Windows artifact's `97.0171289 s` median makes the fix a `6.53x` speedup.
Output from the fixed and pre-fix Windows paths was bitwise identical for the
same 881,559-sample stereo fixture.

Every Windows release still validates PE architecture and dependencies during
packaging. DirectML remains a separately built fallback archive.
