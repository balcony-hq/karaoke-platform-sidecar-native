# VocalArc native sidecar

This repository contains the self-contained native inference process used by
the VocalArc desktop application. It is a C11 JSONL process for the bundled
`bs_leap_xe_voc` model. The model is embedded into every release executable;
the Electron application does not ship Python, PyTorch, ONNX Runtime, or a
separate model file.

## Runtime

The CPU executable is the required portable fallback for Linux, Windows,
macOS, x86, and ARM builds. It selects AVX2/FMA kernels when available and
otherwise uses scalar kernels. The optional GPU executable uses CUDA/cuBLAS
for NVIDIA hardware. If its CUDA libraries or device are unavailable, the
application starts the CPU executable instead.

The process communicates through line-delimited JSON on stdin/stdout:

```json
{"id":1,"type":"ping"}
{"id":2,"type":"load"}
{"id":3,"type":"separate","inputPath":"mix.wav","vocalsPath":"vocals.wav","instrumentalPath":"instrumental.wav"}
{"id":4,"type":"shutdown"}
```

## Build on Linux or macOS

The C build uses only the platform compiler and does not install dependencies:

```sh
git lfs install
make -C native
bash native/scripts/test_native.sh native/build/vocalarc-separation native/assets/model.f32
```

For an optional NVIDIA CUDA build:

```sh
make -C native NVCC=/path/to/cuda/bin/nvcc gpu
bash native/scripts/test_native.sh native/build/vocalarc-separation-gpu native/assets/model.f32
```

## Build on Windows

Use a Visual Studio Developer PowerShell or another CMake-compatible MSVC
environment:

```powershell
git lfs install
cmake -S native -B native/build-cmake -A x64
cmake --build native/build-cmake --config Release --target vocalarc-separation
```

The resulting executable contains the model and can be copied without a
neighboring weight file.

## Release packaging

The standard-library-only packer verifies the embedded model, enforces the
2 GiB artifact limit, and writes target-specific release names and a manifest:

```sh
python3 native/scripts/package_release.py \
  --cpu native/build/vocalarc-separation \
  --model native/assets/model.f32 \
  --target linux-x64 \
  --output native/release/linux-x64
```

Add `--gpu native/build/vocalarc-separation-gpu` when a CUDA build is
available. For a self-contained Windows GPU release, pass each required CUDA
DLL with `--gpu-runtime-dll`; add `--gpu-archive` to publish the GPU executable
and its DLLs as one ZIP asset. The root GitHub Actions workflow builds and tests
Linux and Windows CPU releases; GPU artifacts are optional additions.

## Repository layout

- `native/` — C runtime, model pack, tests, release packer, and optional Electron smoke host.
- `.github/workflows/native-build.yml` — cross-platform native build workflow.

The checked-in benchmark summaries document native CPU/GPU performance. The
runtime itself has no Python dependency; `native/scripts/benchmark.py` is a
standard-library-only native timing tool.
