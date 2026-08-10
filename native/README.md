# Native runtime internals

This directory contains the production C11 inference runtime for the bundled
`bs_leap_xe_voc` FP32 model. The CPU executable has no inference-library
dependency. The optional CUDA executable adds a fused device-resident path for
NVIDIA GPUs and falls back to the CPU graph when CUDA is unavailable or fails.

## Build targets

From the repository root:

```sh
make -C native model
make -C native
make -C native NVCC=/path/to/cuda/bin/nvcc gpu
```

Windows uses the CMake/MSVC path:

```powershell
cmake -S native -B native/build-cmake -A x64
cmake --build native/build-cmake --config Release --target vocalarc-separation
```

`embed_model.c` appends the flat model pack and a `VSCEMB01` trailer to the
executable. ELF and PE loaders ignore the trailing payload; the runtime
validates and maps it from `argv[0]`. An external model path is retained only
for developer testing of unembedded binaries.

## Execution policy

The CPU path selects AVX2/FMA at runtime when supported and otherwise uses
portable scalar kernels. Scratch buffers are retained between requests and
grow only for larger uploads. Automatic CPU and CUDA batching are bounded by
host/device memory; explicit group sizes can be supplied in protocol requests.

The CUDA provider uses FP32 weights. TF32 Tensor-Core GEMMs are enabled by
default on compute capability 8.0 and newer; set `VOCALARC_GPU_TF32=0` for
strict FP32 GEMM math. AMD, Intel, and Apple GPU providers are not included;
those systems use the CPU executable.

## Tests and tools

```sh
bash native/scripts/test_native.sh native/build/vocalarc-separation native/assets/model.f32
python3 native/scripts/benchmark.py \
  --native native/build/vocalarc-separation \
  --model native/assets/model.f32 \
  --seconds 1
```

The benchmark uses only Python’s standard library and measures native wall
time, process memory, binary size, backend, and output RMS. It does not install
packages or load a reference inference implementation.

`native/package.json` and `native/electron/` form a minimal optional Electron
smoke host. They are test tooling only; the production Electron application
downloads the release executables and starts the GPU candidate before the CPU
fallback.

## Release assets

`package_release.py` verifies the exact embedded model and the 2 GiB limit:

```sh
python3 native/scripts/package_release.py \
  --cpu native/build/vocalarc-separation \
  --model native/assets/model.f32 \
  --target linux-x64 \
  --output native/release/linux-x64
```

Pass `--gpu native/build/vocalarc-separation-gpu` to include the optional
CUDA artifact. For Windows, pass `--gpu-runtime-dll` once per required CUDA DLL
and optionally `--gpu-archive` to group the GPU executable and DLLs into a
single ZIP release asset. The generated release directory is ignored except
for its documentation; the GitHub Actions workflow publishes the verified
binaries.
