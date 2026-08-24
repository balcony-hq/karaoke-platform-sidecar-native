# Windows CUDA sidecar and release plan

Status: design/audit only, 2026-08-24. This document is the only file intended
to be added by this task. It does not make the existing CUDA candidate or any
desktop integration change.

## Decision in one page

Use an explicit, provider-specific Windows CUDA target built against the
official ONNX Runtime 1.28.0 CUDA 12 release asset:

`onnxruntime-win-x64-gpu_cuda12-1.28.0.zip`

The current `win32-x64` target remains DirectML. Add a separate
`win32-x64-cuda` target and never place its CUDA ORT core/provider DLLs in the
DirectML runtime directory. The desktop should select the target and pass
`--provider cuda` explicitly; `auto` is not a safe production selector on
Windows because the current implementation tries DirectML before CUDA.

The preferred speed path is already represented by an uncommitted candidate in
the worktree: CUDA kernels plus cuFFT for STFT/ISTFT, ONNX Runtime CUDA I/O
binding, and optional CUDA Graph capture. It is not currently built by either
release workflow, is not in the current commit, and has not been validated on
native Windows. It must be treated as a candidate until Linux parity/speed
gates and a native Windows DLL/provider smoke test pass.

For the exact ORT 1.28.0 Windows CUDA 12 binary inspected in this audit, the
minimal ORT-only runtime dependency closure is:

```text
vocalarc-onnx-sidecar.exe
model.onnx                         # CUDA graph, not the DirectML graph
onnxruntime.dll
onnxruntime_providers_cuda.dll
onnxruntime_providers_shared.dll
cublasLt64_12.dll
cublas64_12.dll
cudart64_12.dll
```

The candidate CUDA DSP path additionally requires `cufft64_11.dll`. It does
not require cuDNN or cuRAND unless a later model or custom kernel introduces
that dependency. The Microsoft Visual C++ x64 runtime is also required; the
recommended deployment is a centrally installed VC v14 redistributable, with
app-local deployment as a fallback only.

The CUDA provider is not a single fused kernel for the whole application. The
CUDA EP can choose fused implementations for supported ONNX operators (the
current CUDA export uses standard `Attention` at opset 24), while the audio
STFT, mask application, inverse FFT, and overlap-add remain outside the ONNX
graph. The candidate custom CUDA path fuses the surrounding elementwise work
into CUDA kernels and keeps tensors device-resident, but it still launches
several kernels and cuFFT plans. That is the right optimization boundary for
the current sidecar; claim whole-pipeline fusion only after an ORT/Nsight trace
proves it.

## Audit scope and repository state

Audited revisions:

- Sidecar repository: `96fbacc` (`Add native sidecar benchmark and parity harness`).
- Desktop repository: `133c052` (`Lock final cross-platform ONNX release`).
- Release workflow: `../.github/workflows/native-build.yml`.
- Sidecar build/packaging: `native/CMakeLists.txt`, `native/onnx_sidecar.cpp`,
  and `package_runtime.py`.
- Desktop integration: `/home/dev/code/karaoke-platform/apps/desktop/native-runtime.lock.json`,
  `scripts/prepare-sidecar-runtime.mjs`, `src/sidecar.ts`,
  `tests/packaging.test.cjs`, and `package.json`.

There are pre-existing uncommitted candidate changes at audit time:

```text
 M onnx/benchmark_native.py
 M onnx/native/CMakeLists.txt
 M onnx/native/onnx_sidecar.cpp
?? onnx/native/cuda_dsp.cu
?? onnx/native/cuda_dsp.h
```

They are intentionally not staged or changed by this document-only task. They
matter to the design because the release path currently ignores them:

- `VOCALARC_ENABLE_CUDA_DSP` defaults to `OFF`.
- The Linux workflow compiles `onnx_sidecar.cpp` directly with `g++`, so it
  cannot include `cuda_dsp.cu`.
- The Windows workflow does not pass the CUDA-DSP CMake option.
- The current release contains no CUDA Windows model or CUDA Windows bundle.

## Current implementation and release path

### Sidecar source and CMake

The committed sidecar is a C++ ONNX Runtime client. It owns the audio
front-end, bigshifts, TTA, ISTFT, and overlap aggregation; ONNX Runtime owns
model execution. Its explicit provider choices include CUDA, DirectML,
TensorRT, CoreML, OpenVINO, and CPU.

The current Windows `auto` order is effectively:

```text
DirectML -> TensorRT if present -> CUDA -> OpenVINO -> optional CPU
```

Therefore a machine with both a DML runtime and a CUDA-capable GPU can select
DML even when the CUDA bundle is present. A release target must use an
explicit provider. DirectML remains available through `--provider directml`
and the existing target.

The pre-existing CUDA candidate adds:

- `cuda_dsp.cu/.h`, using custom window/pack/unpack/mask/overlap kernels and
  cuFFT R2C/C2R plans;
- a device-resident model input and output buffer;
- ONNX Runtime `IoBinding` with CUDA memory;
- an optional `--cuda-graph on|off` setting;
- `VOCALARC_ENABLE_CUDA_DSP` in CMake; and
- CUDA architecture selection in CMake.

This is directionally correct, but it still needs validation. In particular,
the candidate creates its own CUDA stream and synchronizes before/after ORT
execution. That preserves ordering but is not the lowest-overhead design.
The final implementation should use one user compute stream for the DSP and
ORT where the ORT API permits it, retain
`do_copy_in_default_stream=1`, and use I/O binding rather than allowing ORT to
insert host/device copies. ONNX Runtime documents both the user stream option
and I/O binding as the relevant performance controls:
[CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
and [I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html).

### Existing workflow

The Linux job downloads the ORT 1.28.0 CUDA 12 SDK, but directly compiles the
C++ source. It stages Linux cuBLAS, CUDA runtime, and cuRAND libraries and
publishes individual release assets. It does not build the CUDA DSP candidate
or stage cuFFT.

The Windows job is DirectML-only:

1. Download `Microsoft.ML.OnnxRuntime.DirectML/1.24.4` from NuGet.
2. Construct a small SDK directory from its headers, `onnxruntime.dll`, and
   import library.
3. Build the generic sidecar with CMake.
4. Download `vocalarc-onnx-model-directml-win32-x64.onnx` from the current
   release.
5. Package the DirectML core and shared provider DLL.
6. Upload one `win32-x64` manifest and its four runtime/model/executable
   assets.

There is no Windows CUDA job, no native NVIDIA smoke test, no dependency
closure check with `dumpbin`, and no VC runtime installation/check. The job
also downloads a model from the release rather than exporting or producing a
CUDA Windows model. A CUDA job cannot succeed until that model is published or
the workflow gains a model-export/provisioning step.

### Runtime packager

`onnx/package_runtime.py` is intentionally a file-level staging tool. It
copies the executable, one model, the ORT core, selected provider libraries,
and explicit `--dependency` files into `runtime/` and `models/`, then records
SHA-256 hashes in `manifest.json`.

Its current CUDA provider map contains Linux names only:

```text
libonnxruntime_providers_cuda.so
libonnxruntime_providers_shared.so
```

The current code does support `--provider-library` overrides. Thus the
absolute minimum Windows workflow change can avoid changing the packager by
passing the two Windows provider DLLs explicitly. The maintainable follow-up
is a target/platform-aware provider map, so `--provider cuda --target
win32-x64-cuda` resolves `.dll` names automatically. That source change is
outside this document-only commit.

The packager checks each file against `2 GiB`; it does not impose a total
bundle limit. This is compatible with GitHub Releases: GitHub documents up to
1,000 assets per release and a per-file limit below 2 GiB, with no total
release-size or bandwidth limit in that limit description. Keep the current
individual-asset layout rather than publishing the complete ORT SDK or a
single monolithic archive:
[About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

### Desktop integration

The desktop lock currently has these relevant targets:

| Target | Provider | Current status |
| --- | --- | --- |
| `linux-x64` | CUDA | Existing release target; its lock lists CUDA libraries. |
| `win32-x64` | DirectML | Keep unchanged for compatibility and fallback testing. |
| `darwin-x64`, `darwin-arm64` | CoreML | Unrelated to this plan. |

`prepare-sidecar-runtime.mjs` derives `${process.platform}-${process.arch}`
and stages exactly one target unless macOS universal packaging is requested.
It has no CUDA variant selector. `src/sidecar.ts` always starts the sidecar
with `--provider auto`, and the packaged Windows loader relies on all DLLs
being beside the executable; it does not add a Windows DLL directory to
`PATH`. That same-directory layout is appropriate for the CUDA bundle.

The minimum later desktop change is:

1. Add `win32-x64-cuda` to the lock, leaving `win32-x64` as DirectML.
2. Add an explicit target override to the preparation script, for example
   `VOCALARC_SIDECAR_TARGET=win32-x64-cuda` or a build-script `--target`.
   The default should remain `win32-x64` so existing Windows packages do not
   silently switch providers.
3. Stage only the selected target. Do not co-locate DML and CUDA copies of
   `onnxruntime.dll`, `model.onnx`, or the sidecar in one flat runtime.
4. Pass the selected manifest provider (`cuda` or `directml`) to the sidecar
   instead of hardcoding `auto`.
5. Extend `tests/packaging.test.cjs` for the additional target and verify that
   CUDA and DirectML asset names/paths cannot be mixed.
6. Add the VC redist prerequisite to the Windows installer, or document and
   test the app-local fallback described below.

The existing DirectML implementation remains independently selectable and
testable. DirectML is also the useful non-NVIDIA fallback; ONNX Runtime
describes it as a broad DirectX 12 provider under sustained engineering:
[DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).

## Exact Windows CUDA 12 dependency plan

### Build inputs

Pin these inputs in the future Windows CUDA job:

| Input | Use | Include in release? |
| --- | --- | --- |
| Official ORT 1.28.0 CUDA 12 ZIP, `onnxruntime-win-x64-gpu_cuda12-1.28.0.zip` | Headers, `onnxruntime.lib`, core DLL, CUDA provider DLL, shared provider DLL | Only the required DLLs; never the SDK ZIP, headers, import library, or PDBs. |
| CUDA Toolkit 12.8 developer installation | Required only when compiling `cuda_dsp.cu` | No. |
| CUDA 12.8 redistributable cuBLAS archive | `cublas64_12.dll`, `cublasLt64_12.dll` | The two DLLs only. |
| CUDA 12.8 redistributable CUDA runtime archive | `cudart64_12.dll` | The DLL only. |
| CUDA 12.8 redistributable cuFFT archive | `cufft64_11.dll` for the custom DSP path | Only when the released executable links cuFFT. |
| Visual Studio 2022/MSVC x64 toolchain | Builds the sidecar and CUDA host code | No. |
| NVIDIA display driver | Supplies `nvcuda.dll` and the kernel-mode driver | No; the user must install it. |

The ORT [1.28.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)
contains a dedicated `gpu_cuda12` Windows asset. Do not substitute the
generic `Microsoft.ML.OnnxRuntime.Gpu.Windows/1.28.0` NuGet package: its CUDA
provider imports CUDA 13 DLL names in the inspected package. The current Linux
workflow's dedicated CUDA 12 asset is the appropriate versioning model for
Windows too.

For an ORT-only CUDA target, no CUDA Toolkit is needed at C++ compile time:
the sidecar calls the ORT provider API and links only the ORT import library.
The Toolkit becomes a build dependency when `VOCALARC_ENABLE_CUDA_DSP=ON`
because CMake must compile and link the `.cu` file against `CUDA::cudart` and
`CUDA::cufft`.

The [ORT execution-provider build documentation](https://onnxruntime.ai/docs/build/eps.html)
describes the Windows CUDA build prerequisites and the shared provider DLL
layout. The recommended first implementation is to consume the official
prebuilt ORT provider, not to build ORT from source in this repository.

### Static import closure of the exact ORT binary

The official `onnxruntime-win-x64-gpu_cuda12-1.28.0.zip` was inspected with a
Windows-compatible PE import viewer. The relevant imports were:

| Binary | Required non-system runtime imports observed |
| --- | --- |
| `onnxruntime.dll` | `MSVCP140.dll`, `MSVCP140_1.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, plus Windows system/UCRT APIs. |
| `onnxruntime_providers_cuda.dll` | `cublasLt64_12.dll`, `cublas64_12.dll`, `cudart64_12.dll`, `onnxruntime_providers_shared.dll`, plus MSVC/system APIs. |
| `onnxruntime_providers_shared.dll` | ORT core and MSVC/system APIs. |
| CUDA-DSP sidecar | Add `cudart64_12.dll` and `cufft64_11.dll` according to the final PE import table. |

For the inspected ORT 1.28.0 CUDA 12 provider, cuDNN, cuFFT, and cuRAND are
not ORT-provider imports. Do not ship them speculatively. Conversely, once the
custom DSP is enabled, cuFFT is a real sidecar dependency even though ORT
does not import it. CI must run `dumpbin /DEPENDENTS` (or an equivalent PE
dependency tool) against every staged DLL and executable and fail if any
non-system import is absent from the runtime directory or the documented VC
prerequisite.

Never ship `nvcuda.dll`, `dxgi.dll`, `kernel32.dll`, UCRT API-set DLLs, or
Visual Studio debug DLLs. They are either supplied by Windows/the NVIDIA
driver or are not redistributable debug artifacts.

### NVIDIA redistributable sources and hashes

Use the official [CUDA 12.8 redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0.json)
and verify the archive SHA-256 before extracting DLLs. The relevant Windows
x64 archive records observed in the manifest are:

| Component | Archive | Archive bytes | SHA-256 | Needed for |
| --- | --- | ---: | --- | --- |
| CUDA runtime | `cuda_cudart-windows-x86_64-12.8.57-archive.zip` | 3,034,859 | `2c7aa62a195d79229d4381c8bd0174a30502cf3d8124c6e94ee50a7fc8a1e9f4` | ORT and/or custom CUDA code |
| cuBLAS | `libcublas-windows-x86_64-12.8.3.14-archive.zip` | 574,528,660 | `a2f990cf61f0086d942632f2455727240baa9378c2e9fa2bdda56ef81f6cf8ad` | ORT CUDA provider |
| cuFFT | `libcufft-windows-x86_64-11.3.3.41-archive.zip` | 190,566,082 | `04969fc26520dc085665bf6256d793ddceb277bd6f2401093fded781d9fa7151` | Custom CUDA DSP only |
| cuRAND | `libcurand-windows-x86_64-10.3.9.55-archive.zip` | 61,952,615 | `239172718cca32f153ef6fd27e3950795463732b0f24df6282ba6bf207a9c29d` | Only if a later implementation imports cuRAND |

These are compressed source archive sizes, not promises about the final DLL
sizes. The workflow must measure and hash the extracted DLLs that it uploads.
Do not use Python Linux `nvidia-*` wheels as the Windows release source: their
directory layout and binary platform differ. NVIDIA documents the Windows
CUDA subpackages and redistributable components in its
[CUDA 12.8 Windows installation guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-microsoft-windows/index.html).

### Visual C++ runtime policy

The current executable and the official ORT Windows DLLs import the MSVC
runtime. A clean Windows machine cannot be assumed to have the correct version.

Recommended policy:

- Make the desktop Windows installer install or verify the latest x64 v14
  VC++ Redistributable before starting the sidecar.
- Use Microsoft's official x64 installer link
  [`https://aka.ms/vc14/vc_redist.x64.exe`](https://aka.ms/vc14/vc_redist.x64.exe),
  pin/hash it in the installer supply chain, and install quietly with the
  normal no-restart policy.
- Test the packaged application on a clean Windows VM with no Visual Studio
  installation.

Microsoft states that an application built with MSVC needs a redistributable
at least as recent as its toolset, recommends central deployment for servicing,
and documents app-local deployment as possible but less maintainable:
[latest supported VC redist](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist),
[redistributing Visual C++ files](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170),
and [determining DLLs to redistribute](https://learn.microsoft.com/en-us/cpp/windows/determining-which-dlls-to-redistribute?view=msvc-170).

If the installer cannot gain a prerequisite step, the fallback is to place the
runtime DLLs imported by the exact build beside the sidecar, typically:

```text
MSVCP140.dll
MSVCP140_1.dll
VCRUNTIME140.dll
VCRUNTIME140_1.dll
```

The exact set must come from `dumpbin /DEPENDENTS` for the released sidecar,
ORT DLLs, and cuFFT DLL. App-local deployment requires a license/REDIST audit,
hashes, and an update plan. Do not solve this by changing only the sidecar to
`/MT`: the ORT/provider DLLs still have their own MSVC dependencies, and
central deployment is the serviceable option.

### Driver and architecture compatibility

The CUDA target supports NVIDIA GPUs only. NVIDIA's CUDA 12 compatibility
guidance gives a Windows minimum for CUDA 12.x minor-version compatibility and
a higher driver requirement for the full CUDA 12.8 toolkit feature set; use
the [CUDA 12.8 release-note compatibility table](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)
when writing the product prerequisite. The application must not bundle a
driver.

For custom kernels, compile native SASS for the current CUDA 12.8-era device
families used by the product, and include a deliberate PTX fallback only when
its driver requirement is acceptable. A practical initial matrix to validate
with the installed `nvcc` is:

```text
sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120
```

Pascal/Volta (`sm_60`/`sm_70`) may be added if CUDA 12.8 cuFFT, the ORT CUDA
provider, and the model pass the same tests; they should not be declared
supported solely because a CMake architecture value is accepted. Conversely,
the current uncommitted CMake list omits `sm_90`, so Hopper must be explicitly
covered before calling the target broadly compatible. NVIDIA's
[CUDA 12.8 nvcc architecture table](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html)
is the source of truth for compiler-supported virtual architectures.

Keep the CUDA libraries dynamically linked. Fatbin code for the supported
architectures adds some bytes to the executable, but static-linking ORT or
CUDA would make the artifact much larger and complicate servicing. Strip/omit
PDBs, headers, `.lib` import libraries, CMake files, and the full Toolkit from
the release bundle.

## Target and asset layout

### Preserve DirectML

Keep these existing semantics:

```text
target:   win32-x64
provider: directml
model:    vocalarc-onnx-model-directml-win32-x64.onnx
core:     DirectML NuGet 1.24.4 onnxruntime.dll
provider: onnxruntime_providers_shared.dll
```

Do not replace this target with CUDA and do not merge its model with the CUDA
model. It remains the separately selectable Windows path for DirectX 12
hardware and for regression testing.

### Add CUDA

Use a distinct target identity, preferably:

```text
target:   win32-x64-cuda
provider: cuda
```

Suggested GitHub asset names (the staged runtime names in the manifest remain
the actual loader names):

```text
vocalarc-onnx-manifest-win32-x64-cuda.json
vocalarc-onnx-model-cuda-win32-x64.onnx
vocalarc-onnx-sidecar-win32-x64-cuda.exe
vocalarc-onnx-ort-core-win32-x64-cuda.dll
vocalarc-onnx-ort-cuda-win32-x64.dll
vocalarc-onnx-ort-shared-win32-x64-cuda.dll
vocalarc-onnx-cuda-cublasLt-win32-x64.dll
vocalarc-onnx-cuda-cublas-win32-x64.dll
vocalarc-onnx-cuda-cudart-win32-x64.dll
vocalarc-onnx-cuda-cufft-win32-x64.dll       # only with CUDA DSP
```

The bundle manifest should stage them as:

```text
runtime/vocalarc-onnx-sidecar.exe
runtime/onnxruntime.dll
runtime/onnxruntime_providers_cuda.dll
runtime/onnxruntime_providers_shared.dll
runtime/cublasLt64_12.dll
runtime/cublas64_12.dll
runtime/cudart64_12.dll
runtime/cufft64_11.dll                         # CUDA DSP build only
models/model.onnx
```

Use the actual import names inside `runtime/`; release asset names are only
the download/cache names. The manifest must list every staged file and the
desktop lock must pin every asset hash. The existing packager's
`--provider-library` and `--dependency` flags are sufficient for the first
workflow implementation.

## Minimal implementation sequence

### 1. Finish the CUDA sidecar target

The future implementation should make the existing candidate deliberate and
testable rather than silently changing the default:

1. Keep `VOCALARC_ENABLE_CUDA_DSP=OFF` for CPU/DML builds and enable it only
   for the CUDA target.
2. Build the CUDA target through CMake, not the Linux workflow's direct `g++`
   command. On Windows use a pinned VS2022 x64 generator and CUDA Toolkit
   12.8.
3. Link CUDA and cuFFT dynamically. Ensure the final PE imports are exactly
   what the package step stages.
4. Use `--provider cuda` in the CUDA bundle. Keep `--provider directml` in the
   DML bundle. Use `auto` only for development diagnostics.
5. Keep the current CUDA ONNX export initially: FP16 model weights/output,
   FP32 input transport, opset 24 standard `Attention`. Re-export only if ORT
   profiling shows an unsupported/fallback graph or parity requires a graph
   correction. The DML model remains its portable opset/profile.
6. Validate batch 1 and batch 2 separately. The candidate allocates for the
   model batch and pads a short final batch; the selected published graph must
   match its shape contract.
7. Test CUDA Graph capture after correctness. Fixed production shapes make it
   promising, but graph capture must be disabled for the first parity run and
   re-enabled only after repeated runs prove stable. ORT documents the
   `enable_cuda_graph` provider option in the
   [CUDA EP tuning options](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
8. Prefer one CUDA stream for custom DSP and ORT, or document the explicit
   stream synchronization cost. The candidate's separate-stream barriers are
   a correctness baseline, not the final speed design.

### 2. Add a Windows CUDA workflow job

Keep the existing DirectML job as its own job. Add a sibling CUDA job with
these operations:

1. Pin `windows-2022`/VS2022 and the exact ORT release asset
   `onnxruntime-win-x64-gpu_cuda12-1.28.0.zip`.
2. Extract only a build SDK in the runner. Use its headers and
   `onnxruntime.lib` to build; do not put the import library or PDB in the
   release bundle.
3. Install/use CUDA Toolkit 12.8 only for the CUDA-DSP build, then invoke
   CMake with `-DVOCALARC_ENABLE_CUDA_DSP=ON` and a validated architecture
   matrix. An ORT-only fallback build can leave the option off.
4. Download the CUDA Windows model from the same release tag, or add an
   explicit export/provisioning step. Fail early with a clear message if
   `vocalarc-onnx-model-cuda-win32-x64.onnx` is absent.
5. Verify the NVIDIA redist archive hashes, extract only the required DLLs,
   and call `package_runtime.py` with explicit Windows provider DLLs and
   dependencies. Until the packager map is made platform-aware, use:

   ```text
   --provider cuda
   --provider-library <ORT>/lib/onnxruntime_providers_cuda.dll
   --provider-library <ORT>/lib/onnxruntime_providers_shared.dll
   --dependency <CUDA redist>/cublasLt64_12.dll
   --dependency <CUDA redist>/cublas64_12.dll
   --dependency <CUDA redist>/cudart64_12.dll
   --dependency <CUDA redist>/cufft64_11.dll       # DSP build only
   --target win32-x64-cuda
   ```

6. Run a CPU protocol smoke using the same executable/model only if the model
   and provider setup allow it; the required CUDA smoke must run on a native
   Windows NVIDIA host. At minimum, send `ping` with `--provider cuda` and
   assert `CUDAExecutionProvider`; for the DSP build also assert
   `cudaDsp=true`.
7. Run `dumpbin /DEPENDENTS` on every runtime file, verify that all non-system
   imports resolve from `runtime/` or the installed VC prerequisite, and
   reject PDBs, `.lib` files, TensorRT DLLs, and accidental DML/CPU SDK files.
8. Upload each named asset separately and generate/publish the manifest only
   after all hashes and sizes have been checked.

The CUDA Windows model is the first hard provisioning blocker: the current
workflow downloads only the DirectML graph on Windows.

### 3. Update desktop selection after the release exists

Do not edit the desktop repository as part of this task. When implementing the
target, make the following coordinated changes there:

- Add a `win32-x64-cuda` lock entry with its CUDA manifest and all hashes.
- Keep the existing `win32-x64` DirectML entry and hashes intact.
- Add a target override to `prepare-sidecar-runtime.mjs`; keep the default
  platform target as DirectML for backward compatibility.
- Have `src/sidecar.ts` use the selected manifest provider rather than
  hardcoding `auto`.
- Keep all selected Windows DLLs in `sidecar-runtime/runtime`, beside the
  executable. No Windows `PATH` change should be required for this layout.
- Update the packaging test's target list and add a test that DML and CUDA
  entries have different models, core/provider assets, and provider values.
- Add/verify the VC redist installer prerequisite or document app-local files
  in the lock and installer.

If the product ultimately wants CUDA to become the normal Windows provider,
make that an explicit packaging/configuration decision. Do not change the
meaning of the existing `win32-x64` key implicitly; it is currently the DML
compatibility path.

## Linux and Windows parity of the speed path

The same CUDA DSP sidecar build should be used on both OSes where practical:

- ORT CUDA 12 provider API and model profile are shared.
- Linux needs `libcufft.so.11` in addition to the existing CUDA dependencies
  when the custom DSP is enabled; it should not retain unused cuRAND merely
  because the old packager listed it.
- Windows needs `cufft64_11.dll` in the same runtime directory.
- The CMake CUDA option must be enabled in both jobs. The current Linux direct
  `g++` command cannot produce the CUDA-DSP executable.
- The OS-specific runtime names and VC/UCRT/driver rules remain different.

The current local ignored Linux staging snapshot also deserves a release
closure check: `ldd` reported missing `libcublasLt.so.12`, `libcublas.so.12`,
and `libcudart.so.12` even though the lock/workflow list those assets. This is
not fixed here; the new workflow validation must prove the staged manifest is
self-contained before using it as the Windows template.

## Production benchmark and epsilon gate

The existing `onnx/benchmark_native.py` is the correct starting harness. It
keeps a persistent native process, uses the `profile.elapsedSeconds` value for
the primary sidecar timing, and compares the final vocals output against the
PyTorch reference. Its current defaults are one warmup, three repeats, and
relatively loose parity tolerances; those defaults are useful diagnostics, not
the release gate.

### Baseline already recorded on native Linux

For the deterministic approximately 19.99-second input with `bigshifts=2` and
TTA enabled, the existing benchmark record is:

| Runtime | Median seconds | Real-time factor | Relative to PyTorch |
| --- | ---: | ---: | ---: |
| Native PyTorch CUDA AMP | 14.7667 | 1.3537 | 1.000x |
| Python ONNX Runtime CUDA | 14.8109 | 1.3497 | 0.997x |
| Existing C++ sidecar CPU STFT/ISTFT path | about 18.33 | about 1.090 | about 0.806x |

The current C++ path is therefore about 24% slower than the PyTorch reference.
The main design target is to remove its CPU spectral transforms and repeated
host/device transfers. The recorded C++ waveform comparison against Python
ORT had relative RMSE about `0.024946` and correlation about `0.999690`; that
is not a final epsilon result. The candidate CUDA DSP output must be measured
again, not assumed correct because the kernels compile.

### Required measurement protocol

For each build/provider:

1. Use the same deterministic input or the same fixed production WAV, the same
   checkpoint/config, and the same `bigshifts=2`, TTA-on settings.
2. Run PyTorch AMP, Python ORT CUDA, and the C++ sidecar sequentially on the
   same native Linux GPU so CUDA contexts do not compete.
3. Use a persistent sidecar process and session. Use at least two warmups and
   seven measured requests; report median and p95. Report cold startup
   separately.
4. Synchronize CUDA at timing boundaries. Do not compare an asynchronous GPU
   enqueue time to a synchronized PyTorch wall time.
5. Record the sidecar `ping` provider, `dtype`, `cudaDsp`, and `cudaGraph`
   fields. A CUDA speed result from a DML/CPU fallback is invalid.
6. Compare both vocals and instrumental waveforms. Compare model-level STFT
   and mask buffers as a diagnostic so FFT/layout errors are not hidden by
   overlap aggregation.

An explicit gate can be run with the existing harness along these lines (using
the repository's actual config/checkpoint/model paths):

```text
python -m onnx.benchmark_native \
  --sidecar <cuda-sidecar> \
  --model <cuda-model.onnx> \
  --config <leap-xe-config.yml> \
  --checkpoint <checkpoint> \
  --provider cuda \
  --bigshifts 2 \
  --tta \
  --warmups 2 \
  --repeats 7 \
  --max-abs-tol 0.002 \
  --rmse-tol 0.001 \
  --relative-rmse-tol 0.001 \
  --correlation-tol 0.00001
```

The proposed release acceptance bar is:

```text
speed:        C++ CUDA median <= native PyTorch AMP median
preferred:    C++ CUDA median <= 0.95 * native PyTorch AMP median
quality:      max absolute waveform error <= 2e-3
              waveform RMSE <= 1e-3
              relative RMSE <= 1e-3
              correlation deficit <= 1e-5
coverage:     CUDA ping/provider and every staged DLL pass on native Windows
```

These are proposed epsilon values because no user-supplied epsilon was given;
they must be recorded with the reference artifacts and changed only with an
explanation. If the waveform amplitude convention changes, normalize the
error definition before changing the numerical bar. A failed quality bar is a
release blocker even when the speed result is good.

CUDA Graph should be benchmarked as a separate A/B result after the non-graph
CUDA-DSP path passes. The current benchmark harness does not expose
`--cuda-graph` to the child process, so graph results need a small later
harness change or a direct protocol runner; do not silently mix graph and
non-graph numbers.

### Native Windows validation later

Per the requested sequencing, Windows performance is measured only after the
Linux gate passes and on a native NVIDIA Windows machine. The minimum Windows
matrix is:

- Windows 10/11 x64, clean VC redist policy, and a documented NVIDIA driver;
- one Turing/Ampere GPU and one Ada/Hopper/Blackwell GPU where available;
- CUDA target with `--provider cuda`, CUDA-DSP on, and graph off/on as separate
  runs;
- existing DirectML target with `--provider directml` as a regression/control
  run;
- `ping`, production timing, output epsilon comparison, and dependency
  closure from the installed application layout.

No Windows speed claim should be made from `windows-latest` CI without a
native NVIDIA GPU; a hosted CPU/DML smoke only proves packaging/protocol
behavior.

## Key blockers and risks

1. **CUDA Windows model missing.** The release currently has a DirectML
   Windows graph only. The CUDA job must receive or generate
   `vocalarc-onnx-model-cuda-win32-x64.onnx`.
2. **CUDA DSP candidate is not committed or released.** The existing worktree
   candidate is ignored by the current workflow and has no native Windows
   build result. This document intentionally does not stage it.
3. **Current Windows path is DML-first.** `auto` can hide CUDA availability.
   Separate target identity and explicit provider selection are required.
4. **Packager CUDA names are Linux-only.** The first Windows job must pass
   explicit provider DLLs, then a platform-aware packager map can remove that
   workaround.
5. **VC runtime is unresolved.** The current Windows sidecar/ORT assets are
   not proven standalone on a clean machine. Central VC redist installation
   is the recommended product fix; app-local deployment needs licensing and
   servicing decisions.
6. **No native Windows CUDA host is currently part of validation.** Workflow
   packaging can be prepared now, but speed and provider activation need the
   later native Windows machine specified above.
7. **CUDA architecture/stream/graph behavior is unverified.** The candidate
   CMake list needs Hopper coverage and device-matrix checks; separate CUDA
   streams currently add synchronization barriers; CUDA Graph capture needs a
   fixed-shape correctness run.
8. **Existing Linux closure needs repair/verification.** The local staged
   Linux runtime had missing CUDA libraries under `ldd`, so the release
   validation must test actual staged files rather than trust the lock list.
9. **Release size is per asset, not free of packaging concerns.** The ORT CUDA
   provider is the heavy file. Keep SDK/PDB/import files out, publish separate
   DLL assets, and enforce the packager's per-file check plus an internal
   total-size budget for the desktop installer.

## Official references

- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
- [ONNX Runtime execution-provider build guidance](https://onnxruntime.ai/docs/build/eps.html)
- [ONNX Runtime DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [ONNX Runtime v1.28.0 release assets](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)
- [NVIDIA CUDA 12.8 Toolkit release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)
- [NVIDIA CUDA 12.8 redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0.json)
- [NVIDIA CUDA 12.8 Windows installation guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-microsoft-windows/index.html)
- [NVIDIA CUDA 12.8 compiler architecture table](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html)
- [Microsoft latest supported Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- [Microsoft Visual C++ redistribution guidance](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170)
- [Microsoft DLL redistribution determination](https://learn.microsoft.com/en-us/cpp/windows/determining-which-dlls-to-redistribute?view=msvc-170)
- [GitHub Releases limits and assets](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
