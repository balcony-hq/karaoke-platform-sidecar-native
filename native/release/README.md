# Native release assets

Build the CPU target with either Make or CMake. The resulting executable is
not release-ready until `package_release.py` verifies its embedded model and
stages it with the target asset name:

```sh
python3 native/scripts/package_release.py \
  --cpu native/build/vocalarc-separation \
  --model native/assets/model.f32 \
  --target linux-x64 \
  --output native/release/linux-x64
```

Add `--gpu native/build/vocalarc-separation-gpu` when the optional CUDA build
is available. The output contains a manifest plus one or both self-contained
executables. The CPU asset is required by the Electron staging lock; the GPU
asset is optional and the application probes it before falling back to CPU.
