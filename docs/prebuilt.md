# Linux V100 prebuilt package

The v0.1.2 release includes an optional Linux x86_64 / `sm_70` / CUDA 12.8 binary package.
It targets V100, including the tested two-card 16 GB QUASAR configuration. Model weights,
the NVIDIA driver, Python and Qt are not included. This does not qualify the TP4 candidate;
TP4 work remains assigned to v0.2.0.

Use Ubuntu 22.04 or newer, or a compatible Linux distribution whose glibc meets the exact
`minimum_glibc` in `manifest.json`. A working NVIDIA driver is required; use driver 570 or newer
for the CUDA 12.8 build. Older CUDA 12.x drivers may depend on minor-version compatibility and
are not the release's tested baseline. The CUDA Toolkit, nvcc and system FFmpeg development
packages are unnecessary when using this package. The package bundles its CUDA runtime,
FFmpeg/curl dependencies, C++ runtime and license notices; it uses the host's glibc and NVIDIA
driver. CPU Vision uses statically linked single-threaded OpenBLAS.
The same release also provides the unmodified `ffmpeg-6.1.2.tar.xz` source for the bundled LGPL
runtime. Its build configuration is recorded in `tools/v100/build_dependencies.sh` in the source archive.

Extract the archive and run from any directory:

```bash
./bin/ninfer --help
./bin/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --max-context 98304 --kv-dtype int8 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --vision --vision-device cpu --vision-max-tokens 4096 --volta-long-attention \
  --pending-timeout-ms 600000
```

Keep `bin/`, `libexec/` and `lib/` together; the wrappers select the bundled libraries and accept
the same arguments as source builds. No `LD_LIBRARY_PATH` setup is needed. Download the exact
QUASAR `.ninfer` v3 artifact separately using the source README's command.

The optional launchers need Python 3.11; GUI additionally needs PyQt5 or PyQt6:

```bash
NINFER_PYTHON=/path/to/python3.11 ./bin/ninfer-tui --artifact /path/to/model.ninfer
NINFER_PYTHON=/path/to/python3.11 ./bin/ninfer-gui --artifact /path/to/model.ninfer
```

Media encoding on CPU is slower and consumes host RAM. Text/model layers, KV and MTP stay on
GPU; only BF16 visual embeddings are uploaded. CPU and GPU visual arithmetic are not
bit-identical. Long attention also changes floating-point association, so both can change
logits or generated output. No new GPQA score or losslessness claim is made.

To reproduce the package after a release build with the private dependency prefix:

```bash
.venv/bin/python3 tools/release/package_linux.py --build build-v100 \
  --output dist --version 0.1.2 --dependency-prefix build/_deps/install
```

The packager checks transitive library resolution, glibc symbols, dependency license files and
CLI help. Release qualification additionally runs real inference from a relocated extracted
directory, rather than relying on these help-only checks.
