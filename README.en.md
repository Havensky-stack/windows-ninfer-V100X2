# NInfer for Windows · Dual V100

[中文首页](README.md) · [Download Windows preview](https://github.com/Havensky-stack/windows-ninfer-V100X2/releases)

This fork brings the `tuxKOH/ninfer-V100X2` inference engine to native Windows on two
Tesla V100-SXM2 16GB GPUs. Tested on Windows 11, GRID 553.62, WDDM, PCIe 3.0 x16 per GPU,
without P2P or NVLink. The existing graphics driver and system CUDA installation are preserved.

Our contributions are Windows platform I/O/memory/network support, an isolated CUDA 12.8 /
MSVC 14.44 build, exact Coder390 v3 artifact/template admission, focused regression tests,
and a portable Chinese launcher with hardware checks, resumable model download, SHA-256
verification, service lifecycle controls and a simple single-turn chat interface.

Download and extract the complete Windows ZIP, then run `NInfer.exe`. Runtime dependencies
are bundled: no Python, CUDA Toolkit or Visual Studio installation is needed. Install a suitable
NVIDIA driver first. The optional model download is 23.8GB and is not included in the ZIP.
This preview requires NVIDIA devices 0 and 1 to be V100s with at least 16GB each.

The launcher defaults to ordinary decode, INT8 KV, 128K capacity and 256-token prefill chunks.
128K passed an API smoke request with about 1.8GiB free per card. 200,000 capacity passed
a short request; 220,000 failed the current memory budget. These are capacity checks, not
full-context quality or throughput measurements. MTP failed 1/64 strict teacher-forced argmax
checks and is disabled in the launcher. DFlash/Vision are not qualified for this preview.

## Sources and attribution

- [tuxKOH/ninfer-V100X2](https://github.com/tuxKOH/ninfer-V100X2): direct upstream; dual-V100 execution, kernels, runtime and artifact foundation.
- [Neroued/ninfer](https://github.com/Neroued/ninfer): original NInfer engine.
- [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m): inherited TP2, cross-device Graph and YaRN work.
- [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100): inherited Volta implementation.
- [ww485000/ninfer-windows-v100](https://github.com/ww485000/ninfer-windows-v100/tree/v100-sm70): Windows platform adaptation reference.
- Merkyor: the published Coder390 model/container; see the [Chinese README](README.md) for its exact source.

Upstream Linux/other-GPU benchmarks belong to their source projects and are not Windows results.
See [Windows build and validation](docs/windows-toolchain.md), [launcher/release development](docs/windows-app.md),
[LICENSE](LICENSE) and [NOTICE](NOTICE).
