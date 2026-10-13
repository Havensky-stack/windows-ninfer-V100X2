# NInfer for Windows · Dual V100

[中文首页](README.md) · [Download Windows preview](https://github.com/Havensky-stack/windows-ninfer-V100X2/releases)

This fork brings the `tuxKOH/ninfer-V100X2` inference engine to native Windows on two
Tesla V100-SXM2 16GB GPUs. Tested on Windows 11, GRID 553.62, WDDM and TCC, PCIe 3.0 x16 per GPU,
without P2P or NVLink. The existing graphics driver and system CUDA installation are preserved.

Our contributions are Windows platform I/O/memory/network support, an isolated CUDA 12.8 /
MSVC 14.44 build, exact Coder390 v3 artifact/template admission, focused regression tests,
and a portable Chinese launcher with hardware checks, resumable model download, SHA-256
verification, service lifecycle controls and a simple single-turn chat interface.

Current source and the local 0.1.1 build use mapped-host transfers for small WDDM TP2
reductions. On the same count-to-300 request, committed ordinary decode rose from 16.49
to 28.08 tok/s; experimental MTP3 rose from 57.07 to 94.68 tok/s. Answers were byte-identical
before and after. The input occupied only 19 tokens, and MTP accepted all drafts; this is not
a long-context or general-chat claim. See [Windows performance](docs/windows-performance.md).
The published 0.1.0 ZIP does not include this optimization.

With the same driver and per-mode workload, TCC measured **36.19 / 122.69 / 189.50 committed
decode tok/s** for ordinary decode / MTP3 / DFlash7, versus **28.08 / 94.68 / 153.70** on
optimized WDDM. Each mode's output matched its WDDM output byte for byte. These are single
counting requests with 19 input and 1391 output tokens, INT8 group-64 KV and chunk 256;
ordinary/MTP capacity is 131072, DFlash capacity is 98304. They do not establish general-chat
or long-context performance. TCC still reports no P2P on this machine's fixed topology.
WDDM concurrency reached 180.23 total tok/s for ordinary decode and 207.88 for MTP3 at eight
requests with 4096 capacity each, including prefill/scheduling. DFlash multi-request startup
is not working; see the [full measurements and limitations](docs/windows-performance.md).

Download and extract the complete Windows ZIP, then run `NInfer.exe`. Runtime dependencies
are bundled: no Python, CUDA Toolkit or Visual Studio installation is needed. Install a suitable
NVIDIA driver first. The optional model download is 23.8GB and is not included in the ZIP.
This preview requires NVIDIA devices 0 and 1 to be V100s with at least 16GB each.

The launcher defaults to ordinary decode, INT8 KV, 128K capacity and 256-token prefill chunks.
128K passed an API smoke request with about 1.8GiB free per card. 200,000 capacity passed
a short request; 220,000 failed the current memory budget. These are capacity checks, not
full-context quality or throughput measurements. MTP and DFlash both failed 1/64 strict
teacher-forced argmax checks; the launcher defaults to ordinary decode. DFlash/Vision are not
qualified for this preview.

## Switching V100 modes

The tracked source tools are [enable-tcc.cmd](tools/windows/enable-tcc.cmd),
[restore-wddm.cmd](tools/windows/restore-wddm.cmd) and their shared
[set-v100-mode.ps1](tools/windows/set-v100-mode.ps1). Keep all three together. Close GPU
applications, double-click the desired CMD, approve Windows UAC, check both pending modes,
then save work and reboot. TCC cannot render Windows games; WDDM restores that mode of use.
The script detects exactly two Tesla V100s and uses their UUIDs, leaving other GPU models alone.
It never forces a switch or reboots automatically. Partial failure is reported with final states.
WDDM to TCC has been tested on GRID 553.62; switching back has not yet been retested.

Read-only status, without administrator privileges:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\windows\set-v100-mode.ps1 -Mode Status
```

Switch logs are under `%LOCALAPPDATA%\NInferV100X2\driver-mode`. TCC does not guarantee P2P:
both directions still report `IOH_TOPOLOGY_NOT_SUPPORTED` here, and NVLink is inactive.

`build-tools/` is local generated/downloaded material ignored by `.gitignore`'s `build-*/`
rule, including CUDA components and raw diagnostic results. Maintained scripts are committed
under `tools/windows/`; published measurement summaries are in `docs/windows-performance.md`.
The old local mode-switch scripts have moved to the tracked tools directory.

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
