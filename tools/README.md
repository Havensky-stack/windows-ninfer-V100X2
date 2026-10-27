# NInfer maintainer tools

`tools/` contains the project-owner workflows for artifact conversion and inspection, independent
Python references, numerical parity diagnostics, benchmark orchestration, and serving smoke checks.
These tools are not part of the public download-and-run path; normal users should start with the
[project README](../README.md).

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Task index

| Task | Location |
|---|---|
| Build the 27B artifact | [`convert/qwen3_6_27b/`](convert/qwen3_6_27b/) |
| Build the Qwen3.8-27B artifact | [`convert/qwen3_8_27b/`](convert/qwen3_8_27b/) |
| Build the 35B-A3B artifact | [`convert/qwen3_6_35b_a3b/`](convert/qwen3_6_35b_a3b/) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Run the 27B Python reference | [`reference/qwen3_6_27b/`](reference/qwen3_6_27b/README.md) |
| Run the 35B-A3B Python reference | [`reference/qwen3_6_35b_a3b/`](reference/qwen3_6_35b_a3b/README.md) |
| Compare 27B artifact/source Vision activations | [`parity/qwen3_6_27b/`](parity/qwen3_6_27b/README.md) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Evaluate native RAM-KV on fixed long-context fixtures | [`v100/bench_ram_kv.py`](v100/bench_ram_kv.py) |
| Measure stop-aware TP2 task output on saved workloads | [`v100/bench_tasks.cpp`](v100/bench_tasks.cpp), see [`bench/README.md`](../bench/README.md#stop-aware-v100-task-probes) |
| Record read-only V100/reset telemetry | [`v100/health_log.py`](v100/health_log.py) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Measure fully cached HTTP first-text latency, fresh/reused connections | [`v100/bench_http_ttft.py`](v100/bench_http_ttft.py) |
| Start/configure a local HTTP server with PyQt | [`v100/ninfer_gui.py`](v100/ninfer_gui.py) and [`../docs/gui.md`](../docs/gui.md) |
| Start/configure a local HTTP server in a terminal or SSH session | [`v100/ninfer_tui.py`](v100/ninfer_tui.py) and [`../docs/tui.md`](../docs/tui.md) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |

## Artifact workflow

The converters consume an official local BF16 checkpoint and write one complete `.ninfer`
artifact. The paths below are placeholders for the maintainer's local checkpoint checkouts:

```bash
python3 -m tools.convert.qwen3_6_27b.convert \
  --model /path/to/Qwen3.6-27B \
  --out out/qwen3_6_27b.ninfer

python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --out out/qwen3_8_27b.ninfer

python3 -m tools.convert.qwen3_6_35b_a3b.convert \
  --model /path/to/Qwen3.6-35B-A3B-base \
  --dflash-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --out out/qwen3_6_35b_a3b.ninfer
```

Inspect either result:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

The exact source revisions, inventories, formats, and conversion recipes are recorded in
[`docs/maintainer/`](../docs/maintainer/). Published users download the completed artifacts from
Hugging Face instead of running these workflows.

## Python references and parity

```bash
python3 -m tools.reference.qwen3_6_27b \
  --weights out/qwen3_6_27b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128

python3 -m tools.reference.qwen3_6_35b_a3b \
  --weights out/qwen3_6_35b_a3b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128
```

The Python implementations are independent diagnostic references, not alternate public inference
products or generated-token goldens for the C++ engine. See the parity README for the direct 27B
artifact/source Vision comparison command.

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

`v100/bench_ram_kv.py` also records 200 ms GPU power/temperature/ECC/clocks/PCIe and per-second host
CPU/memory/temperature/AER/EDAC counters under each run's `health/`, together with PCIe replay and
traffic, live kernel/system journal events, case boundaries, cgroup memory/events/pressure and
workload thread states/wait channels. The existing one-second Engine progress logs are enabled
and copied into synced logs; this does not add CUDA synchronization. Each record is synced to
disk; `report.json` is atomically updated
after each completed case. This does not change power limits, clocks or system settings, and
cannot guarantee that a sudden reset will leave its cause in the logs. Use the collector alone
with a new output directory:

```bash
.venv/bin/python3 tools/v100/health_log.py --output profiles/bench/health-run
```

Add `--pid PID` to capture only that workload's RSS, faults/CPU counters, I/O and thread
states/wait channels once per second. Python runners can use `HealthLog.attach_pid(pid)` and
`follow_file(path, name, from_start=False)` to preserve new application events without replaying
previous resumed logs. Startup also records the GPU topology; no settings are changed.

Stop it with Ctrl-C after the workload. GPU rows use local timestamps; host/events use UTC ISO
timestamps, and the event log records the boot ID. RAM-KV archive budget and the benchmark's
32 GB server cgroup limit are separate; see [the native RAM-KV contract](../docs/maintainer/paged-kv-cache.md#15-optional-native-ram-kv-experiment).

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
