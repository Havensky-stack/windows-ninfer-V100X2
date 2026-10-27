# RAM-KV and PP2 experiments

These startup-fixed experimental routes are separate from the recommended complete-history TP2 configuration. RAM-KV is approximate attention; PP2 uses serial two-stage decode. See [CLI](cli.md) for options and [paged KV contracts](maintainer/paged-kv-cache.md#15-optional-native-ram-kv-experiment) for retrieval semantics.

## Native RAM-KV: larger windows and short inputs

Measured on the local two V100-SXM2-16GB cards, CUDA 12.8 / SM70, PCIe P2P without NVLink,
using `qwen3.8-27b/quasar-nvfp4`, INT8 group-64 KV, optimized-head greedy MTP3 and a 1024-token
prefill chunk. RAM-KV is approximate Q/K mean-vector retrieval, including during long prefill;
the ordinary route retains the complete history.

The matched long code fixture occupies 261556 input tokens at 262144 capacity and generates
512 committed output tokens. Each row is one cold request. The output-window check counts
literal credential strings, not complete code correctness; retrieval counts are selected
fact pages at the measured frontier, not a general quality score.

| GPU history window | Prefill tok/s | Committed decode tok/s | Fact pages resident | Credential strings in output |
|---|---:|---:|---:|---:|
| Complete history | 759.95 | 55.33 | 15/15 | 6/6 |
| 98304 (96K) | 903.29 | 80.71 | 13/15 | 5/6 |
| 163840 (160K) | 745.46 | 63.25 | 14/15 | 5/6 |
| 196608 (192K) | 742.62 | 55.79 | 15/15 | 6/6 |

The 192K row raises the RAM archive ceiling to 64000000000 bytes. Its planned packed archive
is only 9412018176 bytes; the whole-process cgroup remains capped at 32000000000 bytes,
with swap disabled and a measured memory peak of 7079198720 bytes. Raising the archive ceiling
does not increase retrieved history: `--ram-kv-window` controls that and consumes GPU memory
and attention work. The 192K window improves these narrow recall checks, but loses almost
all measured decode benefit over complete history. These results do not establish general
quality equivalence.

The short-input sweep fixes 180000 capacity, a 163840-token RAM-KV window, the same saved
code input IDs for each paired row and 256 timed committed decode tokens plus one prefill
output. Each route has one cold request per row; loading, graph priming and three-minute
cooldowns are excluded.

| Actual input tokens | Complete prefill tok/s | RAM-KV prefill tok/s | Complete decode tok/s | RAM-KV decode tok/s |
|---:|---:|---:|---:|---:|
| 512 | 1425.75 | 1367.38 | 130.70 | 123.24 |
| 3072 | 1650.62 | 1589.31 | 134.03 | 125.60 |
| 8192 | 1632.64 | 1572.99 | 125.42 | 117.37 |
| 16384 | 1582.70 | 1523.91 | 132.87 | 124.84 |
| 32768 | 1490.49 | 1426.89 | 120.74 | 113.71 |
| 65536 | 1320.83 | 1270.56 | 99.68 | 93.61 |
| 85000 | 1232.81 | 1183.35 | 99.61 | 93.75 |

All seven pairs reproduce the same 257 output IDs and complete MTP counters, with zero
RAM-KV transfer bytes. These inputs never fill the window, so no history is evicted;
index/page-table maintenance still costs about 6% decode throughput. Use complete KV for
these short inputs. This identity-window parity is not evidence for quality at a sparse
long-context window. These saved sweeps precede the later paired gate/up optimization;
their rates must not be substituted for that route's 85K results above.

Local reports: `profiles/bench/kvmem-native-qk/256k-{full,w96k,w160k}-tp2/report.json`,
`256k-w192k-ram64g-tp2/report.json`, and the `short-{full,w160k}-tp2/` reports in the same
directory. See the [native RAM-KV contract](maintainer/paged-kv-cache.md#15-optional-native-ram-kv-experiment)
for the approximate semantics and separate archive/process budgets.

## Experimental PP2 layer pipeline

Measured on 2026-10-07 on the local two V100-SXM2-16GB cards, CUDA 12.8 / SM70,
PCIe 3.0 x16 PHB, qualified PCIe P2P and no NVLink. The artifact is
`qwen3.8-27b/quasar-nvfp4`, at
`/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer`. This is a startup-fixed
alternative to TP2, not an automatic long-input switch or PP-prefill/TP-decode hybrid.

`--tp 1 --pp 2 --devices 0,1` places layers 0..31 on device 1 and layers 32..63,
final norm, output head and MTP on device 0. Each stage retains complete head/channel
geometry and owns its half of the text KV/GDN layers. `--pipeline-chunks 8` submits
up to eight chronological 1024-token chunks per wave. Source retirement waits only
for the residual copy, permitting the next first-stage chunk to proceed during the
previous second-stage chunk. `--pipeline-chunks 1` is the serialized control.
Decode, including MTP target verification, traverses both stages serially.

The matched sweep fixes capacity at 180000 (180032 allocated), complete INT8 group-64
KV, greedy sampling, CUDA Graphs and a 1024-token prefill chunk. None and optimized
MTP3 each run 512/3072/8192/16384/32768/65536/85000 actual input tokens. All three
layouts use the same saved code-task IDs at each occupancy, with two cold requests,
no prefix reuse or extra request warmup, one prefill output and 256 committed timed
decode outputs. Loading and graph priming are excluded. Active batches have at most
480 seconds before a three-minute cooldown, with an early rest when the next case
would exceed that window; cooldowns are excluded from request measurements.

The PP2 real-model gate checks exact graph/eager commits and sampled routing; all
64 teacher-forced positions agree with the ordinary argmax, with zero emitted-logit
deficit. The prefix gate passes repeated restore, append, checkpoint, partial-stop
and exact-frontier state controls. One cached/cold continuation diverges at output
position 4 with a 0.125 logit deficit; cold re-prefill is not bit-identical. The
24-layer, 48-value-head GDN replay geometry passes its independent recurrent-state
oracle. These checks do not establish source-checkpoint task accuracy or universal
TP2/PP2 token parity. In the occupancy sweep the two layouts can produce different
output IDs and MTP acceptance rates, so cross-layout results are descriptive, not
strict same-output or quality-equivalence claims.

PP2 supports one Text request, None/MTP and prefix reuse on SM70 Qwen3.8 NVFP4/QUASAR.
Vision, DFlash, expert storage and concurrent requests are rejected. Optional Q/K RAM-KV is
admitted. TP2 remains recommended for decode; PP2 cannot switch back to TP2 at the prefill boundary.

Reproduce the saved-input sweep with the selected Python 3.11 environment:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64" \
  .venv/bin/python tools/v100/bench_pipeline.py \
  --weights /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --bench build-v100-tp4/bench/ninfer_bench \
  --output-dir profiles/bench/pipeline/complete --corpus-dir profiles/bench \
  --backends tp2 pp2-serial pp2 \
  --prompts 512 3072 8192 16384 32768 65536 85000 --drafts 0 3 \
  --capacity 180000 --prefill-chunk 1024 --output-tokens 256 \
  --repetitions 2 --active-seconds 480 --cooldown 180
```

The script saves individual commands/reports, request phases, per-card memory, repetition
checks and PP2 wave/serial equivalence under `profiles/bench/pipeline/complete/`.
Completed cases are reused only if the saved command matches. Raw reports and corpora
are local measurement prerequisites, not committed assets.
