# Serving performance

Windows/GRID measurements and the local dual-V100 latency diagnosis are documented in
[Windows performance](windows-performance.md). The inherited Linux results below are separate measurements.

## V100X2 measurement and acceptance

### Separate NVLink deployment: TP2 stage

The [NVLink deployment table](https://github.com/tuxKOH/ninfer-V100X2/blob/756d136377499c47f0938494461ec71a697f5f08/README.md#nvlink-deployment-tp2-stage) is measured on
2026-10-02 on a different four-V100-SXM2-16GB host, all-pairwise NV2, 300 W per card,
driver 580.178.04 / CUDA 12.8 / SM70. NInfer uses only devices 0,1 at TP2; devices 2,3
are idle. A verified direct NVLink mesh bypasses the PCIe-only translated-IOMMU guard,
then still passes the directed CUDA UVA D2D qualification. No system configuration changed.

The six occupied-context points use the official Qwen3.8-27B NVFP4 v3 artifact, saved final-task
code prompts with 3072/8192/16384/32768/65536/85000 actual input tokens, 180000 capacity
(180032 allocated), INT8 group-64 KV, chunk 2560, greedy optimized MTP3, CUDA Graphs, two
cold requests and no extra warmup/prefix reuse. Each produces one prefill token plus 512
committed timed decode tokens; graphs prime outside measurement. Prefill is 2101.49 tok/s
at 3072 and 1463.73 tok/s at 85000; committed decode is 111.19 and 85.53 tok/s respectively.
Every pair reproduces all 513 output IDs and speculative counters; no EOS/EOG occurs.

The same deployment also completes a capacity-only sweep from 1024 through 65536, with
512 actual input tokens, chunk 1024 and **256** timed decode tokens at every capacity. The
smaller output window leaves sufficient speculative margin in the 1024-capacity row.
All seven requested capacities are allocated exactly. Committed decode is 112.60–112.97 tok/s;
all 257 output IDs and speculative counters repeat exactly at each capacity.

The ordinary control keeps the saved 3072/85000 inputs, capacity 180000, chunk 2560 and
512 timed decode tokens, but disables speculation and omits the optimized draft head.
Its committed decode is 39.584 ± 0.004 / 28.349 ± 0.003 tok/s. MTP/plain output IDs match
at 85000, not at 3072; within each route both repetitions match. This is not universal
speculative/ordinary output parity or a scored quality evaluation.

Mean request-phase seconds for the MTP3 occupancy sweep (model load and graph priming excluded):

| Actual input | Prepare | Prefill | Decode, 512 committed tokens | Total request |
|---:|---:|---:|---:|---:|
| 3072 | 0.000042 | 1.4618 | 4.6048 | 6.0708 |
| 8192 | 0.000075 | 3.8749 | 4.6955 | 8.5763 |
| 16384 | 0.000265 | 7.9909 | 4.9015 | 12.8985 |
| 32768 | 0.000602 | 17.3586 | 5.0202 | 22.3849 |
| 65536 | 0.001199 | 40.8425 | 5.6714 | 46.5187 |
| 85000 | 0.001555 | 58.0711 | 5.9864 | 64.0614 |

The benchmark prepares saved token IDs, so its Prepare column is not raw-text tokenizer latency.
The first occupancy process reports 56.10 s model load, including 53.56 s upload, 21.20 GB
artifact bytes read and 22.46 GB host-to-device traffic; these are startup costs, not request
throughput. Total request also includes boundary/output work outside the two GPU phase timers.

The same four-card host has a separate 1Cat/vLLM control using the existing FP4 and FP8 model
directories (MTP3, maximum context 180000, 1,570-token prompt, 256 streamed output tokens):
NVFP4 measured 0.955 s TTFT and 83.03 tok/s streamed decode; FP8 measured 1.590 s TTFT and
83.23 tok/s. These short-prompt HTTP numbers are not occupied-85K or NInfer-TP4 claims. The
original OneCat UI remained on port 8888; the temporary controls used loopback 6100/6101.

Reports are on that host in `results/tp2-nvlink/`; the corresponding saved commands and
`tools/v100/bench_tp.py` distinguish occupied-context, capacity and plain-decode sweeps.
Wall decode is `n_gen / (total_seconds - first_token_seconds)` and remains separate from
phase decode. These fixed output windows are not a code-quality score, a same-host NVLink A/B,
or TP4 results. Ordinary/streaming API and turn-checkpoint prefix reuse pass a separate smoke.
The TP4 NVFP4 measurements below are separate from the completed native block-128 FP8
qualification reported with them.

The four-card NVLink host separately ran the production multi-device operator checks. The
collective suite covers direct NVLink and staged fallback, uneven gathers, broadcasts and
three graph replays. The quarter-shard BF16 projection suite covers column/row forms at
`3584×5120`, `5120×1536` and `5120×2560`, plus an exactly-once residual-add route at
`5120×1536` and `5120×4352`, for one and four active columns. All ten residual/projection
cases and their graph replays reported zero failures. This proves four-rank transport and
registered BF16 split mechanics; it does not establish a four-rank Engine schedule or a
native block-scaled FP8 decoder.

### Separate NVLink deployment: NInfer TP4

Measured on 2026-10-03 on the same four-V100/NV2 host, all devices 0,1,2,3 active. The
[TP4 occupancy table](https://github.com/tuxKOH/ninfer-V100X2/blob/756d136377499c47f0938494461ec71a697f5f08/README.md#nvlink-deployment-ninfer-tp4) uses the same saved final-task
inputs as the earlier TP2 snapshot: official Qwen3.8 NVFP4 v3, 180000 capacity (180032
allocated), INT8 group-64 KV, chunk 2560, greedy optimized MTP3, CUDA Graphs and two cold
requests per point. One prefill output plus 512 committed timed decode outputs; no prefix
reuse, extra warmup or EOS/EOG. All repetition IDs and acceptance counters agree.

| Actual input | Prepare, s | Prefill, s | Decode, 512 committed tokens, s | Total request, s |
|---:|---:|---:|---:|---:|
| 3072 | 0.000051 | 1.2899 | 4.0739 | 5.3728 |
| 8192 | 0.000108 | 3.3922 | 3.7423 | 7.1409 |
| 16384 | 0.000316 | 6.8790 | 4.1260 | 11.0149 |
| 32768 | 0.000628 | 14.3753 | 4.3404 | 18.7256 |
| 65536 | 0.001211 | 31.6256 | 4.6922 | 36.3251 |
| 85000 | 0.001540 | 43.4132 | 4.9904 | 48.4093 |

Prepare uses saved token IDs, not raw-text tokenization. Load and graph priming are outside
request timing. The first process reports 60.25 s load (56.94 s upload), 21.20 GB artifact
reads and 25.01 GB aggregate host-to-device traffic. Rank 0 reserves 6.25 GB weights,
1.73 GB sequence and 1.61 GB workspace; these are per-device arenas, not four-card sums.
Observed graph residency is 26/22/22/22 MiB across the four ranks. It is measured separately
from the 188 MiB per-device graph allowance.

At 85000 input, prefill is 1957.93 ± 2.16 tok/s, committed decode 102.60 ± 0.08 tok/s and
wall decode 102.51 ± 0.08 tok/s, with 78.56% acceptance. Against the earlier TP2 snapshot,
these are 33.76% / 19.95% prefill/decode increases. Acceptance and arithmetic grouping differ;
this is not isolated width scaling, cross-width token parity or a code-quality score.

The four-card NVFP4 gate passes exact graph/eager and sampled routing, B=2 state and all
64 strict teacher-forced positions, with zero argmax disagreements. Prefix replay/append and
partial-round truncation pass their exact state controls. A cached-vs-cold divergence at
position 28 has zero teacher-forced logit deficit; stop-vs-budget truncation yields identical
append outputs, logits and counters. Cold re-prefill is not claimed bit-identical.
Reports remain on the remote host in `results/tp4-nvfp4/`.

### Native block-128 FP8 TP4

The converted `qwen3_8_27b_fp8.ninfer` artifact was measured on the same four-card NV2 host,
with the same 180000 capacity, INT8 group-64 KV, chunk 2560, optimized MTP3 and two cold
repetitions. The six occupied-context points produced one prefill output plus 512 committed
decode outputs:

| Actual input | Prefill tok/s | Decode phase tok/s | Decode wall tok/s | Acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2415.00 ± 13.42 | 98.75 ± 0.02 | 98.59 ± 0.02 | 80.04% |
| 8192 | 2445.95 ± 5.48 | 100.16 ± 0.01 | 100.07 ± 0.02 | 80.44% |
| 16384 | 2404.93 ± 4.30 | 95.48 ± 0.01 | 95.31 ± 0.02 | 77.27% |
| 32768 | 2298.05 ± 0.04 | 89.15 ± 0.01 | 89.02 ± 0.01 | 74.68% |
| 65536 | 2089.18 ± 1.61 | 85.05 ± 0.11 | 84.99 ± 0.11 | 80.22% |
| 85000 | 1972.43 ± 2.89 | 81.53 ± 0.09 | 81.47 ± 0.09 | 80.58% |

The capacity sweep uses 512 input tokens, 256 committed outputs and chunk 1024. Decode phase
throughput is 101.12–101.55 tok/s at every requested capacity from 1024 through 65536; all
allocated capacities, output IDs and speculative counters repeat exactly. Plain controls are
40.98 tok/s at 3072 and 33.12 tok/s at 85000. The FP8 state gate and prefix gate pass on all
four ranks. Its numerical rule is ordinary-cold-controlled: three speculative teacher-forced
argmax differences (maximum deficit 0.125) are within the ordinary control's one difference
and 0.125 maximum deficit. This is not a token-by-token losslessness claim.

### P2P enabled

The current host was measured on 2026-10-01 after rebooting with `iommu=pt`: two Tesla
V100-SXM2 16 GB cards, CUDA 12.8 / SM70, 300 W per GPU, PCIe 3.0 ×16 and PHB topology.
Both GPU IOMMU domains are `identity`; NVIDIA peer read/write checks pass, and NInfer
automatically enables its existing direct-P2P UVA D2D route. This is PCIe P2P, not NVLink.
No model weights or inference algorithms changed for the MTP P2P A/B evaluation.

The official upstream v3 `qwen3.8-27b/nvfp4` artifact at
`/Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer` uses the public Engine route with TP2, devices
0,1, one active request, complete INT8 group-64 KV, greedy sampling, optimized MTP3 and CUDA
Graphs. Occupancy measurements fix capacity at 180000 (180032 allocated KV positions),
`prefill_chunk=3072`, two cold-prompt repetitions, no prefix reuse and no extra request warmup.
Graphs are primed outside measurement. Each request produces 513 tokens: one during prefill
and 512 committed timed decode tokens. Model default stopping is disabled, but no measured
window actually contains EOS/EOG; every request finishes at the output limit.
This occupancy/capacity snapshot predates the [greedy and collective update](#exact-greedy-and-small-message-p2p-update).
Its other occupancies have not been remeasured with that update.

| Actual input tokens | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | Accepted/drafted per run | Rounds | Acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 1783.96 ± 43.38 | 105.48 ± 0.01 | 105.33 ± 0.02 | 351/481 | 161 | 72.97% |
| 8192 | 1754.43 ± 18.24 | 114.49 ± 0.01 | 114.35 ± 0.01 | 364/444 | 148 | 81.98% |
| 16384 | 1699.06 ± 9.73 | 108.29 ± 0.003 | 108.21 ± 0.002 | 360/453 | 151 | 79.47% |
| 32768 | 1596.24 ± 2.94 | 101.63 ± 0.03 | 101.54 ± 0.03 | 361/453 | 151 | 79.69% |
| 65536 | 1397.83 ± 0.72 | 88.34 ± 0.04 | 88.29 ± 0.04 | 360/455 | 152 | 79.12% |
| 85000 | **1306.48 ± 3.95** | **83.18 ± 0.13** | **83.14 ± 0.13** | 360/455 | 152 | 79.12% |

Rates are mean ± sample standard deviation. Committed decode uses Engine decode-phase time;
wall decode uses `512 / (total_seconds - first_token_seconds)`. Rejected drafts do not count.
Both repetitions at every occupancy match all output IDs and speculative statistics. Prompts
preserve their final task and assistant suffix, but their source bodies and acceptance rates
differ; this is not a single-variable causal experiment in prompt length. Fixed output windows
are not scored as complete code solutions.

The separate [capacity sweep](https://github.com/tuxKOH/ninfer-V100X2/blob/756d136377499c47f0938494461ec71a697f5f08/README.md#nvfp4-v3-capacity-sweep-fixed-512-token-input) holds
occupancy at 512 tokens and uses 256 timed decode tokens, chunk 1024, one discarded warmup
and three measured requests at each of 1024/2048/4096/8192/16384/32768/65536 capacities.
It measured 105.96–106.12 wall decode tok/s, 1438.44–1443.78 prefill tok/s and 70.04% acceptance.
All capacities were allocated exactly, and all 21 runs reproduced the same 257 output IDs and
173/247 accepted drafts over 83 rounds per request. Capacity is not occupied context.

The local reports are `profiles/bench/p2p_capacity/summary.json`,
`profiles/bench/p2p_occupancy/ninfer-<input-token-count>.json` for 3K–64K, and
`profiles/bench/nvfp4_v3_85k_p2p_identity.json` for 85K. Raw reports and corpora are not committed.

#### Current request phases

Average of the two occupancy requests, in seconds. Model loading happens once per invocation,
includes upload, and is outside request timing. Preparation consumes saved raw IDs, excluding
text tokenization and media preprocessing. Graph priming is also outside these timings.

| Input tokens | Load (including upload) | Preparation | Prefill | Decode | Complete resident request |
|---:|---:|---:|---:|---:|---:|
| 3072 | 19.896 (17.323) | 0.000045 | 1.723 | 4.854 | 6.584 |
| 8192 | 19.976 (17.414) | 0.000103 | 4.670 | 4.472 | 9.148 |
| 16384 | 19.667 (17.138) | 0.000323 | 9.643 | 4.728 | 14.375 |
| 32768 | 19.560 (17.075) | 0.000605 | 20.528 | 5.038 | 25.572 |
| 65536 | 19.582 (17.101) | 0.001213 | 46.884 | 5.796 | 52.685 |
| 85000 | 19.662 (17.058) | 0.001548 | 65.061 | 6.155 | 71.220 |

At 85K, remaining request overhead is 2.922 ms. GPU0's arena reservations at 180K capacity are
10.458 GiB weights, 3.190 GiB sequence and 1.399 GiB workspace, totaling 15.047 GiB. This is
not whole-card peak VRAM including driver, graph and desktop allocations.

#### Communication and real-model regression

The matched peer-off control in the next area uses the same 85K input IDs and all inference
settings: **79.1829 → 83.1784 committed decode tok/s (+5.0459%)** and
**1297.9435 → 1306.4817 prefill tok/s (+0.6578%)**. All 513 output IDs and speculative fields
match across both paths and both repetitions. The historical 78.424 result used the pre-reboot
corpus; the rebuilt source body differs, so that old number is not the causal P2P baseline.

The 10 KiB BF16 allreduce over 500 host-synchronized iterations measured 26.6085 µs mean,
25.901 µs p50 and 43.910 µs p99 with P2P, versus 47.7961/47.181/62.424 µs with peer access off.
Mean communication latency decreases 44.33%; this is not the end-to-end inference improvement.
Both paths pass exact transfers, uneven shapes, guards and 64 consecutive rounds.

Q4_K_M's two real V100X2 CTests pass. NVFP4 v3 passes temporary copies of those same gates,
changing only the weight-identity assertion from `gguf-q4-k-m` to `nvfp4`; all numerical,
state and acceptance criteria remain unchanged. They check graph/eager output and acceptance,
MTP/plain 32-token sequences on two probes, 64 teacher-forcing positions (zero disagreements,
worst emitted-logit deficit zero), and speculative egress. Prefix checks cover repeated restore,
append, changed prefix, exact frontier, partial accepted-round stopping, normalized responses
and restored sampling. Graph/eager outputs, captured logits, acceptance and frontiers agree.

At a 3274-token prompt, three cold/cache pairs give median TTFT 3.33492 s / 0.0168689 s for
Q4_K_M and 3.01954 s / 0.0146475 s for NVFP4. Q4 has two cold-versus-cache first divergences
with zero teacher-forced logit deficit under its existing near-tie criterion; not all cold/cache
sequences are bit-identical. NVFP4 reports no such divergences in these fixtures. No criteria
were relaxed. These checks support the measured inference state and communication transition,
not a universal task-quality score or a quality comparison between NVFP4 and Q4_K_M.
The independent CUDA mathematical suites were not rerun for this communication-only change.

#### Qualified NVFP4 prefill update

Wide SM70 TP2 NVFP4 MLP projections with T≥2048 and N/K=17408/5120 or 5120/8704 use
a CUTLASS 128×256×32 threadblock instead of 128×128×32. Other geometries retain their
existing tile. The represented weights, FP32 accumulation/SwiGLU boundary and workspace
contract are unchanged. NVFP4 Linear and fused SwiGLU pass their independent mathematical
oracles without relaxed criteria, including the affected TP2 shapes.

The matched public-Engine measurement uses the official v3 artifact, the saved
`profiles/bench/v100-code-85000-iommu-pt.ids` corpus, 85000 occupied input tokens, 180000
capacity, chunk=2560, INT8 KV, TP2, greedy optimized MTP3, CUDA Graphs and 512 timed decode
tokens. Each implementation uses two cold requests, no prefix reuse and no extra warmup;
loading and graph priming are outside request timing.

| MLP tile | Prefill tok/s | Committed decode tok/s | Acceptance |
|---|---:|---:|---:|
| 128 columns, control | 1297.05 ± 2.21 | 81.630 ± 0.031 | 77.11% |
| 256 columns, delivered | 1311.92 ± 2.06 | 81.715 ± 0.022 | 77.11% |

Prefill improves **1.1466%**. All 513 output IDs and acceptance counters match across both
implementations and repetitions; each request accepts 357/463 drafts over 155 rounds.
The small decode difference is not claimed as a stable gain. Desktop VRAM occupancy made
chunk=3072/2944 fail the startup allowance during this measurement, so chunk=2560 was used
without reducing the allowance. The earlier 3072-chunk occupancy table is not the matched
control. Reports are `profiles/bench/optimization_8h/nvfp4_85k_mtp3_chunk2560_initial.json`
and `nvfp4_85k_chunk2560_tile256.json` in the same directory. Reproduction uses the MTP command
below with `--prefill-chunk 2560` and `-r 2`.

A faster 16-key softmax-update candidate changed output IDs and acceptance and did not
establish a >10% gain with small quality loss. It was removed; the target attention retains
its original FP32 accumulation order. No new lossy-attention switch is delivered.

#### Exact greedy and small-message P2P update

The same saved `profiles/bench/v100-code-85000-iommu-pt.ids` input is used for this separate
matched campaign: official NVFP4 v3, 85000 occupied tokens, 180000 capacity, chunk=2560,
complete INT8 group-64 KV, TP2, one request, greedy optimized MTP3 and CUDA Graphs. Each
implementation runs two cold requests without prefix reuse or an extra request warmup;
load and graph priming are outside request timing. Every request emits one prefill token
and 512 timed committed decode tokens. Rates are mean ± sample standard deviation.

| Implementation | Prefill tok/s | Committed decode tok/s |
|---|---:|---:|
| Control: complete target logits | 1315.36 ± 2.45 | 81.797 ± 0.076 |
| Exact shard-winner verification | 1312.31 ± 3.15 | 82.396 ± 0.037 |
| Plus parallel shard argmax | 1312.38 ± 1.68 | 82.434 ± 0.130 |
| Plus direct-peer small-message sum, delivered | 1311.95 ± 3.70 | **85.212 ± 0.062** |

Combined decode gain is **4.1748%**. The collective step improves decode **3.3699%** over
the preceding route. The isolated argmax step is within end-to-end noise, and no prefill
improvement is claimed. All eight requests produce the same 513 IDs and complete speculative
fields: 357/463 drafts accepted over 155 rounds, 77.1058% acceptance, positions 135/117/105,
zero fallback steps. The fixed output window contains no EOS/EOG. The older chunk=3072 sweep
has different rounding/acceptance and is not a matched control for these changes.

Greedy MTP verification now uses the same exact shard-winner route as DFlash: each rank
computes its represented BF16 logit maximum, equal maxima choose the lower global ID, and
only FP32 values/I32 indices and final IDs cross the link. At SM70's 124160-row vocabulary
shards, widths 1–64 use parallel row tiles plus selected-value extraction instead of one CTA
scanning the whole shard. A cold-L2 captured C4 candidate comparison measured 104.448 →
14.336 µs for shard 0 and 94.208 → approximately 12 µs for shard 1. Final public-Op rotating
input measurements are 11.20/10.68 µs respectively, under different cache/timing conditions.
These are operator results, not an isolated end-to-end speedup claim. Sampled/mixed batches
keep full logits and rank-local penalty counters. Both graph topologies are warmed/captured
at startup, so request sampling changes do not trigger capture.

Qualified SM70 P2P allreduces of at most 81920 bytes read both immutable operands and write
only rank-local sum staging. Both read completions retire before a local D2D copy publishes
the result into the original operands. This preserves the FP32 two-operand sum and one BF16
RNE store, requires no extra device storage, and retains the three-phase cross-device order.
Wide prefill and actual peer-off calls retain DMA plus local combine. A 64-round captured
public-collective candidate comparison measured 15.136 → 9.360 µs at 10 KiB, 18.832 →
13.216 µs at 40 KiB and 22.848 → 18.880 µs at 80 KiB. Large peer-read payloads were slower,
so the delivered dispatch does not use them. These graph-chain measurements are not the
host-synchronized 500-iteration transport measurements above.

| Delivered request phase | Unprofiled mean |
|---|---:|
| Model load, once | 20.019 s, including 17.297 s upload |
| Prompt preparation, saved IDs | 0.001550 s |
| Prefill and first token | 64.789 s |
| Decode, 512 committed tokens | 6.009 s |
| Remaining request overhead | 0.003267 s |
| Complete resident request | 70.803 s |

Wall decode averages 85.169 tok/s. The measured complete-request gain is only 0.116% versus
the 70.885 s control because prefill dominates; this is not claimed as a stable total-request
speedup. Neither Vision nor prefix reuse is active in this workload.

Independent argmax/state checks cover valid-domain padding, negative winners, cross-shard
ties, zero/short drafts, ignored greedy penalties, preserved inputs and eager/Graph replay.
Allreduce passes its independent FP64 oracle on eager/captured routes, odd shapes, the
40959/40960/40961-element dispatch boundary and skewed 64-round chains, with actual P2P
enabled and disabled inside the test process. Linear, LinearAdd, SwiGLU and output-head
split checks pass. NVFP4 real-model and prefix gates pass, including 64 teacher-forcing
positions, greedy → sampled → greedy route switches with penalties, two-row all-greedy/mixed
batch Graph/eager output and acceptance equality, and peer-egress consistency. No numerical
criteria were relaxed. These checks support this transition, not universal model-quality parity.

One broader MTP split suite remains blocked in its pre-existing groupwise Q4/Q5 snapshot
case: its SM70 shard dispatch selects the Ampere-only grouped-MMA kernel, whose trap was
confirmed by compute-sanitizer. The preceding W8 MTP/FP8/NVFP4 cases pass. This path is not
the NVFP4 or GGUF-derived identity used here; it is not counted as a passing regression or
changed by this optimization. Test-only fixes correct stale BF16 registry expectations,
newer-device A4/A8 invocation and current-device selection without weakening comparisons.

Local paired reports are `profiles/bench/p2p_mtp_round2/{baseline_85k,greedy_winners_85k,
greedy_tiled_85k,peer_sum_85k}.json`; they and their corpus are not committed. Reproduction:

```bash
taskset -c 0-15 build-v100/bench/ninfer_bench \
  --weights /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --corpus profiles/bench/v100-code-85000-iommu-pt.ids -pg 85000,512 \
  --max-ctx 180000 --prefill-chunk 2560 --kv-dtype int8 --tp 2 --devices 0,1 \
  --spec mtp --draft-tokens 3 --lm-head-draft --warmup 0 -r 2 \
  --capture-generation -o json --output-file profiles/bench/mtp85k.json
```

#### QUASAR v3: Volta QPN projections

The separate `qwen3.8-27b/quasar-nvfp4` trial uses the published QUASAR NInfer v3 artifact
on this two-card PCIe-P2P host, without NVLink. QUASAR is QAT, not mathematically lossless
compression. Its stored NVFP4 codes/scales and FP32 activation divisors remain unchanged;
this does not establish BF16-equivalent artifact quality. The profile is text-only and loads
zero Vision objects onto either GPU; validation-only Vision data remains in the original file.

SM70 Attention/GDN input and output projections now share the load-time QPN layout with MLP.
Narrow inputs write final output planes directly. Wide projections decode weights once per
call for CUTLASS and distribute the BF16 result. There is no second resident weight copy,
A4 activation quantization or attention-history pruning. Independent FP64 input-projection
oracles cover real TP1/TP2 geometries and QPN/CUTLASS boundaries. Public Engine MTP and DFlash
gates each pass 64 strict teacher-forced positions and graph/eager sampling checks.

Measured on 2026-10-03: two V100-SXM2 16 GB, 300 W/card, CUDA 12.8 / SM70, saved code inputs,
chunk 2560, complete INT8 group-64 KV, greedy optimized-head MTP3, CUDA Graphs and no prefix
reuse. Two measured requests per row; 3072-input rows have no extra warmup, and 85000 input
follows one complete warmup. Each generates one prefill output and 512 timed committed decode
outputs. Load and graph priming are outside timing. Decode is committed output divided by
wall time after the first token; rates are mean ± sample standard deviation.

| Route | Actual input / capacity | Prefill tok/s | Committed wall decode tok/s | Acceptance |
|---|---:|---:|---:|---:|
| Before QPN input integration | 3072 / 8192 | 376.71 ± 0.77 | 102.02 ± 0.06 | 85.78% |
| QPN / MTP3 | 3072 / 8192 | 1751.55 ± 30.90 | 139.80 ± 0.08 | 85.19% |
| QPN / MTP3, warmed | 85000 / 180000 | 1315.28 ± 2.61 | 100.34 ± 0.05 | 86.62% |

The matched short-input gain is 4.65× prefill / 1.37× decode. Warmed 85K mean prefill,
decode phase and total request times are 64.625 / 5.100 / 69.729 seconds; the decode-phase
rate is 100.40 tok/s. MTP weights occupy 8.66 GiB per card. Within each route, repetitions
reproduce all output IDs and speculative counters. Before/after short-input trajectories
first differ at output position 270; numerical qualification does not prove unchanged
end-to-end quality. No BF16 quality score or full-180K/256K occupied-context result is claimed.
Reports: `profiles/bench/quasar-qpn-projections/{before-mtp3,after-mtp3,after-85k-mtp3-warm}.json`.

#### QUASAR v3: final long-context decode evaluation

Measured on 2026-10-13 on two V100-SXM2-16GB cards, CUDA 12.8 / SM70, qualified
PCIe P2P without NVLink. The artifact is `qwen3.8-27b/quasar-nvfp4`; the code input
is the saved `profiles/bench/v100-code-85000-iommu-pt.ids`. Both routes use 85000
actual input tokens, 180000 capacity (180032 allocated), complete INT8 group-64 KV,
chunk 2560, greedy optimized-head MTP4 and CUDA Graphs. Each has two measured cold
requests, no prefix reuse or extra request warmup, one prefill output and 512 timed
committed decode outputs. Loading and graph priming are outside request timing.
Rates are mean ± sample SD; wall decode is `512 / (total_seconds - first_token_seconds)`.

| Route | Prefill tok/s | Committed wall decode tok/s | MTP acceptance |
|---|---:|---:|---:|
| Default complete-history MTP4 | 1313.37 ± 3.89 | 126.894 ± 0.028 | 79.02% |
| `--volta-long-attention` / MTP4 | 1307.41 ± 2.96 | 143.459 ± 0.013 | 76.74% |

| Route | Prepare, s | Prefill, s | First token, s | Decode phase, s | Total request, s |
|---|---:|---:|---:|---:|---:|
| Default | 0.001544 | 64.719 | 64.721 | 4.031 | 68.756 |
| Experimental attention | 0.001593 | 65.014 | 65.016 | 3.566 | 68.585 |

Prepare uses saved IDs, not raw-text tokenizer latency. First-token time includes
prefill; the phase columns are not all additive. Experimental attention improves
decode 13.05% in this pair; prefill variation leaves total request time almost equal.
Both routes repeat all 513 output IDs and speculative counters and match their own
previously qualified controls. Their trajectories differ across profiles. The
requested 150 tok/s was not reached; the user accepted this result and ended that goal.

The retained default optimizations share QK/softmax work across dimension warps,
decode packed INT8 KV cooperatively, stage split-output reduction while retaining
its ascending FP32 association, and precast selected narrow NVFP4 activations.
Paired gate/up warp groups retain FP32 SiLU/product before one BF16 rounding.
The W8 vocabulary head has an exact load-time code/scale permutation, without
requantization or a second resident copy. Its narrow consumers, the W8 MTP
projections and the Q4 proposal head use measured unroll/direct-read paths.
Single-query attention uses eight-row Volta QPN tiles with the default eight-key
softmax and FP16 contribution rounding; five-query attention removes the shared
QK-score handoff while retaining its accumulation associations.

These routes have independent FP64 decoded-weight projection/attention oracles,
exact layout/codec checks, cache/guard checks and CUDA Graph replay evidence.
Twenty-five teacher-forced 85K vocabulary checkpoints of the default attention
changes equal the saved reference exactly. Real-model default and experimental
MTP4 gates separately check 64 graph/eager commits, acceptance counters, partial
commit/fallback and peer egress. Exactness is established for the represented
comparisons, not universal source-model quality.

The optional `--volta-long-attention` flag is startup-fixed and defaults off.
It supports SM70 27B TP2/PP1, one Text/None/MTP request and complete INT8 KV;
Vision, RAM-KV and DFlash are excluded. Above 16390 keys it selects FP32 PV
accumulation for one/five/six-query paths. One/six-query kernels use eight-row
QPN tiles and 32-key online-softmax updates; five-query verification combines
sixteen keys per update. Above 65536 keys those widths target about 1072
keys per split, reducing staging and final-reduction overhead. At 85K the
default has about 355 splits, versus 80 in the experimental profile. Every
causally visible historical KV entry remains included. Wide prefill is unchanged.

Full 85K attention passes the independent FP64 relative-L2 criterion of 0.00315,
with exact packed cache bytes and graph/eager boundary checks. However, the
experimental MTP4 profile fails the separate near-lossless full-vocabulary gate:
25/25 argmax matches, minimum cosine 0.998905, maximum KL 0.007842 nats, against
bounds of cosine ≥0.9999 and KL ≤0.005. More FP32 accumulation and changed
partition/softmax association do not imply bit identity or task-level equivalence.
This is why the profile remains opt-in.

Final local reports are `profiles/bench/quasar-150-tp2/final-20261013-mtp4-{default,long}.json`.
Reproduce either route with the public benchmark; add `--volta-long-attention`
for the experimental row:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64" \
  build-v100-tp4/bench/ninfer_bench \
  --weights /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --max-ctx 180000 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --corpus profiles/bench/v100-code-85000-iommu-pt.ids \
  -pg 85000,512 --warmup 0 -r 2 --capture-generation -o json \
  --output-file profiles/bench/quasar-final-default.json
```

##### Local GPQA-Diamond comparison: 104 paired questions

This is an **85K unrelated-code occupancy variant**, not standard short-context
GPQA-Diamond. A frozen data-independent shuffled order and identical question/
answer-choice prompts are shared by all four routes. Each input occupies 85000
tokens; capacity is 131072, output budget 32768, chunk 2560, greedy MTP4,
`reasoning_effort=xhigh`, complete INT8 KV and no prefix reuse. Embedded frontend
resources match across artifacts. Scoring uses the final `ANSWER: A/B/C/D`;
missing or truncated answers count as incorrect in the full score.

| Local artifact / profile | Correct / 104 | Accuracy | Output-budget truncations |
|---|---:|---:|---:|
| QUASAR, default attention | 88 | 84.62% | 11 |
| QUASAR, experimental attention | 90 | 86.54% | 8 |
| Official NVFP4 | 90 | 86.54% | 5 |
| GGUF-derived Q4_K_M | 87 | 83.65% | 5 |

Default → experimental has two losses and four gains, +1.92 percentage points;
paired McNemar exact two-sided p=0.6875. There is no detected accuracy decrease
in this sample, but no statistical noninferiority or universal quality-equivalence
claim follows. Truncation affects the score; no BF16/FP8 local reference was run.
The user accepted this comparison and stopped at 104 common questions. The
remaining 94 questions were not run; this is not a completed 198-question score.

Runner: `tools/v100/bench_gpqa.py` and the public-Engine `ninfer_v100_gpqa` target.
Saved local results: `profiles/bench/gpqa-85k-matrix/{default,long,nvfp4,q4,summary}.json`.
The summary preserves `complete=false`.

##### Original short JSONL DFlash7 retest

The 2026-10-13 retest uses the original **32-record arithmetic JSONL generation**
case that measured 249.00 tok/s, not the demo-page JSON→JSONL conversion prompt.
Input is 118 tokens; capacity 98304, chunk 1024, complete INT8 group-64 KV,
greedy full-head DFlash7, CUDA Graphs, no prefix reuse and model-default stopping.
One full warmup precedes two measured requests.

| Measurement | Prefill tok/s | Committed wall decode tok/s | Acceptance |
|---|---:|---:|---:|
| Original 2026-10-03 | 564.22 ± 0.33 | 249.00 ± 0.08 | 99.05% |
| Final 2026-10-13 | 580.95 ± 0.01 | 255.31 ± 0.11 | 99.05% |

Each final answer publishes 1190 tokens, with 1189 timed after the first, and
stops naturally. All 32 record values, arithmetic and field order pass exact
checks. All output IDs and speculative counters equal the original measurement.
Committed throughput is 2.53% higher; **300 tok/s was not reached**. This is
high-acceptance structured generation, not a general code/prose speed claim.
The experimental attention flag is off and unsupported for DFlash.
Report: `profiles/bench/quasar-150-tp2/final-20261013-jsonl-native-dflash7.json`;
reproduce with `ninfer_v100_task_bench`, task `structured_jsonl`, occupancy `native`
and the existing `profiles/bench/other_inputs_20261001/cases.json`.
Detailed health logs accompany all three final cases: no reset, zero volatile
uncorrected ECC, maximum observed GPU temperature 64°C and host CPU use below 37%.
This short run does not establish long-duration hardware stability.

Release checks rebuild the CLI, server and affected benchmarks; artifact, CLI/server
option and report-contract tests pass. GPQA scoring tests and schema-v15 matrix
consumption pass, including both saved final reports. Current 85K MTP4 graph/eager
state checks pass with identical commit IDs/counters and zero peer-egress mismatches.
A broader public FP64 attention rerun was stopped by its wrapper's 560-second
timeout while computing the six-query reference, without a numerical failure being
reported; no completed verdict is claimed for that rerun. The affected five-query
operator and full-vocabulary comparisons above retain their completed evidence.

#### QUASAR v3: stop-aware task matrix

Measured on 2026-10-03 through the public Engine using the same frozen token IDs as the
official-v3 non-code sweep below. Four tasks cover a three-chapter Chinese story requesting
900–1200 Chinese characters and fixed literal items; six-section English-to-Chinese translation
with at least 20 glossary entries; exactly 32 prescribed JSONL records; and a five-slot logic
puzzle with a complete service/colour/port mapping and final `CHECK=4606`.

Both backends use TP2 devices 0,1, **98304 context/KV capacity**, chunk 1024, complete INT8
group-64 KV, greedy CUDA Graphs, zero presence/frequency penalties, disabled thinking/prefix
reuse and zero Vision GPU objects. MTP uses three drafts and the optimized proposal head;
DFlash uses seven drafts and the full proposal head. Every case has one full warmup; native
inputs have two measured repetitions per backend, all other points only one. All 80 measured
requests stop naturally at their first model end token before the 2048-output budget. Load,
graph priming, warmup and cooldown are excluded. Committed wall decode is
`(published_tokens - 1) / (total_seconds - first_token_seconds)`; phase decode is retained
separately in the report. Counts include the prefill output and terminating model token.

The actual input counts are 129 / 395 / 118 / 417 for native story / translation / JSONL /
logic, then exactly 1024 / 2048 / 4096 / 8192 / 16384 / 32768 / 65536 / 85000 for each task.
Distinct technical-document backgrounds precede the unchanged final task/template. Visible
background varies with length, so this is not a single-variable length or retrieval-quality
experiment. **D/M means DFlash7 / MTP3** in every paired table. Native decode shows mean ±
sample standard deviation; other entries are single measurements, not stable means.

##### Chinese story

| Actual input tokens | Prefill tok/s D/M | Committed wall decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical IDs |
|---:|---:|---:|---:|---:|:---:|
| 129 (native) | 631.17 / 632.74 | 56.45 ± 0.002 / 89.58 ± 0.007 | 11.43% / 39.80% | 990 / 1152 | No |
| 1024 | 1587.24 / 1676.19 | 59.98 / 89.34 | 13.73% / 40.11% | 1202 / 1202 | Yes |
| 2048 | 1588.60 / 1679.56 | 58.38 / 85.06 | 13.80% / 37.80% | 1034 / 1034 | Yes |
| 4096 | 1576.10 / 1666.56 | 53.75 / 83.76 | 11.42% / 36.47% | 934 / 934 | Yes |
| 8192 | 1548.49 / 1638.20 | 55.62 / 82.65 | 12.90% / 37.00% | 940 / 940 | Yes |
| 16384 | 1500.94 / 1583.26 | 58.74 / 85.15 | 15.47% / 41.59% | 952 / 952 | Yes |
| 32768 | 1415.61 / 1481.05 | 48.32 / 77.06 | 13.08% / 40.12% | 726 / 726 | Yes |
| 65536 | 1257.29 / 1322.89 | 41.32 / 64.82 | 13.72% / 38.96% | 796 / 796 | Yes |
| 85000 | 1176.69 / 1226.43 | 39.26 / 61.28 | 14.63% / 39.70% | 1077 / 1077 | Yes |

##### English → Chinese translation

| Actual input tokens | Prefill tok/s D/M | Committed wall decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical IDs |
|---:|---:|---:|---:|---:|:---:|
| 395 (native) | 1201.31 / 1238.48 | 162.90 ± 0.021 / 143.66 ± 0.044 | 60.36% / 83.74% | 998 / 858 | No |
| 1024 | 1587.78 / 1682.12 | 154.45 / 143.26 | 57.56% / 84.42% | 1027 / 1027 | Yes |
| 2048 | 1581.97 / 1678.69 | 149.46 / 140.75 | 57.28% / 84.32% | 1012 / 1012 | Yes |
| 4096 | 1570.32 / 1665.85 | 144.89 / 138.87 | 55.06% / 82.24% | 899 / 899 | Yes |
| 8192 | 1545.72 / 1637.66 | 145.07 / 139.49 | 56.61% / 85.52% | 924 / 975 | No |
| 16384 | 1505.51 / 1582.73 | 124.65 / 129.88 | 48.96% / 80.69% | 757 / 757 | Yes |
| 32768 | 1413.89 / 1480.07 | 124.26 / 124.26 | 56.03% / 85.08% | 882 / 882 | Yes |
| 65536 | 1264.06 / 1322.04 | 98.82 / 102.86 | 52.81% / 81.32% | 883 / 883 | Yes |
| 85000 | 1184.80 / 1232.95 | 87.06 / 93.29 | 49.87% / 78.03% | 723 / 723 | Yes |

##### 32-record JSONL

| Actual input tokens | Prefill tok/s D/M | Committed wall decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical IDs |
|---:|---:|---:|---:|---:|:---:|
| 118 (native) | 563.48 / 567.46 | 247.73 ± 0.134 / 163.33 ± 0.032 | 99.05% / 99.89% | 1190 / 1190 | Yes |
| 1024 | 1588.03 / 1680.68 | 242.15 / 161.98 | 99.05% / 100.00% | 1190 / 1190 | Yes |
| 2048 | 1581.29 / 1679.06 | 232.84 / 158.92 | 98.30% / 99.78% | 1190 / 1190 | Yes |
| 4096 | 1574.58 / 1665.46 | 234.86 / 158.91 | 99.05% / 100.00% | 1190 / 1190 | Yes |
| 8192 | 1549.08 / 1635.96 | 229.81 / 156.56 | 98.30% / 99.78% | 1190 / 1190 | Yes |
| 16384 | 1504.45 / 1581.44 | 221.85 / 151.38 | 98.30% / 99.89% | 1190 / 1190 | Yes |
| 32768 | 1413.90 / 1472.01 | 198.65 / 139.45 | 98.30% / 99.78% | 1190 / 1190 | Yes |
| 65536 | 1261.11 / 1318.70 | 165.74 / 119.52 | 98.30% / 99.78% | 1190 / 1190 | Yes |
| 85000 | 1180.44 / 1227.40 | 152.72 / 111.73 | 98.30% / 99.78% | 1190 / 1190 | Yes |

##### Five-slot logic puzzle

| Actual input tokens | Prefill tok/s D/M | Committed wall decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical IDs |
|---:|---:|---:|---:|---:|:---:|
| 417 (native) | 1240.28 / 1283.12 | 204.68 ± 0.034 / 152.93 ± 0.042 | 80.06% / 91.83% | 1301 / 1301 | Yes |
| 1024 | 1594.90 / 1681.08 | 185.00 / 143.35 | 71.69% / 84.41% | 656 / 656 | Yes |
| 2048 | 1581.55 / 1678.54 | 193.55 / 151.54 | 78.17% / 92.79% | 699 / 699 | Yes |
| 4096 | 1569.54 / 1664.90 | 151.70 / 137.03 | 58.27% / 81.33% | 514 / 514 | Yes |
| 8192 | 1547.80 / 1636.89 | 195.49 / 143.77 | 81.29% / 89.14% | 670 / 642 | No |
| 16384 | 1502.74 / 1580.28 | 177.22 / 139.75 | 75.55% / 89.77% | 699 / 697 | No |
| 32768 | 1412.87 / 1476.17 | 162.39 / 132.71 | 77.59% / 93.58% | 612 / 612 | Yes |
| 65536 | 1258.20 / 1316.37 | 135.01 / 112.12 | 77.25% / 91.89% | 693 / 693 | Yes |
| 85000 | 1186.82 / 1238.87 | 127.43 / 105.94 | 79.14% / 93.06% | 655 / 655 | Yes |

##### Complete requests and output qualifications

| Task at 85000 input tokens | Prefill seconds D/M | Decode wall seconds D/M | Total resident request seconds D/M |
|---|---:|---:|---:|
| Chinese story | 72.237 / 69.307 | 27.404 / 17.560 | 99.642 / 86.868 |
| English → Chinese translation | 71.742 / 68.940 | 8.293 / 7.740 | 80.036 / 76.681 |
| 32-record JSONL | 72.007 / 69.252 | 7.786 / 10.642 | 79.794 / 79.895 |
| Five-slot logic puzzle | 71.620 / 68.611 | 5.132 / 6.173 | 76.754 / 74.785 |

At 85K, DFlash improves JSONL/logic decode by 36.69% / 20.28%, with identical complete output
IDs; story/translation decode is slower. DFlash prefill is slower throughout the matrix.
JSONL total request time differs by only 0.101 seconds (about 0.13%), within what a single
pair can establish; this is not claimed as a stable whole-request gain. Logic is slower
overall despite faster decode. The 247.73 tok/s native JSONL rate is high-acceptance structured
generation, not ordinary translation, general code or a 12K-input JSONL translation speed.

All ten JSONL completions per backend pass exact count/value/arithmetic/field-order checks.
All ten logic answers per backend contain the complete expected mapping and `CHECK=4606`,
including separate assignment lists and combined mappings. Translations retain six headings
and at least twenty glossary entries; this is not an independent translation-accuracy score.
All stories retain the three chapter headings and required literal items, but only 8K, 16K
and 64K satisfy the 900–1200 Chinese-character bound. The 85K story contains 1228 Chinese
characters in both routes. Native repetitions reproduce output IDs and speculative counters
within each backend. Only 31 of 36 task/length pairs match complete IDs: native story/translation,
8K translation/logic and 16K logic differ. All 85K pairs and every JSONL pair match. No universal
token parity, artifact-quality score or quality-losslessness claim follows from these checks.

MTP reserves 8.66 GiB weights per card, DFlash 10.36 GiB. At this capacity the reported per-card
KV/workspace reservations are 1.64/0.74 GiB for MTP and 1.625/0.727 GiB for DFlash; these
figures are GPU arenas, not process RSS or aggregate two-card memory.

Three unexpected host resets interrupted this campaign. Completed results survived and only
missing repetitions were resumed. The final continuation uses 600-second active windows with
a 120-second next-request margin and **180-second idle periods**, at completed-request boundaries
and between backends. Observed DFlash active blocks before rest were 505.27 / 487.91 / 553.25 s.
Rest does not enter request times. The third reset still occurred after the three-minute-rest
policy, during the last logic measurement's prefill; cooldown is not a demonstrated reset fix.
The last saved samples showed 54/59°C and zero corrected/uncorrected GPU ECC, with no recorded
collector error or kernel Xid explaining the reset. Root cause remains undetermined; completed
measurements do not establish sustained hardware stability.

Reports and corpus remain local under `profiles/bench/quasar-qpn-projections/{all-mtp3,
all-dflash7,matrix-summary}.json` and `profiles/bench/other_inputs_20261001/cases.json`.
The read-only health logs preserve 200 ms GPU samples, per-second host/process/thread and Engine
progress, request/phase/cooldown timestamps, PCIe AER/EDAC and kernel events with synced writes.
No power limits, clocks or system settings were changed. Reproduction uses the public
[`ninfer_v100_task_bench`](../bench/README.md#stop-aware-v100-task-probes); raw outputs, telemetry,
model files and the local filming demo are not committed.

#### DFlash2 v3: separate capacity

The official v3 artifact's optional five-layer drafter is bound for text-only TP2 execution,
preserving its W8 projections and BF16 auxiliary tensors. All five draft layers use local
sliding-window attention; the unused draft Full-KV pool is not allocated for this all-local
27B variant. The main target KV remains complete. The verified profile is 98,304 capacity with
a 1,024-token prefill chunk; 180,000-capacity DFlash is not claimed on two 16-GiB cards.
Per-device sequence storage is 1,974,840,832 bytes and workspace is 780,790,272 bytes.

The delivered implementation reduces two full-vocabulary transfers. Greedy target verification
computes an argmax value/ID on each vocabulary shard, gathers those pairs, merges the global
ID and broadcasts it. Sampling still uses full target logits. DFlash2 proposal selection
computes each shard's exact top-16 keys (ordered represented BF16 logit plus complemented
global token ID), gathers a `[32,columns]` image and merges the global top-16 before the same
lattice walk. A global top-16 member must occur in its shard's top-16, so this reduces transfer
volume without pruning any previously eligible candidate. Neither change alters target KV,
weight precision, attention history or the lattice scoring formula.

On the saved `profiles/bench/optimization_8h/lru_code_85000.ids` corpus, actual occupancy is
85,000 input tokens. The following are **single cold requests**, not repeated means: greedy,
INT8 group-64 KV, TP2, DFlash7, CUDA Graphs, no prefix reuse and no extra request warmup.
Graph priming and model loading are outside request timing. Each request emits one additional
prefill token before the timed committed decode window.

| Timed decode tokens | Prefill tok/s | Committed decode tok/s | Accepted/drafted | Rounds |
|---:|---:|---:|---:|---:|
| 512 | 1185.52 | **99.93** | 422/623 (67.74%) | 90 |
| 2048, forced window | 1182.63 | **95.87** | 1674/2612 (64.09%) | 374 |

The 512-token window contains no EOS/EOG. In the 2,048-token window, `<|im_end|>` appears at
total output token 1,150 (zero-based index 1,149). Model-default stopping is disabled by the
benchmark, so the rest includes special-message tokens and further reasoning. This is sustained
execution data, not useful long-code completion throughput or a scored complete solution.
All 513/2049 output IDs and every acceptance counter match the prior argmax-only route. That
route measured 99.32/95.12 decode tok/s respectively; the 0.6–0.8% single-run difference does
not establish a stable selector speedup. No full-logit versus argmax-only controlled A/B has
been rerun for these windows, so older reports do not establish an isolated argmax speedup.

The selector test checks full-vocabulary and sharded routes against the same independent CPU
sort and FP64 lattice oracle, including logit ties, uneven shards, multiple columns and the
248320-row model vocabulary. The real DFlash7 TP2 Engine gate checks graph/eager committed IDs
and acceptance and 64 fresh non-speculative teacher-forcing positions: zero disagreements,
worst emitted-logit deficit zero. Earlier v3 gates additionally cover prefix replay and early
stopping. These checks are not a universal model-capability score or an NVFP4/Q4 quality comparison.

Earlier P2P-enabled v3 snapshots, before these transfer reductions, measured 143.06 ± 0.13
decode tok/s on 3,072 input / 512 timed output tokens (two repetitions). On the distinct
`profiles/bench/v100-code-85000-iommu-pt.ids` corpus at the same 98,304 capacity and chunk,
DFlash7 measured 73.38 ± 0.012 decode tok/s (47.18% acceptance, 1178.39 prefill tok/s), versus
83.13 ± 0.002 for its MTP3 control (78.56% acceptance, 1226.21 prefill tok/s). These are earlier
code snapshots, not current remeasurements. Different prompts, output lengths and acceptance
prevent calculating a current speedup against those figures or the 180K MTP tables. There is
no paired P2P-disabled measurement of the current DFlash route.

The separate saved high-code corpus `profiles/bench/v100-code-85000-iommu-pt.ids` was also
measured at 85000 input tokens, capacity 98304 and chunk1024, with the same TP2/INT8/greedy/
Graph settings. These are single cold requests, not repeated means. Model-default stopping
is disabled; **all of these windows continue past the response's first end token**.

| Backend | Timed decode tokens | Prefill tok/s | Committed decode tok/s | Accepted/drafted | Resident request s |
|---|---:|---:|---:|---:|---:|
| DFlash3 | 2048 | 1183.83 | 79.30 | 1432/1845 (77.62%) | 97.63 |
| DFlash5 | 2048 | 1181.42 | 76.10 | 1578/2350 (67.15%) | 98.87 |
| DFlash7 | 2048 | 1178.94 | 87.91 | 1640/2853 (57.48%) | 95.40 |
| MTP3 | 2048 | 1229.42 | 86.14 | 1461/1761 (82.96%) | 92.92 |
| DFlash3 | 1024, fresh | 1184.32 | 70.87 | 679/1034 (65.67%) | 86.23 |
| DFlash5 | 1024, fresh | 1181.50 | 63.82 | 741/1411 (52.52%) | 87.99 |
| DFlash7 | 1024, fresh | 1179.42 | 68.72 | 763/1826 (41.79%) | 86.98 |

The first `<|im_end|>` is at total output token 942 for DFlash3/7 and MTP3, and 917 for
DFlash5. Post-stop output includes repeated special-message/code sequences; its improved
predictability affects acceptance and throughput. These rates are sustained stress results,
not useful long-code completion speeds. DFlash3/7 have identical 2049 IDs to MTP3 and their
fresh 1025 IDs match that prefix. DFlash5 first differs at total output token 355; this
comparison does not establish token-identical behavior for every draft window.
DFlash7's single-run decode advantage over MTP3 is only 2.06%, while the complete resident
request is slower (95.40 versus 92.92 seconds). No stable universal DFlash speedup is claimed.

Temporary draft9/11/15 support measured 58.47/45.77/42.79 tok/s on the fresh 1024-token
window, with acceptance falling to 36.73%/30.38%/22.27%. It did not improve performance
and was reverted; the supported 27B maximum remains seven. These fresh 1024-window and
wider-draft measurements were taken after the optimization deadline, not within the eight-hour
window. Local reports use `highcode_<backend>_85k_2048_current.json`,
`highcode_<backend>_85k_1024_fresh.json` and `highcode_dflash<k>_85k_1024_experimental.json`
under `profiles/bench/optimization_8h/current/`.

An existing eager diagnostic trace isolates the stages at 85K input, capacity98304,
chunk1024, DFlash7 and 128 timed outputs over 28 rounds. NVTX GPU-projected averages are
7.27 ms/round for proposal (including selector), 1.06 ms for selector (including the head
projection), and 48.24 ms for target verification/acceptance. The selector range is nested,
and dual-GPU work overlaps; these are not additive kernel totals. Eager execution and tracing
also add overhead, so the timings diagnose this route rather than predict production Graph
throughput. Target verification dominates this trace, not just the five-layer drafter.
The saved trace is `profiles/nsys/optimization_8h/highcode_dflash7_85k_128_eager_instrumented`;
temporary diagnostic instrumentation was removed. The Graph trace did not expose enough node
detail to support a complete Graph-mode stage breakdown.

To reproduce using the existing artifact and saved LRU corpus (change `-pg` to `85000,2048`
for the forced long window):

```bash
taskset -c 0-15 env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/bench/ninfer_bench \
  --weights /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --corpus profiles/bench/optimization_8h/lru_code_85000.ids -pg 85000,512 \
  --max-ctx 98304 --prefill-chunk 1024 --kv-dtype int8 --tp 2 --devices 0,1 \
  --spec dflash --draft-tokens 7 --warmup 0 -r 1 --capture-generation -o json
```

The local reports are under `profiles/bench/optimization_8h/current/`, named
`lru_dflash7_sharded_selector_85k_512.json` and `lru_dflash7_sharded_selector_85k_2048.json`.
Raw reports and corpora are not committed.

#### Stop-aware non-code workloads: DFlash7 versus MTP3

On 2026-10-01, the same official NVFP4 v3 artifact was measured through the public Engine
on this PCIe-P2P V100X2 using four tasks: a three-chapter Chinese story requesting 900–1200
Chinese characters and `ORCHID-37`; a six-section English-to-Chinese translation with at least
20 glossary entries; exactly 32 JSONL records with prescribed fields, arithmetic and field
order; and the five-slot logic puzzle with expected final `CHECK=4606`.

Both backends use **98,304 context/KV capacity**, prefill chunk 1,024, TP2 devices 0,1,
INT8 group-64 KV, CUDA Graphs, one active request, greedy sampling and zero presence/frequency
penalties. DFlash uses seven drafts and the full proposal head; MTP uses three drafts and
the optimized proposal head. Thinking and prefix reuse are disabled. Model loading and graph
priming are outside request timing. The maximum output is 2,048 tokens, with model-default
stopping **enabled**. All 80 requests stop at their first model end token, below the output
limit; there is no forced post-stop continuation. Committed decode throughput is
`(generated_token_count - 1) / decode_seconds`; the first generated token belongs to prefill.
Total output counts below include that first token and the terminating model token.

Native prompts contain 129 / 395 / 118 / 417 tokens for story / translation / JSONL / logic.
Each native prompt has two cold-prompt repetitions per backend; decode rates show mean ± sample
standard deviation and prefill rates show means. Each task at exactly
1,024 / 2,048 / 4,096 / 8,192 / 16,384 / 32,768 / 65,536 / 85,000 occupied input tokens has
**one measured request per backend**, not a repeated mean. Distinct local technical documents
provide non-repeated background before the unchanged final task/template. Their raw token IDs
were frozen before measurement and shared by both backends. This is neither a retrieval-quality
test nor a single-variable causal experiment in length, since visible background content varies.

The campaign resumed after an unexpected host reset, retaining completed requests and running
only missing cases. Power limits remained 300 W per card; post-reset P2P read/write checks still
passed. All within-backend native repetitions reproduce their output IDs and acceptance counters.
The local task-specific harness, frozen cases and reports are
`profiles/bench/other_inputs_20261001/{other_inputs.cpp,cases.json,dflash7.json,mtp3.json}`;
the harness uses the public Engine for all inference. Raw reports and corpora are not committed.

In every paired table, **D/M means DFlash7 / MTP3**. An identical-ID entry is an exact comparison
of the complete generated token sequence, not a task-quality score.

##### Chinese story

| Actual input tokens | Prefill tok/s D/M | Committed decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical output IDs |
|---:|---:|---:|---:|---:|:---:|
| 129 (native) | 593.52 / 618.57 | 49.20 ± 0.004 / 74.83 ± 0.004 | 12.16% / 39.35% | 1257 / 1087 | No |
| 1024 | 1520.14 / 1663.46 | 51.24 / 76.66 | 13.74% / 41.56% | 1084 / 1084 | Yes |
| 2048 | 1569.95 / 1661.93 | 51.09 / 74.41 | 14.48% / 40.66% | 1162 / 1162 | Yes |
| 4096 | 1559.68 / 1650.43 | 47.60 / 70.98 | 12.34% / 36.87% | 835 / 835 | Yes |
| 8192 | 1537.54 / 1623.52 | 49.11 / 70.17 | 13.69% / 37.43% | 1122 / 1122 | Yes |
| 16384 | 1489.53 / 1571.28 | 44.80 / 68.16 | 12.04% / 37.52% | 914 / 861 | No |
| 32768 | 1401.28 / 1474.85 | 41.44 / 63.23 | 12.52% / 37.09% | 1030 / 1030 | Yes |
| 65536 | 1250.19 / 1303.80 | 36.02 / 55.63 | 13.04% / 37.44% | 1033 / 1033 | Yes |
| 85000 | 1182.64 / 1233.38 | 32.26 / 51.67 | 12.06% / 36.30% | 961 / 961 | Yes |

##### English → Chinese translation

| Actual input tokens | Prefill tok/s D/M | Committed decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical output IDs |
|---:|---:|---:|---:|---:|:---:|
| 395 (native) | 1196.01 / 1234.86 | 134.68 ± 0.005 / 123.00 ± 0.039 | 58.31% / 86.43% | 930 / 908 | No |
| 1024 | 1576.32 / 1661.44 | 127.44 / 117.58 | 55.27% / 81.42% | 779 / 779 | Yes |
| 2048 | 1571.21 / 1661.55 | 128.45 / 116.14 | 57.40% / 81.62% | 839 / 839 | Yes |
| 4096 | 1559.48 / 1650.06 | 116.31 / 113.49 | 50.71% / 78.92% | 688 / 688 | Yes |
| 8192 | 1534.84 / 1623.27 | 125.74 / 116.00 | 57.38% / 83.90% | 923 / 923 | No |
| 16384 | 1487.15 / 1571.02 | 112.08 / 110.15 | 51.64% / 80.86% | 743 / 741 | No |
| 32768 | 1401.27 / 1471.30 | 110.56 / 108.44 | 57.29% / 87.20% | 952 / 952 | Yes |
| 65536 | 1248.25 / 1302.42 | 85.29 / 89.32 | 50.40% / 80.53% | 893 / 893 | Yes |
| 85000 | 1180.46 / 1230.43 | 79.61 / 84.86 | 50.80% / 81.41% | 893 / 893 | Yes |

##### 32-record JSONL

| Actual input tokens | Prefill tok/s D/M | Committed decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical output IDs |
|---:|---:|---:|---:|---:|:---:|
| 118 (native) | 580.80 / 582.25 | 210.50 ± 0.058 / 137.09 ± 0.043 | 99.05% / 100.00% | 1190 / 1190 | Yes |
| 1024 | 1577.96 / 1658.23 | 206.30 / 135.98 | 99.05% / 99.89% | 1190 / 1190 | Yes |
| 2048 | 1569.36 / 1662.03 | 200.78 / 133.63 | 99.05% / 99.89% | 1190 / 1190 | Yes |
| 4096 | 1560.76 / 1649.45 | 201.09 / 133.82 | 99.05% / 100.00% | 1190 / 1190 | Yes |
| 8192 | 1529.92 / 1623.11 | 198.69 / 132.06 | 99.05% / 99.78% | 1190 / 1190 | Yes |
| 16384 | 1489.98 / 1570.78 | 192.60 / 128.33 | 99.05% / 100.00% | 1190 / 1190 | Yes |
| 32768 | 1400.07 / 1472.45 | 175.01 / 119.59 | 99.05% / 100.00% | 1190 / 1190 | Yes |
| 65536 | 1246.28 / 1302.94 | 148.14 / 104.67 | 98.30% / 99.89% | 1190 / 1190 | Yes |
| 85000 | 1178.83 / 1228.32 | 137.65 / 98.77 | 98.30% / 100.00% | 1190 / 1190 | Yes |

##### Five-slot logic puzzle

| Actual input tokens | Prefill tok/s D/M | Committed decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical output IDs |
|---:|---:|---:|---:|---:|:---:|
| 417 (native) | 1237.40 / 1279.63 | 158.88 ± 0.013 / 126.10 ± 0.034 | 71.64% / 89.30% | 1204 / 1204 | Yes |
| 1024 | 1577.52 / 1664.03 | 157.87 / 125.01 | 71.83% / 89.14% | 652 / 652 | Yes |
| 2048 | 1571.71 / 1661.37 | 173.32 / 126.34 | 82.86% / 92.18% | 850 / 850 | Yes |
| 4096 | 1562.27 / 1649.73 | 162.94 / 125.98 | 76.98% / 91.47% | 805 / 805 | Yes |
| 8192 | 1531.70 / 1623.05 | 166.37 / 125.39 | 80.61% / 93.24% | 837 / 823 | No |
| 16384 | 1486.20 / 1570.62 | 161.27 / 120.91 | 80.65% / 92.24% | 824 / 824 | Yes |
| 32768 | 1399.80 / 1465.85 | 142.03 / 111.83 | 77.55% / 90.98% | 676 / 676 | Yes |
| 65536 | 1245.29 / 1302.07 | 122.20 / 98.14 | 78.44% / 91.67% | 689 / 689 | Yes |
| 85000 | 1176.65 / 1226.37 | 111.05 / 91.60 | 76.41% / 90.37% | 693 / 693 | Yes |

##### Whole-request interpretation and output qualifications

| Task at 85,000 input tokens | Prefill seconds D/M | Decode seconds D/M | Complete resident request seconds D/M |
|---|---:|---:|---:|
| Chinese story | 71.87 / 68.92 | 29.76 / 18.58 | 101.64 / 87.50 |
| English → Chinese translation | 72.01 / 69.08 | 11.20 / 10.51 | 83.21 / 79.60 |
| 32-record JSONL | 72.11 / 69.20 | 8.64 / 12.04 | 80.75 / 81.24 |
| Five-slot logic puzzle | 72.24 / 69.31 | 6.23 / 7.55 | 78.47 / 76.87 |

DFlash7 improves JSONL decode by 39.37% and logic decode by 21.24% at 85K, with identical output
IDs, but is slower on story and translation decode. Its prefill is slower for all tested tasks
and lengths. At 85K, JSONL's complete request improves by only about 0.61% in this single pair;
logic's complete request is slower despite faster decode. These single measurements do not
establish stable whole-request gains or a universal DFlash speedup.

All 10 JSONL completions per backend pass exact record/value/count and field-order checks.
All 10 logic completions per backend end in `CHECK=4606`; the expected service/color/port
mapping was also reviewed, including responses that use separate assignments rather than
the checker's canonical `Slot N:` block. All translations retain six section headings and
at least 20 glossary entries (25–60 here); those structural checks are not an independent
translation-accuracy or technical-boundary score.

Story compliance is incomplete in both backends: native stories contain 1,586 / 1,379 Chinese
characters, beyond the requested 900–1,200, and the native DFlash story omits `ORCHID-37`.
Both 4K stories also omit that literal ID. Only the 16K and 85K story rows satisfy all of the
listed literal/length/chapter checks; this does not score plot quality or every semantic detail.

Complete output IDs differ for native story/translation, 8K translation/logic, and 16K
story/translation; their first differences are at total output tokens 20/692, 243/296 and
33/380 respectively. The other 30 of 36 task/occupancy pairs match exactly, including every
85K pair and every JSONL pair. Different-output rows are descriptive workload measurements,
not strict same-output speedups. No claim of universally token-identical or quality-loss-free
DFlash behavior follows from this sweep.

#### Native RAM-KV: larger windows and short inputs

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

#### Experimental PP2 layer pipeline

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

#### Reproduction on the current host

Use the existing artifact and saved corpus; if regenerating a corpus, keep the final task and
assistant prefix intact. The `--output-tokens 1024` corpus option is task wording, separate from
the benchmark's 512-token decode window. Replace 85000 in both corpus and `-pg` arguments with
3072/8192/16384/32768/65536 for the other occupancy workloads.

```bash
export LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  profiles/bench/v100-code-85000-iommu-pt.ids --code-chat 85000 --output-tokens 1024 \
  src/core/host_worker_pool.h src/core/host_worker_pool.cpp \
  src/runtime/engine/concurrent_executor.h \
  src/targets/qwen3_6/impl/runtime/program_impl.h \
  src/targets/qwen3_6/impl/runtime/text_context_impl.h \
  src/targets/qwen3_6/impl/runtime/layouts_impl.h \
  src/targets/qwen3_6/impl/runtime/mtp_impl.h \
  src/ops/kernel/gqa_attention_decode_i8.cuh \
  src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh
build-v100/bench/ninfer_bench \
  --weights /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --corpus profiles/bench/v100-code-85000-iommu-pt.ids -pg 85000,512 \
  --max-ctx 180000 --prefill-chunk 3072 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --tp 2 --devices 0,1 \
  -r 2 --warmup 0 --capture-generation -o json \
  --output-file profiles/bench/nvfp4_v3_85k_p2p_identity.json

build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  profiles/bench/v100-code-512-p2p.ids --code-chat 512 --output-tokens 1024 \
  src/core/host_worker_pool.h
.venv/bin/python3 tools/v100/bench_capacity.py --engine ninfer \
  --weights /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --capacities 1024 2048 4096 8192 16384 32768 65536 \
  --corpus profiles/bench/v100-code-512-p2p.ids --output-dir profiles/bench/p2p_capacity
```

### P2P disabled

The same-input peer-off control uses a process-local `cudaDeviceCanAccessPeer` query shim to
select the existing verified staging path. It neither changes system configuration nor exposes
a new product CLI option. At 85000 occupied tokens and 180000 capacity, two repetitions give:

| Prefill tok/s | Committed decode tok/s | Wall decode tok/s | Acceptance |
|---:|---:|---:|---:|
| 1297.94 ± 2.66 | 79.18 ± 0.014 | 79.151 | 79.12% |

Each run accepts 360/455 drafts over 152 rounds and produces 513 output IDs identical to P2P.
The local report is `profiles/bench/nvfp4_v3_85k_no_peer_identity.json`. The following kernel,
request-ledger and Q4_K_M results are earlier staging measurements, not current direct-P2P
measurements. Their original corpus was `/tmp/v100-code-85000.ids`.

#### Prefill and external optimisation review

The promoted route is measured on the real Q4_K_M artifact with INT8 KV, TP2 and
180,000-token capacity. With `prefill_chunk=4096`, an 8,192-token prompt took **4.897 s**
to prefill (**1,672.9 tok/s**); the fixed 85,000-token code corpus measured **1,251.44 ± 2.87
tok/s** on the pre-P2P staging transport path. The latter has exactly 85,000 occupied prompt tokens.
These are prefill rates, not decode rates, and both exceed the requested 1,000 tok/s target.
The 4,096-token chunk is the V100X2 launcher default; the Engine and generic CLI retain their
target-agnostic 1,024-token default.

On the 8,192-token probe with this kernel, a chunk sweep measured **1,454.4**, **1,599.3**,
**1,636.1**, **1,672.9**, **1,672.3**, and **1,195.2 tok/s** for chunk sizes 1,024, 2,048,
3,072, 4,096, 5,120, and 8,192 respectively. The 85,000-token corpus measured **1,133.0**,
**1,213.4**, and **1,251.0 tok/s** at 1,024, 2,048, and 4,096 respectively. Larger chunks
increase the startup workspace reservation; at 4,096 it was **1.49 GiB per device**, still within
the two 16 GB cards' capacity at 180,000 context. Values are individual cold-prefill runs, not
averages across repeated campaigns.

Before the multi-output route was promoted, an 8,192-token probe took **27.130 s** with
`prefill_chunk=1024` and **27.075 s** with `prefill_chunk=4096`. A CUDA Nsight Systems capture
attributed **89.0%** of GPU kernel time to the existing Volta GGML-K Tensor-Core GEMM (`1036`
launches, `11.544 s` aggregate in the 2,048-token capture). Those measurements are retained as
historical attribution only; they are not the current implementation result.

Two controlled tile experiments were rejected: using the 32-token GEMM tile for long prefill
increased the 8K probe to **30.828 s**, and a 128-token/512-thread tile increased it to
**38.725 s**. Both were reverted. The independent GGML-K FP64-oracle suite remains passing.
The promoted SM70 route decodes each Q4_K block with one cooperative block-wide pass,
materializes GGML-K rows into caller-owned FP16 workspace, and uses the existing CUTLASS Volta
Tensor-Core GEMM, including an FP32 output path for GDN control projection. The packed Q4_K/Q6_K
bytes remain unchanged. The first two generated IDs were unchanged (`71093, 10504`) in the
controlled route comparison. These are prefill results only; the earlier staging long-decode
result is recorded below.

#### NVFP4 prefill and decode

The active SM70 implementation uses QPN for narrow NVFP4/FP8 projections and CUTLASS for wide
prefill projections, including prepacked TP2 weights. Wide Linear calls with caller workspace
decode weights once for the whole call instead of repeating the narrow kernel over 32-token
chunks. NVFP4 K16 tuples are decoded cooperatively. Both wide prefill and decode retain FP32
gate/up projections through SwiGLU and round only the final activation to BF16. FP8 wide
projections apply the stored BF16 row scale in FP32 before the nonlinear combination.

The prepacked FP8 vocabulary head requires a matching kernel even at T=1. Its previous
row-major GEMV dispatch read the wrong weights for the first token and one-token chunk tails.
Independent FP64 Linear tests reproduced that defect on both full and TP2 head shapes, and
pass after the dispatch fix. Performance runs containing that bug are excluded from the current
result; plausible generated text alone did not establish correctness.

On 2 × V100-SXM2 16 GB, CUDA 12.8, the fixed local `/tmp/v100-code-85000.ids` corpus,
`qwen3.8-27b/nvfp4`, 85,000 occupied tokens, 180,000 capacity, complete INT8 group-64 KV, TP2,
CUDA Graphs, greedy sampling, MTP3 and the optimized draft head:

| Implementation | Runs | Prefill tok/s | Committed decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| FP32 SwiGLU, shared attention scores/reducer weights, prepared GDN Q/K (v2 artifact) | 2 | **1,279.44 ± 2.94** | **78.36 ± 0.04** | **79.12%** |
| Same route, official upstream v3 container | 2 | **1,277.61 ± 3.12** | **78.424 ± 0.0004** | **79.12%** |

The same workload was rerun against the official `NINFER\x00\x03` container
`neroued/Qwen3.8-27B-nvfp4-NInfer` after the reader compatibility adapter was added. The v3
artifact measured **1,277.61 ± 3.12 prefill tok/s** and **78.424 ± 0.0004 committed decode
tok/s** over two repetitions, with the same **79.12%** MTP3 acceptance. The report is
`profiles/bench/nvfp4_v3_85k.json`; this is the earlier staging v3-container result. A separate stable
512-prompt/512-output run measured **98.831 ± 0.027 committed tok/s** at 2,048-token capacity;
the 120.66 tok/s result from a 21-token smoke request is a short-window peak, not a full benchmark.

These use `prefill_chunk=3072`, no prefix reuse and no extra benchmark warmup; graphs are primed
before measurement. Every run produces one token during prefill followed by 512 timed decode
tokens with model stopping disabled. The two runs match all 513 output IDs and the same 360/455
accepted drafts over 152 rounds per run. This is a fixed throughput window; the generated C++ is
truncated at the output limit and is not scored as a complete solution. The 3,072-token chunk reserves
1,502,358,016 bytes of workspace per GPU; a 4,096-token chunk does not fit alongside the same
180K capacity on this host. The retained local report is
`profiles/bench/nvfp4_85k_fp32_swiglu.json`; the
[P2P-disabled README area](https://github.com/tuxKOH/ninfer-V100X2/blob/756d136377499c47f0938494461ec71a697f5f08/README.md#p2p-disabled) summarizes these earlier results.
Current reproduction commands are above.

The earlier 72.28, 75.53 and 78.90 tok/s figures preceded the wide SwiGLU correction. Those paths
materialized gate/up as BF16 before SiLU/product, with another BF16 rounding after FP8 row scaling.
Expanded independent FP64 tests failed for both weight formats. The FP32 implementation passes
the original A16 criterion (relative L2 0.0033, gross absolute 0.005 plus 0.0063 times the maximum
reference magnitude), including full and TP2 shapes at 3,072 tokens. No tolerance was widened.
The corrected generated sequence differs, so the older figures are historical measurements rather
than qualified baselines or a controlled measure of the correction's performance cost.

The reducer computes each split's softmax weight once per CTA and stages eight vectors in
parallel, preserving the prior FP32 max/denominator tree and ascending numerator accumulation.
It retains every KV token and adds no quantization. An isolated 85K TP2 four-query reducer probe
decreased from about 172 to 54 µs with identical BF16 outputs; this is operator-stage timing,
separate from the end-to-end decode measurement above. A reassociated-sum experiment was faster
but changed the generated sequence and is not the delivered route.

The INT8 partial-attention kernel now shares two independent QK tiles across four PV warps.
Each warp keeps the same softmax and PV arithmetic order. The isolated TP2 85K/four-query probe
decreased from 739 to 647 µs with identical partial accumulators and softmax statistics.
Long SM70 GDN prefill normalizes Q/K once into FP32 scratch, shared by every state tile/value
head. The FP32 state update order and BF16 output boundary remain unchanged; 3,072-token TP2
scratch decreases from 102.28 MiB of previously unused chunked storage to 24 MiB. The public
GDN benchmark measures 3.887 ms median with cold L2 and CUDA Graph replay (5 warmups, 21 samples).
Its independent FP64 test covers all outputs and final state, including the full 3,072-token
TP2 recurrence, nonzero initial state, in-place operation and the 63/64/65-token route boundary.

The complete GQA public-contract suite and the TP2 head-local suite pass the independent FP64
oracle, including every head/key for four queries at 85K and a 180K envelope. Native/prepacked
FP8 and NVFP4 Linear prefill tests at T=128/1024/4096 also pass their independent oracle. This
evidence covers numerical behavior and the measured generation; it does not establish a general
task-quality score or predict other machines' reported 77–90 tok/s.

The final Volta TP2 validation also corrected Q5 and BF16 residual row-shard dispatch: their
Ampere MMA routes were not executable on sm_70. The Q5 A16 shard now uses runtime-K SM70 routes
with its bounded split-K workspace sized at the interior peak, and the BF16 shard uses SM70
CUTLASS. Independent FP64 tests cover the registered shard shapes; the two-card LinearAdd and
SwiGLU composition suites pass. These dispatch corrections do not change the measured NVFP4
artifact's weight format or the profiler breakdown above.

#### NVFP4 full-request ledger

The earlier unprofiled two-run staging measurement above averages:

| Phase | Time | Scope |
|---|---:|---|
| Resident model load | 18.650 s | once per invocation; includes 16.246 s upload |
| Prompt preparation | 0.001570 s | raw 85K token IDs; no tokenizer or media preprocessing |
| Prefill and first token | 66.435 s | all 85,000 tokens, no prefix reuse |
| Decode | 6.534 s | 512 committed tokens; 152 MTP rounds |
| Remaining request overhead | 0.003225 s | submission/publication between measured phases |
| Complete resident request | 72.974 s | includes preparation, prefill and decode; excludes load/graph priming |

Graph priming happens before the measured request; its duration is not included in this ledger.
Vision and prefix reuse are inactive for this workload. Model load transfers 22.458 GB to the two
devices from 21.197 GB of artifact payload; this is separate from request-phase TP2 traffic.

A separate pre-P2P `--profile-measured` run captures all CUDA kernels, copies and runtime calls
in the same staging workload. It produces identical output IDs and acceptance to its unprofiled
control. Tracing increases prefill to
67.588 s and decode to 6.966 s, so the following attribution must not replace unprofiled throughput.
GPU 0's cumulative kernel times are below; GPU 1 runs concurrently and is reported separately in
the local JSON ledger. Every captured kernel is included, including those grouped in the last row.

| GPU work | Prefill seconds | Decode seconds | Optimization finding |
|---|---:|---:|---|
| CUTLASS matrix multiply | 23.096 | — | dominant prefill projection work; wide SwiGLU now writes FP32 intermediates |
| FP8 QPN projections | 0.001 | 1.701 | native row access remains costly; paired-lane loading was slower and rejected |
| NVFP4 QPN projections | — | 1.050 | packed Tensor-Core path; retain qualified arithmetic while investigating access/issue cost |
| W8 draft projections | 0.455 | 0.203 | part of native MTP, measured separately from target projections |
| Q4 draft vocabulary head | 0.001 | 0.120 | already uses reduced draft vocabulary; speculative tokens do not count as output |
| Prefill attention core | 18.574 | — | major remaining prefill target: query tile and key-loop utilization |
| Decode attention partials | 0.002 | 1.719 | QK scores shared across output warps; all history keys retained |
| Attention reducers | 0.035 | 0.159 | ordered shared-weight reducer retained; reassociation rejected |
| KV gather/append and masks | 0.317 | — | full context preserved; no retrieval/subsampling |
| GDN recurrence/QK preparation | 4.643 | 0.121 | prepared Q/K reused; same FP32 state recurrence |
| GDN control projection | 0.437 | 0.070 | much smaller than main projections; separate FP32 control output |
| GDN convolution | 0.222 | 0.033 | preserve persistent convolution state and MTP rollback behavior |
| Weight decoding | 1.918 | — | weights decoded once per wide projection call |
| Activation conversion | 0.739 | 0.018 | possible producer/consumer fusion, subject to the same numerical boundary |
| FP8 row scaling | 0.361 | — | selected Linear shapes fuse scaling into GEMM; SwiGLU scales in its FP32 combine |
| RMS normalization | 0.448 | 0.195 | fusion candidates must be evaluated with their consumers |
| Residual addition | 0.911 | 0.128 | includes collective local sums; communication ordering constrains fusion |
| SwiGLU/sigmoid gates | 0.667 | 0.021 | corrected FP32 SwiGLU intermediates; further epilogue fusion remains a candidate |
| RoPE | 0.010 | 0.018 | small contribution; preserve position semantics |
| Embedding | 0.005 | 0.004 | small contribution |
| Split, sampling, state/control kernels | 0.271 | 0.009 | includes acceptance and token publication, not discarded drafts |
| All kernels | **53.113** | **5.569** | single-device sums, not two-GPU wall time |

The FP8 decode total consists of GDN input 0.667 s, attention input 0.186 s, gate/up 0.218 s,
and linear/output-head work 0.631 s. Of all decode kernels, target verification/acceptance uses
4.875 s and subsequent MTP alignment/proposal uses 0.694 s on GPU 0. The split follows each
`speculative_prepare_verify_inputs` / `mtp_prepare_next_round` boundary; it excludes copies and
host waits. This is the actual MTP3 schedule, with 360 accepted out of 455 drafts.

| Timeline accounting | Prefill | Decode |
|---|---:|---:|
| GPU 0 H2D activity sum | 0.323 s / 112.283 GB | 0.170 s / 1.036 GB |
| GPU 0 D2H activity sum | 0.376 s / 112.282 GB | 0.113 s / 1.036 GB |
| GPU 0 local D2D activity sum | 0.127 s / 41.780 GB | 0.003 s / 0.214 GB |
| GPU 0 timeline with no recorded device activity | 13.636 s | 1.123 s |
| Neither GPU has recorded device activity | 13.251 s | 0.761 s |

The copy rows are CUPTI activity records for CUDA's driver-managed staging, not complete physical
PCIe transfer timings. In particular, their large-copy byte/duration ratios cannot be interpreted
as PCIe bandwidth. The uncovered intervals can contain staging, synchronization and CPU issue
cost; calling all of them communication or all of them CPU overhead would be unsupported.
API durations also include waiting and must not be added to device durations. A direct two-rank
UVA transfer probe found no benefit from issuing the two pulls on separate CPU threads: the
31,457,280-byte payload measured 3.646 ms sequential versus 3.714 ms parallel (median of 30
samples, after 5 warmups, exact bytes verified). That candidate is not in production.

The retained staging breakdown was exported with:

```bash
nsys export --type sqlite \
  --output profiles/nsys/nvfp4_85k_fp32_swiglu.sqlite \
  profiles/nsys/nvfp4_85k_fp32_swiglu.nsys-rep
.venv/bin/python3 tools/v100/profile_ledger.py \
  profiles/nsys/nvfp4_85k_fp32_swiglu.sqlite \
  --benchmark profiles/bench/nvfp4_85k_fp32_swiglu_profile.json \
  --output profiles/bench/nvfp4_85k_fp32_swiglu_ledger.json
```

Capture uses `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
--cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop`.
The ledger retains every kernel's calls and mean duration for both devices, plus API/copy sums
and interval-union accounting. The stage table identifies live optimization opportunities;
it is not a claim that every stage has reached its hardware limit. To collect a new P2P breakdown,
profile the current reproduction command with `-r 1 --profile-measured` and export its own report;
do not relabel the retained staging attribution as a measurement of the current P2P path.

#### External references and integration boundaries

- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), an SM70-focused vLLM fork. Its
  Flash-V100 attention and quantized kernels are not drop-in compatible with NInfer's preserved
  GGUF Q4_K/Q6_K storage; porting one requires a separate oracle and graph-capture qualification.
- [DFlash2](https://inco.ai/blog/dflash2/) and the [Qwen3.8-27B drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
  integrated with dynamic grouped convolution, lattice selection, and TP2 verification.
  The optional v3 package preserves W8 projections and BF16 auxiliary tensors; its measurements
  are in the [P2P-enabled section](#dflash2-v3-separate-capacity). A temporary llama.cpp `sm_70`
  build was used only as the historical external 85K control below.
- [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp), which stores completed KV
  blocks in host RAM and retrieves a query-selected subset into a bounded GPU window. That is an
  approximate attention policy, not a transparent full-180K KV spill: it changes which history
  participates in attention. The default V100X2 route keeps complete-context semantics. The
  [optional native RAM-KV experiment](maintainer/paged-kv-cache.md#15-optional-native-ram-kv-experiment)
  instead uses Q/K mean-vector retrieval adapted to NInfer's paged cache. It remains approximate.

#### Historical pre-v3 DFlash2 route

The following table is historical and belongs to the pre-v3 route. The NInfer benchmark also accepts
`--spec dflash --draft-tokens K` and feeds a saved corpus directly through `Engine::prepare_tokens()`.

For an external engine control, the same raw 85,000-token IDs were sent to the local llama.cpp
build with 98,304 capacity, Q8 KV, DFlash3, and a 128-token fixed window. It produced 30.1 tok/s
(121 drafted, 87 accepted); NInfer's matching 128-token DFlash3 run produced 25.3 tok/s (110
drafted, 91 accepted). A separate older short-prompt experiment is excluded because its workload
was not comparable to this 85K result.

| Historical route | CUDA Graph | Decode tok/s | Acceptance | Output check |
|---|---:|---:|---:|---|
| NInfer DFlash3 | on | **25.19** | 81.21% | same 513 IDs as MTP3 |
| NInfer DFlash3 | off | **25.08** | 81.21% | same 513 IDs as graph |
| NInfer DFlash7 | off | **20.52** | 41.56% | repeated special-message output after token 444 |
| NInfer MTP3 control | on | **52.77** | 75.37% | 513-token window, no EOS/EOG |

The DFlash3 graph/eager output IDs were identical. Requesting a 180,000-token capacity for this
pre-v3 DFlash layout was rejected during startup on the two 16 GiB cards: the draft graph/runtime reservation
requires about 5.48 GiB after weights while only about 3.50 GiB remains per device. These numbers
are therefore a long-code execution measurement at 98,304 capacity, not a 180K-capacity acceptance
claim. DFlash7's lower rate and output degeneration made increasing the draft window
counterproductive in that older route and workload, not in every current v3 workload.

The low historical DFlash rate was not caused by the selector kernel itself. Nsight Systems attributed the
dominant proposal-round cost to the five-layer BF16 drafter and its full Q4_K vocabulary head;
the selector top-k/walk kernels are only a small fraction of a round. Before this change, TP2
also reconstructed the full `[248320, K]` vocabulary image with one cross-device gather per
draft column and wrote an unused copy on rank 1. An intermediate route gathered all columns in one strided
PCIe operation and kept the image only on rank 0 (`gather_columns_rank0`), with exact byte/layout
tests. On that 85K corpus this preserved all 513 output IDs and acceptance, but changed DFlash3
from 25.19 to about 25.29 tok/s (within run-to-run noise). This identified draft projections and
the full-vocabulary head as major remaining costs; it does not profile the current v3 route.

#### PCIe-only transport optimization

These earlier runs used translated `DMA-FQ` IOMMU domains without active NVLink. The current
identity domains enable PCIe P2P as described above. TP2 executes 128 hidden-width all-reduces
per target forward round (64 layers × mixer/MLP row-parallel outputs), each carrying 10 KiB at
batch one. MTP verification batches multiple positions and amortizes that work per committed
token. Weight quantization cannot remove this fixed communication work; equal Q1/Q4/Q6 peaks
alone do not quantify a hardware-only ceiling.

The active runtime transfers with captured UVA `cudaMemcpyAsync(...,
cudaMemcpyDeviceToDevice, ...)` and two-rank event ordering. On the earlier translated-IOMMU setup, CUDA
handles the cross-device D2D copy through its driver-managed staging path. Startup qualifies that
exact API in both directions; collectives do not allocate explicit pinned-host staging buffers. An
earlier explicit pinned D2H/H2D experiment measured **34.47 us mean / 34.17 us p50** for a 10 KiB
all-reduce and **53.18 tok/s** on one 85K decode run (89/114 drafts accepted). That route was
removed after whole-prefill profiling showed higher aggregate transfer time than the single UVA D2D
node; its numbers are historical and are not attributed to the active route.

The later [PP2 experiment](#experimental-pp2-layer-pipeline) implements a startup-fixed 32/32
layer split for SM70 Qwen3.8 NVFP4/QUASAR. It replaces per-layer TP collectives with a
layer-boundary transfer, but decode must visit both stages serially. It is distinct from
this historical TP2 transport evaluation and does not replace the default TP2 route.

The active port uses two Tesla V100-SXM2 16 GB cards, CUDA 12.8 (`sm_70`), and the local
`qwen3.8-27b/gguf-q4-k-m` artifact converted from LM Studio's Qwen3.8-27B Q4_K_M GGUF.
On the fixed code-generation workload below, two pre-P2P staging runs at exactly 85,000 occupied prompt
tokens measured **50.68 ± 0.04 committed decode tok/s** (mean ± sample standard deviation) with
**1,251.44 ± 2.87 prefill tok/s**, 180,000-token capacity and a 128-token output window. This
establishes the earlier staging result for this prompt and output window; it does not guarantee the
same speed on every prompt.

Startup disables PCIe direct P2P for Linux IOMMU `DMA` and `DMA-FQ` domains. An active direct
NVLink mesh is identified via NVML's remote PCI endpoints; those links do not traverse the host
IOMMU and are qualified separately. NVML is used only at startup and is not required for the
PCIe/fallback route. Startup verifies the UVA D2D copy
with exact byte comparisons in both directions, and rejects startup if verification fails. The
collective suite covers different tensor sizes, guards and 64 consecutive rounds; the public Engine
CUDA Graph path is exercised by the measurements below. The INT8 attention test also passes its
independent FP64 oracle at 85K occupied keys for all four queries, all 24 heads and every visible
key, with TP1/TP2 comparisons and exact cache checks. Real-model MTP/non-MTP teacher forcing and
graph/eager regression gates pass on their short-prompt fixtures. The source Q4_K/Q6_K codes and
scales remain unchanged; these checks support the tested routes without asserting universal quality
parity for all prompts.

| Setting | V100X2 comparison profile |
|---|---|
| GPUs | 2 x Tesla V100-SXM2 16 GB, TP2, devices 0 and 1 |
| CUDA compile/runtime | 12.8 / 12.8 |
| Artifact | `qwen3.8-27b/gguf-q4-k-m`; original Q4_K/Q6_K blocks and scales |
| Request mode | One active request |
| Context capacity | 180,000 tokens, native RoPE |
| Primary occupied context | Exactly 85,000 actual prompt tokens |
| KV cache | INT8 group-64; compared with LM Studio's Q8 KV configuration |
| CUDA Graph | Enabled |
| Prefill chunk | 4,096 tokens |
| NInfer MTP | Fixed draft window of three; optimized proposal head (`--lm-head-draft`) |
| Measured output window | 128 decode tokens; two repetitions |
| TP2 transport | Verified UVA D2D copies; CUDA-managed staging on the pre-P2P IOMMU setup |

NInfer's `--mtp-draft-tokens 3` uses a fixed three-token proposal window, shortened when the remaining
output or context budget requires it. Verification may accept zero drafts. This is distinct from
LM Studio's maximum-three/minimum-zero draft configuration: a minimum draft count of zero permits
its proposal policy to vary how many drafts it attempts, whereas zero accepted drafts describes
the verification result. The two engines therefore use related, but different, MTP schedules.

| Repetitions | Prefill tok/s | Committed decode tok/s |
|---:|---:|---:|
| 2 | **1,251.44 ± 2.87** | **50.68 ± 0.04** |

Both repetitions completed without EOS/EOG; aggregate MTP acceptance was **78.07%**. The
superseded explicit pinned-host route's 53.4075 tok/s result is historical and is not the active
transport measurement. Host CPU use remains one worker core while CUDA performs the measured work.

A separate short-input code-chat measurement used 512 prompt tokens, the same 180,000-token
capacity and execution settings, one warmup, and three 256-token decode windows. Its wall decode
rates were **59.9644, 60.0509 and 60.0721 tok/s**, or **60.0291 ± 0.0571 tok/s** (mean ± sample
standard deviation), with **65.89%** draft acceptance. All three 257-token outputs were identical
and EOS/EOG-free. This short-input result is reported separately from the 85K workload.

The previously reported LM Studio values (approximately 45 committed tok/s and a 57 tok/s peak)
are retained only as historical observations. Their workload metadata was incomplete and they are
not used as current acceptance targets or as a matched-engine comparison.

A matched diagnostic run before the current NInfer transport change used LM Studio's CUDA
backend **2.33.0** with its automatic two-GPU split, the same source GGUF and exact 85,000
prompt IDs, Q8 KV, a requested 180,000-token capacity
(rounded by the backend to 180,224), and maximum-three/minimum-zero MTP. Both engines used greedy
sampling and generated 513 output tokens: one from prefill and 512 in the measured decode interval.

| Engine | Decode tok/s | Prefill | MTP acceptance |
|---|---:|---:|---:|
| NInfer V100X2, current UVA-D2D mean of two runs | **50.68** | **1,251.44 tok/s** | 78.07% |
| LM Studio CUDA 2.33.0, historical one run | **35.4977** | 185.374 s | 365 / 440 (82.95%) |

The LM run decoded for **14.42347 s**, evaluated all 85,000 prompt tokens without cache reuse,
and stopped at the output limit without any EOS/EOG token. The LM row used a different 513-token
output window and the older NInfer transport, so it is retained only as historical cross-engine
context, not as a current like-for-like comparison.

Measure committed output tokens per decode second, excluding the first token produced by prefill.
Rejected draft tokens do not count as output. Report repeated measurements and context occupancy;
a short-prompt result at `--max-context 180000` cannot establish performance at 85K occupied tokens.
Use the same prompt content and sampling for the final LM Studio comparison. Record GPU memory,
power and aggregate CPU use; keep CPU below the user's approximately 85% ceiling.

The public Engine benchmark provides a reproducible greedy diagnostic. The code-chat corpus below
uses the artifact's embedded tokenizer and chat template with thinking disabled, distinct repository
source excerpts, and a final bounded blocking task-queue implementation request. The tool trims the
source excerpt body to make the complete prompt exactly 85,000 tokens, preserving the task and
assistant prefix; it does not repeat excerpts to fill the context. Feed the saved token IDs directly
to both engines, without applying another template or decoding and retokenizing them. Keep the
corpus command's `--output-tokens 1024`: this is part of the fixed prompt's wording, independent of
the measured 512-token decode window. The bundled 65,536-token benchmark corpus is too short for
this case:

```bash
cmake -S . -B build-v100 -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-v100 --target ninfer_bench ninfer_v100_corpus -j

LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  /tmp/v100-code-85000.ids --code-chat 85000 --output-tokens 1024 \
  src/core/host_worker_pool.h \
  src/core/host_worker_pool.cpp \
  src/runtime/engine/concurrent_executor.h \
  src/targets/qwen3_6/impl/runtime/program_impl.h \
  src/targets/qwen3_6/impl/runtime/text_context_impl.h \
  src/targets/qwen3_6/impl/runtime/layouts_impl.h \
  src/targets/qwen3_6/impl/runtime/mtp_impl.h \
  src/ops/kernel/gqa_attention_decode_i8.cuh \
  src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh

LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_bench \
  --weights /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  --tp 2 --devices 0,1 --max-ctx 180000 --kv-dtype int8 \
  --prefill-chunk 1024 --mtp-draft-tokens 3 --lm-head-draft \
  --corpus /tmp/v100-code-85000.ids \
  -pg 85000,512 --warmup 0 -r 3 --capture-generation -o json \
  --output-file /tmp/ninfer-v100x2-85000.json
```

`-pg 85000,512` fixes the output window at 513 tokens: one from prefill and 512 from decode.
Each measured decode follows the full 85K-token prefill, and the Engine primes its CUDA Graphs
before use; `--warmup 0` omits an additional benchmark warmup generation. For a cross-engine
wall-time decode comparison, compute each repetition's rate as
`512 / (timings.total_seconds - timings.first_token_seconds)`. Both timestamps include the same
prompt preparation and submission origin, so subtraction leaves the interval from first-token
commit to request completion, including work between decode rounds. The separately reported
`decode_output_tok_s` uses accumulated Program decode-phase time and excludes some Engine work
between rounds.

`--capture-generation` retains each measured repetition's raw text and token IDs under
`reps[].generation`. NInfer's fixed-budget benchmark disables model-default stopping but can emit
EOS tokens. An accepted repetition must contain no EOS within its entire 513-token output window;
also inspect the captured text for repetition before counting its rate as useful answer throughput.
The matched 512-token LM comparison uses `compare_llama.py` with `ignore_eos=false`, without
EOG-token logit biases. It requires `tokens_predicted=513` and a non-EOS ending; otherwise it saves
the returned output and rejects the repetition instead of calculating a complete-window rate.

The comparison used this local LM Studio backend command, leaving GPU splitting automatic:

```bash
LD_LIBRARY_PATH=/home/z/.lmstudio/extensions/backends/vendor/linux-llama-cuda-vendor-v1 \
/home/z/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.33.0/llama-server \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --ctx-size 180000 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 \
  --threads 16 --threads-batch 24 \
  --host 127.0.0.1 --port 18081 --no-webui
```

With that backend ready, run the matched single-round diagnostic using Python 3.11:

```bash
.venv/bin/python3 tools/v100/compare_llama.py \
  --url http://127.0.0.1:18081 \
  --corpus /tmp/v100-code-85000.ids \
  --prompt-tokens 85000 --decode-tokens 512 --repetitions 1 \
  --output /tmp/llama-v100x2-85000.json
```

This one-round LM Studio diagnostic does not establish a stable average and is retained only for
historical cross-engine context.

These are fixed-window code-generation throughput measurements, not evaluations of whether the
generated code correctly solves the requested programming task.

Quality checks are separate from timing. Preserving the source quantization blocks is an exact
conversion claim; control-tensor transformations and floating-point operators require numerical
oracles. Real-model MTP/non-MTP teacher forcing, graph/eager comparisons and cross-device state
checks qualify the execution path. A plausible answer or a faster kernel alone is insufficient to
claim unchanged model quality or an end-to-end speedup.

## V100X2 P2P-disabled capacity methodology

The earlier staging Q4_K_M capacity comparison in the README varies only the requested maximum
context from 1,024 through 65,536 tokens, doubling at each step. Every request uses the same 512-token code-chat
prompt and generates 257 tokens: one from prefill and 256 in the measured decode interval.
It measures the effect of the capacity setting, not inference with that many occupied tokens.

Each engine starts with a fresh resident model at each capacity, discards one complete warmup
request, and measures three complete requests with prompt-cache reuse disabled. Both consume
the exact saved prompt IDs, use greedy sampling, and retain their configured MTP3 policies.
NInfer uses TP2, INT8 group-64 KV, CUDA Graphs and the optimized proposal head; LM Studio uses
backend 2.33.0, automatic GPU splitting, Q8 KV and maximum-three/minimum-zero MTP as above.
Rates exclude the first token and loading time. The tool rejects EOS/EOG, incomplete windows,
truncated input and unexpected LM prompt-cache reuse. Reported deviations are sample standard
deviations across three repetitions.

Reproduce the prompt and comparison using the existing artifact and Python 3.11 environment:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  /tmp/v100-code-512.ids --code-chat 512 --output-tokens 1024 \
  src/core/host_worker_pool.h

.venv/bin/python3 tools/v100/bench_capacity.py \
  --corpus /tmp/v100-code-512.ids --output-dir /tmp/v100-capacity
```

The runner executes the engines sequentially and stops only its own temporary LM server.
Its output directory contains raw JSON responses, engine logs, requested and actual capacities,
per-repetition timings, MTP counts, and `summary.json` / `summary.md`. The `--engine ninfer`
and `--engine llama` options allow running the two sides separately in that same directory.

Before P2P was enabled, the corrected NVFP4 implementation was separately measured with the
same 512-token code-chat fixture,
256-token decode window, one warmup and three repetitions at 8,192/16,384/32,768/65,536 capacities.
The results were **97.65 ± 0.08 / 97.67 ± 0.09 / 97.76 ± 0.11 / 97.55 ± 0.15 tok/s**,
respectively; prefill was 1,369.9–1,370.9 tok/s and MTP acceptance was 70.04% at every capacity.
All requested capacities were allocated exactly. All 12 runs reproduced the same 257 output IDs,
without EOS/EOG, accepting 173/247 drafts over 83 rounds each. This is a capacity-setting check
with 512 occupied prompt tokens, not a filled-context measurement or a matched comparison against
LM Studio's Q4_K_M weights. The following commands define that workload, but startup selects
transport automatically: running them on the current identity-domain host measures P2P, not
peer-off staging. The new P2P sweep above adds 1024/2048/4096 capacities. Earlier workload commands:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  /tmp/v100-code-512.ids --code-chat 512 --output-tokens 1024 \
  src/core/host_worker_pool.h
.venv/bin/python3 tools/v100/bench_capacity.py --engine ninfer \
  --weights /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --capacities 8192 16384 32768 65536 --corpus /tmp/v100-code-512.ids \
  --output-dir profiles/bench/nvfp4_fp32_swiglu_capacity
```

## V100X2 prefix cache

The same two V100-SXM2 16 GB cards and GGUF-derived Qwen3.8-27B Q4_K_M artifact support
retained-prefix reuse with TP2 and optimized MTP3. The real-model gate uses INT8 group-64 KV,
4,096-token capacity, 256-token prefill chunks, greedy sampling and 32 output tokens. A rendered
code prompt contains 3,274 tokens; `preserve_thinking=true` saves its complete response boundary.

In the P2P-enabled regression, three warmed cold/cache pairs measured median Engine time to first
token of **3.33492 s cold** and **0.0168689 s cached** (197.7×) for Q4_K_M. NVFP4 v3's equivalent
diagnostic measured **3.01954 s cold** and **0.0146475 s cached** (206.1×). Model loading, prompt
rendering and HTTP transport are excluded.
Diagnostic logit/peer copies are disabled for these pairs. Cached requests report 3,274 reused
tokens and zero computed prefill tokens. Two exact-history follow-up turns each prefill only 30
tokens. These are prompt-work savings, not a decode-rate
increase or a cache-enabled comparison against LM Studio.

The gate covers response-checkpoint replay, repeated append, zero-suffix sampling, changed-prefix
reset, rewritten response suffixes and stopping inside an accepted MTP round before continuing.
Repeated checkpoint execution, direct continuation versus checkpoint restore with identical
prefill partitions, and CUDA Graph versus eager execution must agree exactly in generated tokens,
captured logits and MTP acceptance. Both ranks' speculative egress must agree.

Cold re-prefill uses a different BF16 GEMM/GDN partition from retained decode and suffix state.
Two cold comparisons first diverged after 26 and 25 identical output tokens respectively; at each
shared history, a fresh single-output target evaluation assigned the cached choice exactly the
same logit as its selected winner (deficit 0). The test checks this first divergence against the
existing TP2 0.5-logit near-tie bound; it does not claim bit-identical free-running output across
different prefill partitions. Other cold output comparisons remain exact.

Reproduce the Engine gate and the actual HTTP turn/response-checkpoint smoke with the existing
artifact:

```bash
export LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NINFER_V100X2_ARTIFACT=/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  build-v100/tests/ninfer_qwen3_8_27b_v100x2_prefix_real_test

.venv/bin/python3 tools/smoke/serve_thinking_preservation.py \
  --artifact /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  --server-bin build-v100/apps/ninfer-serve --backend mtp \
  --tp 2 --devices 0,1 --kv-dtype int8
```

The HTTP smoke uses a temporary local server with a 1,024-token capacity and 128-token chunks.
The cache remains process-local and can resume only the current frontier or its saved complete
turn/response checkpoint; see [serving cache behavior](serving.md#execution-behavior).

## Inherited RTX 5090 campaigns

Tested Git revisions for the inherited campaigns:

- Qwen3.8-27B NVFP4 MTP0 context-length serving:
  `f08597d6eaafce5b875934aaa85854fcd5426df8`;
- Qwen3.8-27B NVFP4 MTP3 single-request and concurrent fixed-corpus serving:
  `32c9881b6783949df4999422a764b3dcaa111b13`;
- Concurrent MTP3 decode saturation for the three measured Qwen3.6 artifact profiles:
  `26da9df7c1b3d3c04ea7bbd730271aa01d00742a`;
- Refreshed Qwen3.6-35B-A3B and Qwen3.6-27B NVFP4 MTP3:
  `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e`;
- Qwen3.6-35B-A3B stored MTP3 response audit:
  `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B DFlash block=8 (`k=7`):
  `0dc94097e8ec5c5bcf59b9e13e9d1852f504eb61`;
- Qwen3.6-27B NVFP4 accuracy and MTP0:
  `b3d4d0f50b868711c62432bbd68e746217a2f49a`;
- Qwen3.6-27B groupwise-int MTP3: `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423`;
- Qwen3.6-35B-A3B MTP0 and Qwen3.6-27B groupwise-int MTP0:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`.

The Qwen3.6 measurements characterize its three registered artifact profiles independently on one
NVIDIA GeForce RTX 5090. They cover long-context prefill and baseline decode with speculative
decoding disabled, plus long-reasoning and cross-scenario decode with MTP and DFlash. The Qwen3.6
concurrent decode-saturation campaign measures all three profiles at C=1, 2, 4, and 8. The
Qwen3.8-27B NVFP4 campaign covers the MTP0 long-context profile and the complete MTP3
speculative-decode corpus at C=1, 2, 4, and 8; its C=1 point also supplies the single-request MTP3
results below. The registered Qwen3.8-27B `groupwise-int` profile remains outside the published
benchmark campaign.

Every inherited campaign is single-GPU except **Dual-GPU (TP2) and YaRN 1M context**, which
is measured across two RTX 5090s and under a power limit the single-GPU campaigns were not; the two
sets are not comparable to each other.

The single-request corpus requests were submitted serially to a persistent `ninfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## Single-request serving performance method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime | 13.1 / 13.1 |
| CUDA driver API | 13.3 for NVFP4 and refreshed 35B MTP3; 13.1 for the remaining single-request campaigns |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens; 131,072 for refreshed NVFP4 MTP3 |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| Greedy profile | Exact argmax (`--sampling greedy` in the corpus runner) |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash block=8 | `--spec dflash --draft-tokens 7 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The speculative-decode corpus contains three long-reasoning fixtures with thinking enabled and a
65,536-token output limit, followed by twelve fixtures covering code, story, translation, and
structured output. The cross-scenario fixtures disable thinking and use a 4,096-token output limit.
The tables report actual completion lengths rather than assuming that every request reaches its
limit.

Metrics are computed from the server's unrounded phase timings and speculative-decode counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
spec_acceptance = accepted_tokens / drafted_tokens
spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
```

Decode throughput is a transport/execution measurement, not a correctness score. The response text,
finish reason, and fixture-level structural requirements are audited separately below. A request
that exhausts its output budget or enters a repetition loop remains useful as a sustained-decode
stress sample, but is not presented as a successfully completed task.

## Qwen3.8-27B NVFP4 concurrent MTP3 corpus makespan

This campaign uses the complete speculative-decode corpus described above: three long-reasoning
fixtures and twelve cross-scenario fixtures, each with five fixed seeds, for 75 requests. The
runner shuffles that fixed request set once with seed `20260811` and preserves the same ordered HTTP
send sequence at every concurrency. Exactly C persistent client workers each submit their next
request only after receiving the current response. C=1 is therefore a serial single-request corpus
on one persistent server and supplies the per-fixture Qwen3.8 results in the final section.

Each point starts a fresh server on an RTX 5090 with CUDA 13.1 compile/runtime, CUDA driver API
13.3, stochastic sampling, INT8 group-64 KV, a 1,024-token prefill chunk, CUDA Graphs, prefix reuse
disabled, a 131,072-token per-request context ceiling, `--kv-capacity auto`, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Makespan begins when all client workers are released
and ends when the final complete HTTP response has been read. Prefill and decode rates divide the
corresponding server token totals by that full makespan; average batch includes the entire run,
including workload transitions and drain.

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Prefill tok/s | Decode tok/s | Avg batch | MTP acceptance | Speedup vs. C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 15,460 | 752,160 | 4,670.27 | 0.0161 | 3.3 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 15,460 | 739,951 | 2,510.78 | 0.0299 | 6.2 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 15,460 | 713,384 | 1,647.74 | 0.0455 | 9.4 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 15,460 | 723,602 | 2,164.90 | 0.0346 | 7.1 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which constrains effective
batching and makes the end-to-end result slower than C=4. Sampling is stochastic: prompts, seeds,
and send order are fixed, but concurrency-specific numerical routes can change sampled
continuations and their lengths. The makespan speedup is therefore a fixed-workload serving result
rather than a fixed-token normalization; the exact decode-token totals are retained in the table.

## Concurrent MTP3 decode saturation

The concurrent campaign uses the `long_decode_aime26_15` fixture with thinking enabled. The
rendered prompt is 293 tokens, and every request has an 8,192-token output budget. For each
concurrency C, the runner starts a fresh `ninfer-serve` process with `max_concurrency=C`, releases
C non-stream requests together using distinct fixed seeds, and waits for every HTTP response.
Startup and server warmup occur before the measured wave.

All points use an RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3, stochastic sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0), INT8 group-64 KV, a 1,024-token
prefill chunk, CUDA Graphs, prefix reuse disabled, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Each request has a 16,384-token context ceiling.
`--kv-capacity auto` resolved to exactly `C * 16,384` tokens at every point.

Saturated throughput uses only complete one-second server intervals satisfying all of the following:

- computed prefill tokens are zero;
- `running=C`, `prefilling=0`, and `decode_ready=C`;
- at least one decode round completed;
- every decode round had exactly C rows.

Ramp-up, prefill, and drain intervals are excluded. The reported aggregate rate is:

```text
steady_decode_tok_s = sum(committed_decode_tokens) / sum(interval_seconds)
```

Wave makespan starts when the client threads are released and ends after the last complete HTTP
response. MTP acceptance is aggregated over the complete wave. Each row below is one sustained
wave rather than a repeated-sample mean.

| Model profile | C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.6-27B `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.6-27B `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.6-27B `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.6-27B `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.6-27B `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.6-27B `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.6-27B `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B `groupwise-int` | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B `groupwise-int` | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B `groupwise-int` | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B `groupwise-int` | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

All 45 requests reached their output limit, producing 368,640 completion tokens. The campaign
contained 608 complete full-batch steady intervals and had no request, CUDA, or out-of-memory
failure. At C=8, available device memory after startup was 2.66 GiB for 27B groupwise-int,
2.18 GiB for 27B NVFP4, and 4.38 GiB for 35B-A3B.

## Dual-GPU (TP2) and YaRN 1M context

This inherited campaign was measured on the Qwen3.8-27B
NVFP4 artifact across two RTX 5090s with `--tp 2 --devices 0,1`, INT8 group-64 KV, CUDA Graphs
enabled, and greedy decoding. The extended-context rows additionally use
`--rope yarn --yarn-factor 4.0 --yarn-origin 262144`.

**Power condition.** The campaign was measured with both GPUs held at a **400 W per-GPU cap** --
the minimum settable limit on these cards; the vendor defaults on the measurement host are 600 W
and 575 W, and the maximum is 600 W on both. Sampled draw sat at 346-353 W against the cap, so the
limit was binding. The publishable subset was then **re-measured with both cards at 575 W**, and
the tables below carry both conditions. Lifting the cap helps `--tp 1` considerably more than
`--tp 2`: one card running the whole model saturates its limit (peak sampled draw 575.5 W) while
two cards sharing it peak at 391 and 406 W, so the TP2-over-TP1 decode advantage narrows from
1.44x to 1.40x. Never quote a figure from this section without its power condition. The single-GPU
campaigns above were measured under neither condition and are not comparable to these rows.

### Method

| Setting | Value |
|---|---|
| GPUs | 2 x NVIDIA GeForce RTX 5090, 32 GiB, no NVLink, peer-to-peer unavailable |
| Power limit | 400 W per GPU (campaign) and 575 W per GPU (re-measurement); vendor defaults 600 W / 575 W, maximum 600 W on both cards |
| Artifact | Qwen3.8-27B NVFP4 |
| Tensor parallel | `--tp 2 --devices 0,1` |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefill chunk | 1,024 tokens |
| Sampling | Greedy (exact argmax) unless a row states otherwise |
| Rope | `native` at 262k; `yarn` factor 4.0, origin 262,144 above it |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |

### Single request, matched 249,955-token prompt

Byte-identical prompt on both widths, 512 generated tokens. TP1 ran at `--max-context 252928`,
the largest window that fits one card after weights; TP2 ran at the full `262144`.

| Metric | TP1 @400 W | TP2 @400 W | TP2/TP1 | TP1 @575 W | TP2 @575 W | TP2/TP1 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill tok/s | 2,269.8 | 2,680.1 | 1.18x | 2,484.2 | 2,787.0 | 1.12x |
| Decode tok/s, MTP off | 52.35 | 75.18 | 1.44x | 53.95 | 75.32 | 1.40x |
| Decode tok/s, MTP3 | 101.7 | 152.1 | 1.50x | 113.60 | 159.39 | 1.40x |
| MTP3 draft acceptance | 50.83% | 57.96% | 1.14x | 50.83% | 57.96% | 1.14x |
| Time to first token, s | 111.0 | 93.7 | 0.84x | 101.0 | 90.0 | 0.89x |
| Per-GPU resident memory | 27.90 GiB (one card) | 15.04 GiB (each card) | | 27.90 GiB | 15.04 GiB | |
| Peak sampled draw, MTP off | — | — | | 575.5 W | 390.9 / 405.7 W | |
| Peak sampled draw, MTP3 | — | — | | 575.8 W | 483.8 / 444.5 W | |

Draft acceptance is identical to four decimal places across the two power conditions (0.5083 and
0.5796), which is the expected result: power changes timing, not arithmetic.

On a 536-token reasoning prompt the same comparison is 152.0 to 189.5 decode tok/s and 56.61% to
58.06% acceptance. TP1 wins short-prompt prefill (7,582 versus 5,451 tok/s at 8,147 tokens), where
the cross-device collectives are not amortized by a long chunked prefill. **Both of these
short-prompt comparisons were measured at the 400 W per-GPU cap only** and were not re-measured at
575 W, so they must not be read against the 575 W rows above.

### Saturated concurrent decode at a 262,144-token window

One server per concurrency point, `--decode-tokens 8192`, stochastic sampling, aggregate committed
decode tok/s over complete full-batch intervals.

| Concurrency | MTP off @400 W | Speedup | MTP3 @400 W | Speedup | MTP off @575 W | MTP3 @575 W |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 93.5 | 1.00x | 172.4 | 1.00x | 94.1 | 177.8 |
| 2 | 183.0 | 1.96x | 287.0 | 1.67x | not re-measured | not re-measured |
| 4 | 314.3 | 3.36x | 466.0 | 2.70x | 320.3 (3.40x) | 475.2 (2.67x) |

MTP raises absolute throughput at every point while scaling less steeply from batching, which is
consistent with its single-lane throughput already sitting closer to the ceiling batching pushes
toward. Engine-accounted per-GPU memory at C=4 was 14.97 GiB (MTP off) and 15.64 GiB (MTP3).

### Extended context, single request

Served from one `ninfer-serve` process at `--max-context 1048576 --max-concurrency 1`. Prefill and
decode are the server's own phase timings.

| Prompt tokens | MTP | Power | Prefill tok/s | Prefill wall | Decode tok/s |
|---:|---|---|---:|---:|---:|
| 8,146 | off | 400 W | 5,517.5 | 1.5 s | 97.89 |
| 652,954-652,955 | off | 400 W | 1,348.4 | 484.2 s | 58.22 |
| 652,954-652,955 | off | 400 W (re-run) | 1,379.5 | 470.7-476.0 s | 58.69 |
| 1,045,954-1,045,955 | off | 400 W | 928.9 | 1,126.0 s (18.8 min) | 46.39 |
| 1,045,954-1,045,955 | off | 575 W | 975.1 | 1,072.6 s (17.9 min) | 48.08 |
| 1,045,954 | off, 512 generated | 575 W | 971.4 | 1,076.7 s | 46.11 |
| 1,045,954 | MTP3, 512 generated | 575 W | 975.0 | 1,072.7 s | 100.54 at 56.41% acceptance |
| 949,885 | off | 400 W | 1,011.89 | 938.7 s | 45.49 |
| 949,885 | MTP3 | 400 W | 1,010.19 | — | 99.51 at 58.65% acceptance |

The two 512-generated-token rows are a **like-for-like** MTP measurement: same prompt, same window,
same token budget, so their ratio -- **2.18x** -- needs none of the cross-window caveat the
949,885-token pair in the same table carries. The 24-token rows are needle requests, whose decode
average starts and ends at a lower position. The 400 W and "400 W (re-run)" rows at 652,954 tokens
are two measurements of the same configuration under the same power condition, taken three days
apart; the 653k tier was not re-measured at 575 W.

Ten 653k requests spread 0.3% in prefill rate and ten 1M requests spread 1.4% over a 4.5-hour run:
no thermal fade and no drift. Decode degrades smoothly with context rather than falling off a
cliff, but the decode split policy was tuned at 262k and has not been swept at 1M.

At ~950k tokens MTP3 reaches 99.51 tok/s at 58.65% acceptance on non-repeating greedy text
(400 W). That comparison against 45.49 tok/s was cross-window -- its denominator averaged a decode
from ~950k to the 1,048,576 ceiling while its numerator decoded only ~950k to ~962k -- and it was
quoted as "about 2.2x" for that reason. The like-for-like pair in the table above settles it:
**100.54 against 46.11 tok/s = 2.18x**, same prompt, same window, same 512-token budget, at 575 W.
Acceptance is flat across context: 57.96% at 250k, 58.65% at ~950k, 56.41% at 1,046k.

Two figures from the same run are reported separately and must not be quoted as the headline: a
repetitive greedy stream reaches 93.54% acceptance and 135.78 tok/s, and a
temperature-0.8 sampled stream 80.36% and 122.21 tok/s. Sampled and greedy acceptance are different
algorithms, and the repetitive figure is a loop artifact. Two different degeneration mechanisms
produce those loops: the MTP-off soak stream repeats whole turns because `--ignore-eos` suppresses
its end-of-turn token, while the MTP3 greedy stream contains no end-of-turn token at all and
collapses into a 187-token content-level loop whose first repeat begins at generated index 1,201.
(The 1,341-token acceptance row above is a suffix-period cut, `12,000 - 57 x 187`, not the loop
onset; about 140 of its tokens sit inside the loop's first block, so 58.65% is a mild upper bound
on the novel-text figure.) Sampled decoding at temperature 0.8 does not loop.

### Memory, per GPU

`Resident` is `nvidia-smi` per-process memory. It was flat across every sample of every run: a 1M
prefill adds nothing to the residency chosen at load, and the workspace peaked at 112.29 MiB inside
a 182.81 MiB reservation during a 949,863-token prefill.

| Context | MTP | Weights | Sequence | Workspace | Reserved | Resident |
|---:|---|---:|---:|---:|---:|---:|
| 262,144 | off | 10.08 GiB | 4.28 GiB | 0.18 GiB | — | 15.04 GiB |
| 262,144 | MTP3 | 10.46 GiB | 4.54 GiB | 0.19 GiB | — | 15.69 GiB |
| 1,048,576 | off | 10.08 GiB | 16.66 GiB | 182.81 MiB | 26.93 GiB | 27.41 GiB |
| 1,048,576 | MTP3 | 10.46 GiB | 17.69 GiB | 192.93 MiB | 28.42 GiB | 28.84 GiB |

`Reserved` is the CLI load summary's per-device row; the 262,144-token rows were measured through
`ninfer-serve`'s startup record and `nvidia-smi` instead. The gap of about 0.48 GiB between
`Reserved` and `Resident` is the CUDA context and driver-side allocations the planner does not
count, so the summary's `planned slack` over-reports free memory by that much. CUDA Graph residency
at 1M was 2.00-3.00 MiB per device against a 20.00 MiB allowance.

Turning MTP3 on costs a measured 1.49 GiB of reserved memory per device at 1,048,576 tokens and
0.65 GiB at 262,144. Its two dominant terms are 0.38 GiB of head weights, fixed at any window, and
1.03 GiB of MTP KV per 1M tokens of window; the remainder of each measured delta is workspace and
sequence-arena rounding. At 1M with MTP3 the margin to a 30 GiB per-device budget is 1.16 GiB.

### Reproduction

The dual-GPU campaign is driven by the same concurrency runner as the single-GPU tables, with the
tensor-parallel flags added:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation --concurrency 1 --concurrency 2 --concurrency 4 \
  --tp 2 --devices 0,1 --device 0 \
  --max-context 262144 --kv-capacity 262144 \
  --output profiles/bench/tp2_decode_saturation
```

The extended-context rows are single requests against a server started with the 1M configuration:

```bash
./build/apps/ninfer-serve out/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 --prefill-chunk 1024 \
  --request-log-jsonl run.requests.jsonl
```

Read prefill and decode from that log's `request_done.timings_seconds`, and confirm the power
condition with `nvidia-smi --query-gpu=power.limit,power.default_limit,power.draw --format=csv`
before quoting any figure.

### Cross-engine comparison against vLLM (NVFP4, 500 W per GPU)

This comparison is a **third power condition**: both cards capped at **500 W per GPU**, distinct
from the 400 W campaign and the 575 W re-measurement above. Rows from the three conditions are not
comparable with each other.

One request at a time, byte-identical needle-in-a-haystack prompts at three context tiers, 512
output tokens, temperature 0, thinking off, and a freshly booted server per tier on both sides so
that every prefill is genuinely cold -- zero prefix-cache hits, verified from each engine's own
counters. `Prefill tok/s` is `prompt_tokens / TTFT`; `decode tok/s` is measured client-side over
the streamed window. `Peak VRAM/GPU` is the peak the engine's own process held on each device, and
peak draw comes from a 3 s sampler. Every row below was taken at the 500 W per-GPU cap.

| Engine | Window served | Tier | MTP | Prompt tokens | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | Peak VRAM/GPU | Peak draw (GPU0 / GPU1) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| vLLM 0.25.1 | 750,000 | 250k | MTP3 | 249,955 | 79.14 | **3,158.5** | 110.78 | 83.75 | 28.90 GiB | 443.9 / 467.6 W |
| vLLM 0.25.1 | 750,000 | 653k | MTP3 | 652,955 | 353.84 | **1,845.3** | 41.94 | 366.03 | 29.00 GiB | 473.6 / 496.4 W |
| vLLM 0.25.1 | 750,000 | 700k | MTP3 | 699,955 | 402.56 | **1,738.8** | 41.01 | 415.02 | 28.90 GiB | 475.3 / 497.7 W |
| NInfer TP2 | 1,048,576 | 250k | off | 249,955 | 92.56 | 2,700.5 | 73.35 | 99.51 | 27.41 GiB | 385.2 / 396.8 W |
| NInfer TP2 | 1,048,576 | 250k | MTP3 | 249,955 | 91.66 | 2,726.9 | **155.80** | 94.93 | 28.84 GiB | 380.9 / 373.6 W |
| NInfer TP2 | 1,048,576 | 653k | off | 652,955 | 465.82 | 1,401.7 | 56.70 | 474.81 | 27.41 GiB | 414.1 / 440.0 W |
| NInfer TP2 | 1,048,576 | 653k | MTP3 | 652,955 | 471.01 | 1,386.3 | **118.74** | 475.28 | 28.84 GiB | 475.3 / 488.1 W |
| NInfer TP2 | 1,048,576 | 700k | off | 699,955 | 529.26 | 1,322.5 | 54.84 | 538.55 | 27.41 GiB | 415.2 / 458.8 W |
| NInfer TP2 | 1,048,576 | 700k | MTP3 | 699,955 | 532.42 | 1,314.7 | **103.42** | 537.33 | 28.84 GiB | 465.0 / 463.0 W |

vLLM 0.25.1 served `unsloth/Qwen3.8-27B-NVFP4` at `--tensor-parallel-size 2` with FP8 KV, YaRN x4
injected through `--hf-overrides`, `--max-model-len 750000`, `--max-num-batched-tokens 16768`,
FlashInfer, and `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. Its
speculative configuration is fixed at launch, so all three of its rows are MTP3. NInfer served its
own NVFP4 artifact at `--tp 2 --rope yarn --yarn-factor 4.0 --yarn-origin 262144
--max-context 1048576 --kv-dtype int8 --prefill-chunk 1024`.

The two engines counted **identical prompt token totals at every tier** -- 249,955 / 652,955 /
699,955, each server tokenizing the same prompt string independently. That is the evidence that
they were given the same input.

**Prefill goes to vLLM at every tier**, by 1.17x at 250k, 1.32x at 653k and 1.32x at 700k:
3,158.5 / 1,845.3 / 1,738.8 tok/s against NInfer's 2,700.5 / 1,401.7 / 1,322.5. The most likely
single cause is prefill chunking -- vLLM batches up to 16,768 tokens per prefill step, NInfer
1,024 -- and that is a tunable rather than a ceiling. It was not swept.

**Decode goes to NInfer, and the cause is vLLM's speculative acceptance past its native window.**
That deployment's native window is 262,144 tokens; beyond it, its MTP head still drafts 3 tokens
per step and has every one rejected. Acceptance measured from each engine's own counters, for
exactly the requests in the table:

| Tier | vLLM drafted / accepted | vLLM acceptance | NInfer drafted / accepted | NInfer acceptance |
|---|---:|---:|---:|---:|
| 250k | 606 / 311 | 51.3% | 567 / 322 | 56.8% |
| 653k | 1,533 / **0** | **0.0%** | 547 / 328 | 60.0% |
| 700k | 1,533 / **0** | **0.0%** | 602 / 310 | 51.5% |

So vLLM pays the drafter's cost for nothing at 653k and 700k, and its decode falls to 41.94 and
41.01 tok/s. NInfer's acceptance is flat across the same range, and its MTP3 decode is **2.83x and
2.52x** vLLM's there (118.74 and 103.42 tok/s). NInfer's MTP-*off* decode at those tiers (56.70 and
54.84 tok/s) already beats vLLM's speculative decode. At 250k, where vLLM's speculation still
works, vLLM decodes 110.78 tok/s against NInfer's 155.80 with MTP3 and 73.35 with MTP off.

**Context ceiling and memory.** vLLM's KV pool measured 759,297 tokens at boot (12.73 GiB, FP8,
`--gpu-memory-utilization 0.85`), with 28.90-29.00 GiB held per GPU. NInfer holds 1,048,576 tokens
-- 38% more window -- in 27.41 GiB per GPU with MTP off and 28.84 GiB with MTP3. The 700k tier sits
near vLLM's ceiling and comfortably inside NInfer's. A second boot of the same vLLM configuration
measured 760,847 tokens; the pool varies by about 0.2% between boots with the free-memory profile
at launch.

**End to end at 512 output tokens, vLLM finishes first at every tier**, because a request of that
shape is almost entirely prefill. NInfer's decode advantage repays its slower prefill beyond
roughly **4,800 output tokens at 250k, 7,600 at 653k and 8,800 at 700k** (NInfer MTP3 against
vLLM). The Qwen3.8 card's own guidance for a 1M window -- up to 262k tokens of reasoning and 131k
of final response on agentic tasks -- sits far above all three break-even points.

#### Caveats on the cross-engine rows

- **Different weights.** vLLM served `unsloth/Qwen3.8-27B-NVFP4`, an NVFP4 quantization of the base
  Qwen3.8-27B fine-tune; NInfer served its own conversion of the huihui abliterated fine-tune.
  These are different quantizations of different fine-tunes. The comparison is *engine plus
  quantization pipeline*, not a controlled same-weights benchmark. Architecture, layer count and
  hidden sizes are identical, so the prefill and decode arithmetic has the same shape, but nothing
  here isolates the engine from the checkpoint.
- **Different KV dtypes.** vLLM FP8, NInfer INT8. That affects both the memory rows and attention
  bandwidth, so it is present in both the prefill and the decode columns.
- **Different prefill chunking, unswept.** vLLM `--max-num-batched-tokens 16768` against NInfer
  `--prefill-chunk 1024`, the value the 1M configuration ships with. Neither was swept.
- **No vLLM MTP-off row.** Speculative decoding is fixed at launch and turning it off needs a
  restart with a different `--speculative-config`; that run was not made. vLLM's 653k and 700k rows
  are therefore MTP-off *behaviour* at MTP-on *cost*, which is worse than a true MTP-off run would
  be.
- **No quality claim.** Both engines ran at `temperature 0`, `seed 42`, 512 max tokens, but their
  rejection-sampling paths under speculative decoding are not guaranteed identical and no
  token-level equivalence was checked. This is a throughput comparison only.
- **n = 1 per cell.** Each row is a single request.
- **The 500 W NInfer 250k rows are not a like-for-like re-run of the 575 W 250k rows.** These ran
  YaRN at a 1,048,576-token window, to match vLLM's YaRN deployment; the 575 W rows ran native rope
  at 262,144. The roughly 2% difference between them combines the lower cap with the window change
  and does not separate the two.
- **The vLLM client needed a wrapper.** That server runs `--reasoning-parser qwen3`, which routes
  output to `delta.reasoning_content`, and it ignores a top-level `enable_thinking` field (its
  equivalent is `chat_template_kwargs`). The project probe reads `delta.content` and sends the
  top-level field, so the vLLM rows were taken with a wrapper client that times the first delta on
  either channel and sends `chat_template_kwargs={"enable_thinking": false}`. Every vLLM row
  reports its tokens arriving on the `content` channel, so thinking was off on both sides.

Two further points about vLLM's extended context, independent of the table:

- **Extended context in vLLM is a checkpoint property, not a serving flag.** An NVFP4 repackaging
  that ships without a YaRN block in `config.json` `rope_parameters` is capped at its
  `max_position_embeddings` (262,144) until one is injected at load. `--hf-overrides` does exactly
  that, and vLLM then serves this checkpoint far beyond 262,144 tokens. The difference from NInfer
  is **where the configuration lives** -- a serving flag on an unmodified artifact here, a
  checkpoint-config override there -- not a capability difference.
- **A prefix-cache measurement, not a comparison row.** One extra vLLM run replayed the identical
  250k prompt on the same server: 248,000 of 249,955 tokens served from the prefix cache (99.22%),
  TTFT 1.80 s instead of 79.14 s, decode unchanged at 108.75 tok/s.

The raw probe records, per-request server logs, power and VRAM samples, speculative counters and
the full methodology are committed under
[`eval/results/cross-engine-nvfp4/`](../eval/results/cross-engine-nvfp4/README.md).

### Measurement gaps in this campaign

- **Concurrency C=2 was measured at the 400 W cap only**; the 575 W re-measurement covered C=1 and
  C=4, and the 653k extended-context tier was not re-measured at 575 W.
- **The soak is greedy only.** A seeded temperature > 0 soak, as the decode-coverage complement, has
  not been run.
- **The cross-engine comparison has no vLLM MTP-off row**, because speculative decoding is fixed at
  vLLM's launch and that restart was not made, and **the prefill-chunk difference was not swept** on
  either engine (`--prefill-chunk 1024` against `--max-num-batched-tokens 16768`).

The design decisions and correctness gates behind these numbers are in
[Dual-GPU (TP2) execution and YaRN 1M context](maintainer/tp2-yarn-1m.md).

## Reproduction

Build `ninfer-serve` and prepare the registered `.ninfer` artifacts. The refreshed per-target
serving tables use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 262144 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_35b_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_qwen3_8_27b_nvfp4_mtp0_20260817

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_qwen3_8_27b_nvfp4_mtp3_20260817
```

The concurrent decode-saturation campaigns use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_35b_mtp3_20260811
```

Use `--mode dflash7` for the corresponding DFlash block=8 campaign; add `--sampling greedy` for
the exact-argmax profile.

Omit `--mode` and supply the two measured Qwen3.6 groupwise-int artifacts to run the complete
published Qwen3.6 MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

For the 27B NVFP4 accuracy run, start the model service with:

```bash
build/apps/ninfer-serve out/qwen3_6_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Then run the repository's full 27B reasoning suite in a separate shell:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_6_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| `long_decode_aime26_30` | 5 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 15 | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured | 15 | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`), stochastic sampling

The fixtures, five seeds, sampling parameters, and output limits are identical to MTP3. Different
speculative backends consume random values differently, so this is a fixed-workload comparison
rather than a token-identical paired-output comparison.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,495.4 ± 2,221.2 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| `long_decode_aime26_30` | 5 | 53,330.4 ± 11,198.5 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 |
| Story | 15 | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 |
| Translation | 15 | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 |
| Structured | 15 | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 |

#### Decode throughput versus MTP3

| Workload | MTP3 tok/s | DFlash tok/s | DFlash change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 726.2 | 764.1 | +5.2% |
| `long_decode_aime26_15` | 620.3 | 584.0 | -5.9% |
| `long_decode_aime26_30` | 671.9 | 638.3 | -5.0% |
| Code | 657.6 | 562.3 | -14.5% |
| Story | 456.2 | 261.7 | -42.6% |
| Translation | 649.7 | 490.8 | -24.5% |
| Structured | 770.9 | 786.4 | +2.0% |

### DFlash block=8 (`k=7`), greedy sampling

Greedy uses exact argmax; all other corpus and server settings remain unchanged. The five seeds
repeat the same deterministic generation path, so within-fixture standard deviation measures
runtime variation rather than output variation.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 6,692.0 ± 0.0 | 872.4 ± 3.3 | 74.4% ± 0.0% | 6.21 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 651.6 ± 0.6 | 58.6% ± 0.0% | 5.10 ± 0.00 |
| `long_decode_aime26_30` | 5 | 65,536.0 ± 0.0 | 994.9 ± 3.4 † | 98.0% ± 0.0% | 7.86 ± 0.00 |

† The generation is a deterministic repetition loop, not a valid AIME response. The raw rate is
retained to describe what was measured, but is excluded from performance comparisons.

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 599.8 ± 12.3 | 46.4% ± 1.4% | 4.25 ± 0.10 |
| Story | 15 | 291.5 ± 55.6 | 14.9% ± 5.7% | 2.04 ± 0.40 |
| Translation | 15 | 475.5 ± 50.6 | 33.0% ± 5.1% | 3.31 ± 0.36 |
| Structured | 15 | 869.0 ± 120.2 | 74.5% ± 13.1% | 6.21 ± 0.92 |

#### Decode throughput versus stochastic DFlash

| Workload | Stochastic tok/s | Greedy tok/s | Greedy change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 764.1 | 872.4 | +14.2% |
| `long_decode_aime26_15` | 584.0 | 651.6 | +11.6% |
| `long_decode_aime26_30` | 638.3 | 994.9 † | not comparable † |
| Code | 562.3 | 599.8 | +6.7% |
| Story | 261.7 | 291.5 | +11.4% |
| Translation | 490.8 | 475.5 | -3.1% |
| Structured | 786.4 | 869.0 | +10.5% |

### Speculative-decode output audit

The audit covers all 225 stored July responses from the 35B-A3B MTP3 stochastic-sampler, DFlash
stochastic-sampler, and DFlash greedy campaigns. It checks termination, exact repetition, and
fixture-specific mechanical constraints. AIME 1 was checked algebraically; the AIME 30 answer
(`393`) was checked by independent enumeration. This audit does not attempt to assign a subjective
quality score to prose or translations.

#### Long-reasoning answers

| Fixture | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| `long_decode_aime26_01` | 5/5 correct, natural stop | 5/5 correct, natural stop | 5/5 correct, natural stop |
| `long_decode_aime26_15` | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit |
| `long_decode_aime26_30` | 3/5 correct, 1 wrong, 1 no answer | 2/5 correct, 1 wrong, 2 no answer | 0/5 answers; all enter the same repetition loop |

The greedy AIME 30 response has an empty final-content field and fills its 65,536-token reasoning
budget. The exact line `Wait, $x_7 x_1 x_3$ is $x_7 x_1 x_3$.` occurs 2,406 times among 2,538
non-empty reasoning lines. Its 98.0% acceptance and 994.9 tok/s therefore characterize a highly
predictable pathological loop, not normal reasoning performance.

AIME 15 is also not a valid completion in any of the three campaigns: every sample exhausts the
budget without a boxed answer. Its output is long, non-convergent reasoning rather than the short
exact cycle seen in greedy AIME 30. The AIME 15 rates may be read only as sustained long-decode
throughput.

#### Cross-scenario outputs

| Category | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| Code | 1/15 natural stops; 0/15 prompt-complete | 2/15 natural stops; 0/15 prompt-complete | 0/15 natural stops |
| Story | 9/15 natural stops; the nine Chinese outputs pass requested division and minimum length | 8/15 natural stops; the eight Chinese outputs pass requested division and minimum length | 10/15 natural stops; five Chinese dialogue outputs are under length |
| Translation | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks |
| Structured | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract |

The code prompts require complete runnable multi-file deliverables, but almost all outputs end at the
4,096-token limit. The three natural-stop exceptions also contain decisive contract failures: the
MTP3 CUDA response substitutes CUDA 12.8 and an older architecture list; the DFlash CUDA response
copies FP32 input into a half-sized 16-bit allocation and passes raw `unsigned short` values to BF16
intrinsics; and the DFlash Python response never writes its advertised JSONL event stream to the
configured log file. Code throughput is therefore a truncated-generation stress result, not
successful code-generation throughput.

All English mystery samples reach the output limit with an unfinished ending. The naturally stopped
Chinese stories have the requested chapter/act counts; the MTP3 and stochastic-DFlash samples also
meet their requested Chinese-character minima. Greedy's five dialogue stories contain 3,239 Chinese
characters each, below the requested 3,500. Story results are consequently a mixed normal/truncated
workload.

All translation outputs stop naturally. Each plain-document result preserves six sections and
provides at least twenty glossary entries; each Markdown result preserves heading levels, the
six-line table, all required inline identifiers, and the exact fenced JSON object. Translation is
the cleanest cross-scenario normal-completion comparison in this corpus.

The structured prompts intentionally exceed what these generations fit into 4,096 tokens. MTP3,
stochastic DFlash, and greedy DFlash produce only 49–60, 49–58, and 57 valid JSONL records,
respectively, versus the requested 160. Their complete-width CSV ranges are 122–139, 121–143, and
133 rows versus the requested 220. No SQL output satisfies all four tables, two views, at least 80
rows, and six final analytical queries. These high-acceptance results describe predictable partial
record generation only.

The exact-line and repeated-token scan found no other response with a short-cycle collapse comparable
to greedy AIME 30. Output-limit and prompt-compliance failures above remain material even when no
repetition loop is present.

## `qwen3_6_27b`

### EvalScope reasoning accuracy

Both weight profiles were evaluated through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. All 258 samples completed and were scored for each profile.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### `nvfp4`

The fixtures, seeds, sampling parameters, output limits, and runtime options are identical to the
groupwise-int serving campaign. Quantization can change sampled tokens, so the MTP3 results are a
fixed-workload comparison rather than a token-identical output comparison.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.

## `qwen3_8_27b`

### `nvfp4`

The MTP0 table comes from the serial Long NIAH campaign described by the single-request method. The
MTP3 tables come from the C=1 point of the fixed concurrent-corpus campaign, which serially runs the
same three long-reasoning and twelve cross-scenario fixtures. Each fixture has five fixed seeds. The
tables report arithmetic mean ± sample standard deviation from the server's per-request phase
timings and speculative counters.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 5 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 5 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| `long_decode_aime26_15` | 5 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| `long_decode_aime26_30` | 5 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 15 | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 15 | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured | 15 | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |
