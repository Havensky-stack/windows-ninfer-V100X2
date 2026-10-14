# Other V100 profiles and transport measurements

Official NVFP4, GGUF-derived Q4_K_M, pre-v3 DFlash and peer-off controls are kept here. The recommended QUASAR results are in [Performance](performance.md); remote four-card results are in [TP4/NVLink](tp4.md).

## P2P enabled

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

The separate [capacity sweep](#nvfp4-v3-capacity-sweep-fixed-512-token-input) holds
occupancy at 512 tokens and uses 256 timed decode tokens, chunk 1024, one discarded warmup
and three measured requests at each of 1024/2048/4096/8192/16384/32768/65536 capacities.
It measured 105.96–106.12 wall decode tok/s, 1438.44–1443.78 prefill tok/s and 70.04% acceptance.
All capacities were allocated exactly, and all 21 runs reproduced the same 257 output IDs and
173/247 accepted drafts over 83 rounds per request. Capacity is not occupied context.

The local reports are `profiles/bench/p2p_capacity/summary.json`,
`profiles/bench/p2p_occupancy/ninfer-<input-token-count>.json` for 3K–64K, and
`profiles/bench/nvfp4_v3_85k_p2p_identity.json` for 85K. Raw reports and corpora are not committed.

## Current request phases

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

## Communication and real-model regression

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

## Qualified NVFP4 prefill update

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

## Exact greedy and small-message P2P update

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

## NVFP4 v3 capacity sweep: fixed 512-token input

Only the maximum context changes. TP2, INT8 KV, greedy MTP3, optimized draft head, CUDA Graphs
and 1,024-token chunks stay fixed. Each capacity uses one discarded warmup and three measured
requests, with 512 input tokens and 256 timed decode tokens (257 total output tokens).
Decode here uses first-token-to-completion **wall time**, not just Engine decode-phase time.

| Maximum context | Wall decode tok/s | Prefill tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 1,024 | 106.12 ± 0.01 | 1,443.78 | 70.04% |
| 2,048 | 106.12 ± 0.04 | 1,438.44 | 70.04% |
| 4,096 | 105.96 ± 0.02 | 1,440.81 | 70.04% |
| 8,192 | 105.98 ± 0.01 | 1,442.96 | 70.04% |
| 16,384 | 106.06 ± 0.04 | 1,443.37 | 70.04% |
| 32,768 | 106.00 ± 0.03 | 1,443.77 | 70.04% |
| 65,536 | 106.01 ± 0.06 | 1,443.42 | 70.04% |

All seven capacities were allocated exactly. All 21 measured runs produced the same 257 IDs
and acceptance statistics: 173/247 drafts accepted over 83 rounds per request. This is a
short-input capacity check, not a 64K-filled-context result.

## DFlash2 v3: separate capacity

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

## Stop-aware non-code workloads: DFlash7 versus MTP3

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

### Chinese story

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

### English → Chinese translation

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

### 32-record JSONL

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

### Five-slot logic puzzle

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

### Whole-request interpretation and output qualifications

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

## Reproduction on the current host

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

## P2P disabled

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

## Prefill and external optimisation review

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

## NVFP4 prefill and decode

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
[P2P-disabled README area](#p2p-disabled) summarizes these earlier results.
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

## NVFP4 full-request ledger

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

## External references and integration boundaries

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

## Historical pre-v3 DFlash2 route

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

## PCIe-only transport optimization

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

The later [PP2 experiment](experiments.md#experimental-pp2-layer-pipeline) implements a startup-fixed 32/32
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
