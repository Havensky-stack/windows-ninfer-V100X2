# Four-card V100 / NVLink deployment

These results belong to a separate four-card NVLink host. They are not local V100X2 QUASAR measurements.

### NVLink deployment: TP2 stage

Measured on 2026-10-02 on a second machine with four Tesla V100-SXM2 16 GB cards, all pairwise
NV2, 300 W per card, driver 580.178.04 and CUDA 12.8 / SM70. **NInfer uses only devices 0,1
at TP2 in this table.** The other two GPUs remain idle. Direct NVLink P2P is qualified despite
DMA-FQ host IOMMU domains; no GRUB, driver or reboot changes were needed.

Official Qwen3.8-27B NVFP4 v3, INT8 group-64 KV, 180000 context capacity (180032 allocated),
chunk 2560, greedy optimized MTP3 and CUDA Graphs. Two cold requests per row, no prefix reuse
or extra request warmup; loading and graph priming are excluded. Each request produces one
prefill token plus 512 committed timed decode tokens. Saved prompts retain the final coding
task and assistant suffix. Mean ± sample standard deviation:

| Actual input tokens | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | MTP acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2101.49 ± 4.03 | 111.19 ± 0.01 | 111.10 ± 0.01 | 72.16% |
| 8192 | 2114.14 ± 6.82 | 109.04 ± 0.03 | 108.91 ± 0.03 | 70.67% |
| 16384 | 2050.33 ± 1.22 | 104.46 ± 0.02 | 104.34 ± 0.02 | 69.00% |
| 32768 | 1887.71 ± 0.55 | 101.99 ± 0.07 | 101.89 ± 0.08 | 74.47% |
| 65536 | 1604.61 ± 3.27 | 90.28 ± 0.03 | 90.23 ± 0.03 | 76.39% |
| 85000 | 1463.73 ± 4.35 | 85.53 ± 0.04 | 85.50 ± 0.05 | 77.11% |

All repetitions reproduce all 513 output IDs and speculative counters, with no EOS/EOG in the measured window.
These fixed coding windows are performance measurements, not scored complete solutions or a
matched motherboard/NVLink A/B. Reports reside on that machine under
`results/tp2-nvlink/`; [bench_tp.py](../tools/v100/bench_tp.py) runs the occupancy, capacity and
plain-decode matrices sequentially through `ninfer_bench`.

Capacity-only sweep: the same **512-token input**, chunk 1024, MTP3, two repetitions and
**256 timed decode tokens** at every capacity (512 timed outputs would overflow the 1024 row).
Every row allocates exactly its requested capacity and reproduces all 257 output IDs and
speculative counters. Acceptance is 70.33% throughout; these rows are not long-context inputs.

| Context capacity | Prefill tok/s | Committed decode tok/s | Wall decode tok/s |
|---:|---:|---:|---:|
| 1024 | 1669.31 ± 29.44 | 112.60 ± 0.04 | 112.56 ± 0.03 |
| 2048 | 1676.79 ± 30.19 | 112.86 ± 0.01 | 112.82 ± 0.01 |
| 4096 | 1666.35 ± 33.44 | 112.70 ± 0.02 | 112.66 ± 0.02 |
| 8192 | 1670.13 ± 31.18 | 112.71 ± 0.02 | 112.67 ± 0.02 |
| 16384 | 1669.02 ± 34.89 | 112.91 ± 0.03 | 112.86 ± 0.03 |
| 32768 | 1668.32 ± 37.79 | 112.81 ± 0.04 | 112.75 ± 0.03 |
| 65536 | 1670.67 ± 27.88 | 112.97 ± 0.01 | 112.90 ± 0.02 |

The ordinary-decode control uses the same saved inputs, capacity 180000, chunk 2560 and
512 timed outputs, but `--spec none --draft-tokens 0` without the draft head:

| Actual input | Plain prefill tok/s | Plain decode tok/s | MTP3 decode tok/s |
|---:|---:|---:|---:|
| 3072 | 2132.37 ± 8.53 | 39.584 ± 0.004 | 111.19 ± 0.01 |
| 85000 | 1480.16 ± 4.93 | 28.349 ± 0.003 | 85.53 ± 0.04 |

Within each control the repeated output IDs agree. MTP/plain outputs match at 85K, but not
at 3K; these speed ratios do not prove universal token parity or a no-quality-loss result.
The [phase accounting](#separate-nvlink-deployment-tp2-stage) separates
model loading, prompt preparation, prefill and committed decode.

### NVLink deployment: NInfer TP4

Measured on 2026-10-03 on the same four-card NV2-linked V100 host above, with all four
devices active. Official Qwen3.8-27B NVFP4 v3, capacity 180000 (180032 allocated), INT8
group-64 KV, chunk 2560, greedy optimized MTP3 and CUDA Graphs. Two cold requests per
point, one prefill output plus 512 committed timed decode outputs, no prefix reuse or
extra warmup. Load/graph priming are outside request timing. Mean ± sample standard deviation:

| Actual input tokens | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | MTP acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2381.83 ± 37.81 | 125.68 ± 0.08 | 125.42 ± 0.08 | 73.13% |
| 8192 | 2415.01 ± 15.32 | 136.81 ± 0.07 | 136.61 ± 0.07 | 80.62% |
| 16384 | 2381.76 ± 9.29 | 124.09 ± 0.08 | 123.82 ± 0.09 | 72.31% |
| 32768 | 2279.47 ± 2.72 | 117.96 ± 0.02 | 117.73 ± 0.02 | 74.47% |
| 65536 | 2072.25 ± 0.66 | 109.12 ± 0.13 | 108.99 ± 0.13 | 79.30% |
| 85000 | 1957.93 ± 2.16 | 102.60 ± 0.08 | 102.51 ± 0.08 | 78.56% |

Every pair reproduces all 513 output IDs and speculative counters, without EOS/EOG. At
85K, this is 33.76% more prefill and 19.95% more decode throughput than the earlier TP2
snapshot. Acceptance and arithmetic grouping differ, so this is not isolated TP-width scaling
or proof of identical continuations across widths. Reports are in `results/tp4-nvfp4/` on
the remote host; these fixed windows are not scored complete code solutions.

Four-card NVFP4 graph/eager, sampled routing, B=2 MTP and 64 strict teacher-forced positions
pass, with zero argmax disagreements. Prefix checkpoint replay, repeated append, exact
frontiers and partial-round truncation pass exact state controls. A cold-vs-cached choice at
position 28 was a BF16 tie (zero logit deficit); stop-truncation and output-budget truncation
produce identical append outputs, logits and acceptance. Cold re-prefill is not claimed
bit-identical to retained-state execution.

### Four-card 1Cat/vLLM deployment control

For the requested friend-host setup, the isolated launchers under `/home/z/1cat-vllm-v100`
used all four NV2-linked V100-SXM2 cards, the existing OneCat Studio model directories and
loopback ports 6100/6101. The original OneCat UI on port 8888 was not modified. With
180000 maximum context, MTP3 and a 1,570-token benchmark prompt plus 256-token stream:

| 1Cat model | TTFT | streamed decode |
|---|---:|---:|
| Qwen3.8-27B NVFP4 | 0.955 s | 83.03 tok/s |
| Qwen3.8-27B FP8 | 1.590 s | 83.23 tok/s |

These are single short requests through the HTTP streaming client, not 85K occupied-context
measurements and not NInfer TP4 results. The launchers use the model's own FP8/NVFP4 kernels;
NInfer's native block-scaled FP8 identity is measured separately. After the control runs, the
1Cat processes were stopped and the NInfer TP4 API was started on loopback 6200.

The deployed TP4 API launch is `bash tools/v100/serve-tp4.sh /home/z/Models/ninfer/qwen3_8_27b_nvfp4.ninfer`:
loopback port 6200, all four NVLink cards, 180K capacity, MTP3, one active request, 65536
default output budget and prefix reuse. Text-only OpenAI/Responses/Anthropic requests,
unsupported-image rejection and turn-checkpoint replay pass.
The friend can connect through an SSH tunnel; the API is not exposed on a public interface:

```bash
ssh -N -L 6200:127.0.0.1:6200 -p 12031 z@YOUR_HOST
curl http://127.0.0.1:6200/health
```

For a graphical launcher, run `bash tools/v100/ninfer-gui.sh`. The PyQt console exposes the
artifact, TP/device mapping, context/KV capacity, prefill chunk, MTP/DFlash draft window,
CUDA Graph/prefix reuse, concurrency, output limit and sampling defaults, while validating
unsupported combinations before starting `ninfer-serve`. See [`docs/gui.md`](gui.md).

**NVFP4 and native block-128 FP8 TP4 model/state gates, prefix replay, occupancy/capacity
controls and the TP4 HTTP deployment are complete.**

Native FP8 TP4 occupancy (same 180K capacity, INT8 KV, MTP3, two cold repetitions and
512 committed decode tokens) is:

| Actual input | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | Acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2415.00 ± 13.42 | 98.75 ± 0.02 | 98.59 ± 0.02 | 80.04% |
| 8192 | 2445.95 ± 5.48 | 100.16 ± 0.01 | 100.07 ± 0.02 | 80.44% |
| 16384 | 2404.93 ± 4.30 | 95.48 ± 0.01 | 95.31 ± 0.02 | 77.27% |
| 32768 | 2298.05 ± 0.04 | 89.15 ± 0.01 | 89.02 ± 0.01 | 74.68% |
| 65536 | 2089.18 ± 1.61 | 85.05 ± 0.11 | 84.99 ± 0.11 | 80.22% |
| 85000 | 1972.43 ± 2.89 | 81.53 ± 0.09 | 81.47 ± 0.09 | 80.58% |

The native FP8 capacity sweep (512-token input, 256 committed outputs, chunk 1024) measured:

| Capacity | Prefill tok/s | Decode phase tok/s | Decode wall tok/s |
|---:|---:|---:|---:|
| 1024 | 2003.40 ± 38.05 | 101.52 ± 0.06 | 101.49 ± 0.07 |
| 2048 | 1996.61 ± 50.16 | 101.43 ± 0.08 | 101.40 ± 0.08 |
| 4096 | 1997.52 ± 47.56 | 101.48 ± 0.01 | 101.45 ± 0.01 |
| 8192 | 1986.08 ± 56.70 | 101.12 ± 0.05 | 101.10 ± 0.05 |
| 16384 | 1990.54 ± 54.50 | 101.55 ± 0.01 | 101.52 ± 0.01 |
| 32768 | 2000.66 ± 49.85 | 101.33 ± 0.20 | 101.29 ± 0.20 |
| 65536 | 1999.82 ± 41.27 | 101.26 ± 0.06 | 101.20 ± 0.06 |

All allocations, output IDs and speculative counters repeated exactly. Plain FP8 controls
were 40.98 tok/s at 3072 and 33.12 tok/s at 85000. The FP8 teacher-force gate records three
speculative argmax differences (maximum emitted-logit deficit 0.125), while ordinary cold
re-prefill records one at the same 0.125 bound; this is the documented numerical envelope,
not a claim of token-by-token losslessness.

The same four-card host has passed the production four-rank collective suite and the BF16
quarter-shard projection/residual suite (10 geometry/phase cases, eager plus graph replay,
zero failures). Four-rank NVFP4/row-FP8/native block-FP8 attention and GDN projections, including
quarter-shard BF16 GDN control at T=1/4/128/2560, pass independent FP64 checks. A short four-card
NVFP4 ordinary eager request generates coherent code; this is a smoke check, not a throughput
or quality score. Local SM70 tests pass
the native block-FP8 Linear, fused SwiGLU/residual FP64 oracles, exact TP4 tensor slicing,
GQA 6Q/1KV and GDN 4QK/12V replay. These qualify operators, not a full four-card request.

The Q4 optimized proposal-head quarter shard is now registered and passes its FP64 operator
checks. Parallel proposal selection merges local FP32 winners/I32 IDs instead of gathering the
whole draft vocabulary; target logits remain complete for sampling. Local TP2 MTP graph/eager,
sampled routing, B=2 state and all 64 teacher-forced output positions pass after this change,
with zero argmax disagreements. TP4 graph residency has its own startup budget, now verified
by the four-card NVFP4 MTP startup/state gate.

The native artifact is converted from the friend's existing OneCat FP8 source, preserving
E4M3 codes and every 128×128 BF16 multiplier; no requantization is performed. BF16 embeddings,
output head and MTP stem remain BF16. The result is 31.30 GB (decimal). Its identity is
`qwen3.8-27b/fp8`, separate from the NVFP4 package's row-scaled FP8 projections.
See the [native artifact contract](maintainer/qwen3.8-27b-artifact.md#native-block-128-fp8-textmtp-package).

The selected TP4 domain is Qwen3.8-27B NVFP4/native FP8 Text/MTP on SM70. Native FP8 requires
TP4; neither TP4 route admits Vision or DFlash. Its graph/eager, MTP commit, prefix-cache,
HTTP and occupancy/capacity measurements above were obtained on the separate four-card host.
The existing TP2 and 1Cat tables above are not relabeled as TP4 measurements.

## Request phases and qualification

### Separate NVLink deployment: TP2 stage

The [NVLink deployment table](#nvlink-deployment-tp2-stage) is measured on
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
[TP4 occupancy table](#nvlink-deployment-ninfer-tp4) uses the same saved final-task
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
