# WSL2 inference qualification

This page tracks the explicitly requested WSL2 / V100 TP2 adaptation for 0.2.0.
It is separate from the native-Linux V100X2 and four-card NVLink measurements.

## Current deployment

The inspected host runs Ubuntu 26.04.1 under WSL2 on an E5-2678 v3, with two CUDA-visible
Tesla V100-SXM2 16 GB devices. Its original 0.1.2 Linux package served QUASAR v3 NVFP4,
TP2, MTP4 with the optimized draft head, INT8 KV, CUDA Graphs, 32768 context capacity
and one active request. Windows exposes driver 582.78; no Linux CUDA toolkit is installed.
The artifact is on a Windows drive mounted through 9p, while the executable is on Linux ext4.
The retained source build uses the same inference settings with cache retention disabled,
as described below. The artifact's v3 container identity matches the registered QUASAR package.

The existing startup log reports that peer access is unavailable and selects verified CUDA
UVA D2D staging. Artifact upload takes approximately 129 seconds. This is a load-time observation;
the artifact's drive location does not establish a decode bottleneck.

## Measured baseline

The 2026-10-24 HTTP control uses the installed 0.1.2 executable and its existing service settings,
temperature 0, seed 7, thinking disabled, 513 output tokens, one request at a time, and two runs
per input. The second identical request can reuse its prefix. Capacity remains 32768, distinct
from the actual input occupancy below.

| Input occupancy | Cached tokens, runs 1 / 2 | Server committed decode, runs 1 / 2 | HTTP first visible output, runs 1 / 2 |
|---|---|---|---|
| 39 tokens | 0 / 39 | 70.7 / 69.0 tok/s | 1.372 / 0.273 s |
| 15399 tokens | 0 / 15399 | 70.6 / 69.9 tok/s | 17.497 / 1.179 s |

The uncached 15399-token case reports 880.9 prefill tok/s. Both repetitions of each input
produce identical output text. First visible HTTP output includes serving/streaming overhead;
it is not an independent Engine TTFT measurement. Private reports and saved request bodies are
under `profiles/bench/wsl-20261024/`.

## Paired transport results

The current-source comparison turns off both historical branches and prefix reuse on both
executables. It keeps the same artifact, request bodies, 32768 capacity, MTP4, sampling, 513-token
output budget and single-request concurrency. The control uses the existing UVA transport;
the candidate differs by the mapped-host small-reduction route below. Each input runs twice,
and all requests report zero cached tokens.

| Input occupancy | UVA decode, runs 1 / 2 | Mapped-host decode, runs 1 / 2 | Mean improvement |
|---|---|---|---|
| 39 tokens | 72.6 / 67.5 tok/s | 136.5 / 136.4 tok/s | 70.05 → 136.45 tok/s, +94.8% |
| 15399 tokens | 73.1 / 72.4 tok/s | 138.1 / 126.7 tok/s | 72.75 → 132.40 tok/s, +82.0% |

These are server-reported committed decode rates. The independent HTTP interval after first
visible output measures 136.22 / 136.17 tok/s for the short input and 137.89 / 126.49 tok/s
for the longer input. Mean complete HTTP times fall from 7.705 to 4.105 seconds and from
22.326 to 19.327 seconds, respectively. Uncached 15K TTFT remains essentially unchanged:
15.077 / 15.486 seconds for UVA versus 15.086 / 15.807 seconds for mapped-host packets;
this is a decode communication improvement, not an established prefill gain.

All four paired output texts match exactly, with matching token counts and reported MTP
acceptance/committed-per-round counters. Short requests report 3.18 committed tokens/round
and 54.5% draft acceptance; 15K requests report 3.63 and 66.1%. These checks and the independent
operator oracles support this transport change; they are not a general model-quality evaluation.
The earlier 0.1.2 baseline has different cache settings and is retained as context, not used to
calculate the transport improvement.

The second GPU reaches 83°C in both comparisons. Its sampled active SM clocks range from
1290–1530 MHz for UVA and 1095–1530 MHz for mapped-host packets; the first GPU holds 1530 MHz
in active samples. No power limit or clock was changed. These are observed rates under the
host's varying automatic clocks, not fixed-clock predictions or an 85K-context qualification.

## Retained transport

[`wsl_transport_probe.cu`](../../tools/tp2/wsl_transport_probe.cu) checks changing payloads on both
devices eagerly and across repeated cross-device CUDA Graph replays. It compares driver-managed
UVA staging, explicit pinned D2H/H2D, event-ordered mapped-memory kernels, and GPU-only mapped
two-epoch packets. The CPU does not access live mailboxes. All four routes passed the initial
WSL transfer check; that first probe explicitly allocated portable mapped pinned memory.

At 10 KiB per rank, the observed graph round times were 291.2 / 180.6 / 99.7 / 24.0 microseconds,
respectively. At 50 KiB they were 374.4 / 314.2 / 144.9 / 79.3 microseconds. These rounds include
payload production and verification kernels. They are communication experiments, **not** model
throughput predictions or a claim that decoding improved by the same factor.

The candidate reduction route retains the independent BF16 sum contract: FP32 addition and
one BF16 store. Two 80-KiB epochs per rank consume 320 KiB pinned host memory in total; epoch
counters stay on the owning GPU. System fences publish payloads and retire empty markers.
It needs neither peer-device pointer reads nor cross-device atomics. Device-mailbox projection
fusions remain separate. Wide/scalar reductions and vocabulary gathers use the existing verified
driver-staged path. Packed model bytes, KV representation, sampling, draft count and context
capacity are not changed by the transport.

The local SM70 TP2 collective suite passes both the native device-packet route and forced
mapped-host packets with peer access disabled. The complete suite also passes on the WSL host,
including the production `cudaMallocHost` mappings. This covers independent FP64 sum/RMSNorm oracles,
special BF16 inputs, exact gathers/argmax, guard bytes, changing-size in-place/out-of-place
chains and repeated graph/eager execution. The paired comparison above additionally exercises
whole-model outputs and throughput. The WSL host-synchronized 10-KiB collective
averages 51.8 microseconds (p50 46.6, p99 97.6); this is a different measurement from the
captured synthetic rounds above.

```sh
cmake --build build-v100-tp4 --target ninfer-serve ninfer_allreduce_test -j2
LD_LIBRARY_PATH=build/_deps/install/lib:/usr/local/cuda-12.8/lib64 \
  build-v100-tp4/tests/ninfer_allreduce_test
LD_LIBRARY_PATH=build/_deps/install/lib:/usr/local/cuda-12.8/lib64 \
  NINFER_TEST_HOST_PACKETS=1 build-v100-tp4/tests/ninfer_allreduce_test
```

## Deployment and 0.2.0

The retained executable lives at `~/ninfer-wsl/retained/ninfer-serve`. The existing package's
`bin/ninfer-serve` wrapper selects it, keeps the package's bundled libraries and appends
`--prefix-cache-tokens 0 --no-prefix-reuse`. The system `ninfer.service` keeps its existing
model path, port, context capacity and MTP settings. Its original package executable remains
unchanged at `libexec/ninfer-serve`; the original wrapper is backed up under
`~/ninfer-wsl/profiles/original-ninfer-serve-wrapper`.
The restored managed service passes `/health` and two repeated HTTP inference requests;
both return the expected answer and report zero cached tokens. Its running command selects
the retained executable and includes both cache-disabling flags.

To roll back the transport while keeping caches disabled, stop the service and replace the
retained executable with `~/ninfer-wsl/current/ninfer-serve`, then start the service again.
Do not overwrite a running executable in place. Keep the existing driver, power limits and
clocks; do not install Linux GPU display drivers in WSL.

For 0.2.0, rebuild the normal source package with this qualified WSL route. Other WSL
hardware/driver combinations, Vision, concurrent requests and longer occupied contexts have
not been qualified by this campaign. No further benchmark campaign is required for the
measured single-request Text/MTP4 deployment.

At the user's request, further WSL tests and the retained deployment disable historical branches
(`--prefix-cache-tokens 0`) and prefix reuse (`--no-prefix-reuse`). The latter also disables
host GDN/tail-hidden prefix checkpoint capture. The historical token budget is a limit, not a preallocated
384000-token buffer; disabling it avoids growth as completed requests accumulate.

Check Windows commit headroom as well as Linux available memory before whole-model testing.
The inspected host has 128 GiB physical memory but a 16-MiB Windows paging file; competing host
applications left about 2.3 GiB commit headroom with the model resident. Windows logged resource
exhaustion near interrupted runs. Stopping the model recovered approximately 22 GiB commit
headroom. These events correlate with interruptions but do not establish their cause. The WSL
memory ceiling, Windows commit limit and GPU memory capacity are separate limits. Strata and
Windows paging settings remain unchanged; only NInfer cache retention is disabled.

These experimental WSL changes are included in release 0.2.0 with the deployment limitations above.
