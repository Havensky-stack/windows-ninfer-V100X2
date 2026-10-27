# llama.cpp / LM Studio comparison

No current matched QUASAR-versus-llama.cpp measurement is available. The results below use earlier Q4_K_M or pre-v3 DFlash profiles; they cannot establish a QUASAR speed ratio or equal quantization quality.

## Historical 85K Q4_K_M control

A historical diagnostic run before PCIe P2P was enabled used LM Studio's CUDA
backend **2.33.0** with its automatic two-GPU split, the same source GGUF and exact 85,000
prompt IDs, Q8 KV, a requested 180,000-token capacity
(rounded by the backend to 180,224), and maximum-three/minimum-zero MTP. Both engines used greedy
sampling. The NInfer staging mean below uses two 128-token timed decode windows; the LM
diagnostic uses 512 timed decode tokens plus one prefill output. Output windows and repetitions
are different, so the rows are not a controlled speed-ratio comparison.

| Engine | Decode tok/s | Prefill | MTP acceptance |
|---|---:|---:|---:|
| NInfer V100X2, historical staging mean of two runs | **50.68** | **1,251.44 tok/s** | 78.07% |
| LM Studio CUDA 2.33.0, historical one run | **35.4977** | 185.374 s | 365 / 440 (82.95%) |

The LM run decoded for **14.42347 s**, evaluated all 85,000 prompt tokens without cache reuse,
and stopped at the output limit without any EOS/EOG token. The LM row used a different 513-token
output window and the older NInfer transport, so it is retained only as historical cross-engine
context, not as a current like-for-like comparison.

## Historical capacity sweep

This earlier Q4_K_M sweep changes only the configured maximum context. It uses a 512-token code prompt,
a 256-token decode window, greedy sampling, Q8/INT8 KV and MTP3. Each cell is the mean ± sample
standard deviation of three measured runs after one warmup; prompt occupancy is only 512 tokens.
NInfer uses TP2 and its optimized draft head. LM Studio CUDA 2.33.0 uses automatic two-GPU
splitting with maximum-three/minimum-zero drafts.

| Maximum context | NInfer decode tok/s | LM Studio decode tok/s |
|---:|---:|---:|
| 1,024 | 60.06 ± 0.02 | 62.67 ± 0.24 |
| 2,048 | 60.12 ± 0.02 | 62.78 ± 0.05 |
| 4,096 | 60.11 ± 0.03 | 62.69 ± 0.10 |
| 8,192 | 60.13 ± 0.06 | 62.78 ± 0.07 |
| 16,384 | 60.14 ± 0.04 | 62.63 ± 0.06 |
| 32,768 | 60.16 ± 0.05 | 62.58 ± 0.004 |
| 65,536 | 60.06 ± 0.05 | 62.49 ± 0.16 |

Both engines honored all seven capacities. The output IDs were identical within each engine and all
windows were EOS/EOG-free. With this short prompt neither engine slowed materially as the capacity
increased; LM Studio was about 4% faster. This table is a capacity-setting comparison, not an
85K-filled-context benchmark.

## Historical DFlash3 control

For an external engine control, the same raw 85,000-token IDs were sent to the local llama.cpp
build with 98,304 capacity, Q8 KV, DFlash3, and a 128-token fixed window. It produced 30.1 tok/s
(121 drafted, 87 accepted); NInfer's matching 128-token DFlash3 run produced 25.3 tok/s (110
drafted, 91 accepted). A separate older short-prompt experiment is excluded because its workload
was not comparable to this 85K result.

## Reproduction

Use `tools/v100/compare_llama.py` for saved token-ID inputs and `tools/v100/bench_capacity.py` for capacity sweeps. [Historical workload commands](performance-v100.md#pcie-only-transport-optimization) record the source GGUF and LM Studio backend invocation. A new matched comparison must use the same model/quantization, actual prompt, sampling, KV format, cache policy, output window and repetition count. Capacity is not prompt occupancy.
