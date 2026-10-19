# QUASAR source-checkpoint quality

Maintainer's personal experience: "It feels in the same tier as FP8."

Public **source-checkpoint quality** results support near-BF16 task accuracy, not mathematical
losslessness. The [QUASAR authors' model card](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4#quality-and-size-comparison)
reports GPQA-D over two runs (396 answers) and AIME'26 over three repeats (90 answers):

| Source checkpoint | GPQA-D (%) | AIME'26 (%) |
|---|---:|---:|
| BF16 original | 91.41 | 100.0 |
| QUASAR NVFP4 | 90.91 | 100.0 |

[Rieker's independent comparison](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192)
uses a DGX Spark GB10 with vLLM, FP8 KV and MTP5, not this V100 runtime. MBPP/HumanEval/GSM8K
use thinking off, with 257/164/500 cases; PPL uses 323 windows and text KLD uses 49 prompts:

| Source checkpoint | MBPP (%) | HumanEval (%) | GSM8K (%) | PPL ↓ | Text KLD ↓ |
|---|---:|---:|---:|---:|---:|
| BF16 original | 70.8 | 93.3 | 97.0 | 7.993 | reference |
| Official FP8 | 69.3 | 95.1 | 97.4 | 8.029 | 0.0117 |
| QUASAR NVFP4 | 68.9 | 93.9 | 96.8 | 8.247 | 0.0682 |

Task scores are close to BF16, but distribution fidelity is not equal to FP8. Neither public
evaluation measures this converted `.ninfer` artifact on V100; broad quality equivalence is
not established by our inference checks. V100 dequantizes these weights without A4 activation
quantization. V3 is a container upgrade, not a new QAT checkpoint.

The runtime now admits optional TP2 image/video Vision with None/MTP from the same v3
container. Text-only startup uploads zero Vision objects; enabled startup uploads the 333
Vision objects only to the primary GPU. They derive from the converter's official Qwen source,
not a separate QUASAR QAT Vision tower. Real chart and five-second temporal-video checks
exercise MTP4, but are functionality checks, not a scored multimodal quality benchmark.
The Vision preset starts with prefix reuse disabled; it may be enabled manually. The existing
QUASAR TP2/MTP4 prefix integration test passes at 4096 capacity, including its visual bridge.

The default-off fast numeric and long-attention profiles can change logits/output. Prior
25-point probes match all argmax but only 9/25 meet near-lossless bounds (min cosine
0.998830814, max KL 0.001903103). The final ~198.1 tok/s combination has no current GPQA
result; source-checkpoint scores above do not qualify that implementation. The full v3 file
occupies 18.42 GiB on disk. Official NVFP4 and GGUF-derived Q4_K_M remain registered alternatives.
