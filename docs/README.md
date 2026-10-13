# NInfer documentation

Start with the [project README](../README.md) or [中文文档](../README.zh-CN.md) for the CUDA 12.8
V100X2 build, Q4_K_M conversion, 180,000-token INT8-KV/MTP3 usage, and separately labeled
P2P-enabled and P2P-disabled measurements. The Chinese guide also includes the NVFP4 API
launch command; general protocol examples live in the serving guide below.

## User guides

| Document | Purpose |
|---|---|
| [Windows toolchain](windows-toolchain.md) | local GRID/WDDM, CUDA 12.8, MSVC and TP2 probe qualification; full Windows port pending |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, dual-GPU (`--tp 2`) execution, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, tool calls, dual-GPU serving, and YaRN extended context |
| [PyQt 控制台](gui.md) | 图形化启动/停止 API、TP/设备、上下文/KV、MTP/DFlash、缓存、并发和采样参数 |
| [Performance](performance.md) | V100X2 85K-occupancy acceptance method and measurement status; inherited RTX 5090 single-request, concurrent-decode, MTP/DFlash and 1M-context results |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `gguf-q4-k-m` (local V100 profile) | generated locally as `qwen3_8_27b_q4_k_m.ninfer` | Text/MTP only; embedded Vision objects are validation-only |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

Runtime and Op references:

- [Small-scale concurrent inference architecture](maintainer/concurrent-inference-architecture.md)
- [Paged KV context storage, ownership, and capacity model](maintainer/paged-kv-cache.md)
- [Op admission, contracts, ownership, qualification, and performance rules](maintainer/op-development.md)
- [ReplaySSM GDN technical reference](maintainer/replayssm-gdn.md)
- [Dual-GPU (TP2) execution and YaRN 1M context](maintainer/tp2-yarn-1m.md)
- [Linear benchmark contract and registered suites](maintainer/linear-benchmark.md)

Artifact and model references:

- [NInfer artifact container](maintainer/artifact-container.md)
- [Persistent tensor numeric formats](maintainer/tensor-formats.md)
- [Persistent storage layouts](maintainer/storage-layouts.md)
- [Qwen3.6-27B model semantics](maintainer/qwen3.6-27b-model.md)
- [Qwen3.6-27B artifact contracts, including NVFP4](maintainer/qwen3.6-27b-artifact.md)
- [Qwen3.8-27B artifact contracts, including the NVFP4 target](maintainer/qwen3.8-27b-artifact.md)
- [Qwen3.6-35B-A3B model semantics](maintainer/qwen3.6-35b-a3b-model.md)
- [Qwen3.6-35B-A3B artifact contracts](maintainer/qwen3.6-35b-a3b-artifact.md)

Pending implementation work:

- [Softmax Attention organization and migration](maintainer/softmax-attention.md) describes the
  single target state for an unfinished source and public-contract cutover; it is not the current
  implementation map.
