# NInfer documentation

Start with the [中文首页](../README.md) or [English overview](../README.en.md) for this Windows
fork's contributions, upstream sources, portable application and verified operating limits.
General engine references below include inherited capabilities that are not all qualified on
Windows; the Windows guide identifies the actual local evidence.

## User guides

| Document | Purpose |
|---|---|
| [Windows application](windows-app.md) | portable launcher, bundled dependencies, model download, tests and release packaging |
| [Windows toolchain](windows-toolchain.md) | local GRID/WDDM, isolated CUDA 12.8/MSVC, native build, real inference and capacity checks |
| [Windows performance](windows-performance.md) | measured cross-card bottleneck, mapped-host transport, counting workload and TCC comparison limits |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, dual-GPU (`--tp 2`) execution, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, tool calls, dual-GPU serving, and YaRN extended context |
| [TUI / 终端启动器](tui.md) | SSH/local terminal API launch, explicit model/device selection, MTP/DFlash presets, live logs and owned-process stop |
| [Docker](docker.md) | CUDA 12.8 / V100 image, automatic QUASAR download, custom model URLs, persistent settings and default TUI |
| [PyQt 控制台](gui.md) | 图形化启动/停止 API、TP/设备、上下文/KV、MTP/DFlash、缓存、并发和采样参数 |
| [QUASAR performance](performance.md) | local 85K code, concurrent throughput, DFlash JSONL, task matrix, numerical limits and partial GPQA |
| [llama.cpp / LM Studio](llama-comparison.md) | historical same-Q4_K_M controls and comparison conditions |
| [TP4 / NVLink](tp4.md) | separate four-V100 host, official NVFP4/native FP8, API and 1Cat controls |
| [WSL2](wsl/README.md) | experimental V100 TP2 deployment, measured bottlenecks and 0.2.0 adaptation work |
| [Other V100 measurements](performance-v100.md) | official NVFP4, Q4_K_M, transport and prefix-cache controls |
| [RAM-KV / PP2 experiments](experiments.md) | approximate retrieval and layer-pipeline measurements |
| [Alternative models / Vision](alternatives.md) | Q4_K_M conversion and opt-in official-NVFP4 Vision |
| [V100 implementation](v100-implementation.md) | kernel, transport and verification details |
| [V100 decode research / 项目调研](v100-decode-survey.md) | public V100 projects, kernel sources, and prioritized candidates for further decode optimization |
| [Source-checkpoint quality](model-quality.md) | public QUASAR quality comparisons and limitations |
| [Inherited upstream performance](performance-upstream.md) | other hardware/models, concurrent decode and extended context |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.8-27B (recommended Text/Vision profile) | `quasar-nvfp4` | [QUASAR v3](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer/tree/v3) | [source quality](model-quality.md); Text/MTP/DFlash; optional TP2 Vision with None/MTP |
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [published model card](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [published model card](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `gguf-q4-k-m` (local V100 profile) | generated locally as `qwen3_8_27b_q4_k_m.ninfer` | Text/MTP only; embedded Vision objects are validation-only |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [published model card](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

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

- [Linux V100 prebuilt package](prebuilt.md): bundled runtime dependencies, requirements and reproduction.
- [Docker](docker.md): default QUASAR download/TUI and custom model URLs.
