# NInfer V100X2

[English](README.md) · [详细测评](docs/performance.md) · [HTTP API](docs/serving.md)

面向 **双卡 Tesla V100-SXM2 16 GB** 的 Qwen3.8-27B 单请求推理，CUDA 12.8 / SM70。
本机使用 SXM2 转 PCIe，已启用 PCIe P2P，**没有 NVLink**。
主推 **QUASAR NVFP4 v3**，采用 TP2、INT8 group-64 KV 和 CUDA Graphs。
默认纯文本启动不加载视觉权重；可开启 TP2 图片/视频，复用同一模型文件。GUI/TUI 均提供默认关闭的视觉开关。

维护者个人实际体验是跟 FP8 一档的。作者公布的源模型 GPQA-D 为 90.91%，BF16 为
91.41%；AIME'26 两者均为 100%。这些是外部源模型成绩，不是本机转换后的质量分数，
也不代表数学无损。[公开来源、独立对比和限制](docs/model-quality.md)。

## QUASAR NVFP4 测评

**实际输入 85,000 tokens**，容量 180,000，保存的仓库代码任务；chunk 2560，
greedy、优化草稿头 MTP4、完整 INT8 KV、CUDA Graphs。2026-10-13 每种配置各两次冷请求，
不复用前缀、不额外热启；每次一个 prefill 输出加 **512 个计时的已提交 decode tokens**。
下表为均值 ± 样本标准差，不计模型加载和 Graph 预热。

| 配置 | Prefill tok/s | 已提交 Decode wall tok/s | 草稿接受率 |
|---|---:|---:|---:|
| 默认注意力 / MTP4 | 1313.37 ± 3.89 | **126.89 ± 0.03** | 79.02% |
| 实验长注意力 / MTP4 | 1307.41 ± 2.96 | **143.46 ± 0.01** | 76.74% |

`--volta-long-attention` 保留全部历史 KV，但改变浮点归约，输出可能变化，**默认关闭**。
只支持单请求 SM70 TP2、Text/None/MTP、INT8 KV；不能叠加 DFlash、RAM-KV 或 Vision。
整次请求分别为 68.756 / 68.585 秒，decode 提升不等于整次请求同幅提升。
固定输出窗口只测吞吐，没有评分完整代码解答。
本地 **104 道题、85K 代码填充的 GPQA** 为默认 88/104、实验 90/104；
这是部分非标准评测，不能证明质量等价。
[阶段计时、数值限制和复现](docs/performance.md#quasar-v3-final-long-context-decode-evaluation)。

DFlash7 适合草稿接受率高的结构化输出。容量 98,304、chunk 1024、完整 INT8 KV、
greedy、完整草稿头、不复用前缀；一次完整热启后两次测量。
以下使用原来的 **32 条算术 JSONL 生成任务**，不是拍视频页面的 JSON→JSONL 转换：

| 实际输入 tokens | Prefill tok/s | 已提交 Decode wall tok/s | 接受率 |
|---:|---:|---:|---:|
| 118 | 580.95 ± 0.01 | **255.31 ± 0.11** | 99.05% |
| 4096 | 1577.84 ± 1.11 | 235.97 ± 0.04 | 99.05% |
| 85000 | 1184.19 ± 1.07 | 153.12 ± 0.02 | 98.30% |

118-token 行于 2026-10-13 重测，长输入行来自 2026-10-03。六次输出均自然停止在
1190 tokens，记录值、算术和字段顺序检查通过。DFlash 在故事等任务上可能更慢。
[完整 QUASAR 任务矩阵](docs/performance.md#quasar-v3-stop-aware-task-matrix)。
短时间区间约 2500 tok/s 的 prefill 峰值不等于 85K 输入的阶段平均。

2026-10-19 最终保留组合在相同 85K 实际输入、180K 容量、MTP4 下达到
**198.105 / 198.097 committed wall tok/s**，使用两个默认关闭的实验开关
`--volta-fast-decode --volta-long-attention`。它的数值路径与上方历史结果不同，当前组合未跑 GPQA。
未达到 200 后已按用户要求停止新候选。[当前实现、质量限制与验证](docs/performance.md)。

## llama.cpp / LM Studio 对比

目前**没有同模型的 QUASAR 与 llama.cpp 对照**。已有历史测试是两边都用 Q4_K_M，
85,000 输入、180,000 请求容量、Q8/INT8 KV、greedy、MTP 最大三最小零，
当时本机尚未开启 PCIe P2P：

| 历史 Q4_K_M 路线 | Decode tok/s | Prefill |
|---|---:|---:|
| NInfer，staging 路线，两次 | 50.68 ± 0.04 | 1251.44 tok/s |
| LM Studio llama.cpp CUDA 2.33.0，一次 | 35.4977 | 185.374 秒 |

它只能作为历史参考，**不能拿来算 QUASAR 的加速倍数**：模型、输出轨迹、通信方式和
重复次数与当前成绩不同；NInfer 计时窗口为 128 tokens，LM 为 512。
早期 512-token 输入的容量测试中，NInfer 约 60 tok/s，
LM Studio 约 62.5–62.8 tok/s。[完整表格与条件](docs/llama-comparison.md)。

## 构建和启动

需要 Linux、NVIDIA 驱动、CUDA 12.8、CMake 3.28+、C++20 编译器、Ninja、
pkg-config、FFmpeg 开发库和 libcurl。系统版本不足时先构建本地依赖：

```bash
bash tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_TESTING=OFF
cmake --build build-v100 --target ninfer ninfer-serve -j2
```

下载 QUASAR v3，无需本地转换：

```bash
hf download MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer SHA256SUMS \
  --revision v3 --local-dir /Models/ninfer-V100X2/quasar-v3
(cd /Models/ninfer-V100X2/quasar-v3 && sha256sum --check SHA256SUMS)
```

**TUI 终端启动器**只用 Python 3.11 标准库，不需要图形桌面或安装额外依赖：

```bash
bash tools/v100/ninfer-tui.sh
```

选择预设、编辑模型路径/显卡/上下文，查看命令后启动 API。界面显示日志、health 和
Engine 吞吐；`s` 停止自己的服务，`q` 退出。默认 QUASAR MTP4 / 180K，打开 Graph
和前缀缓存；约 198.1 tok/s 的实验组合需要显式选择。视觉预设为 98K 容量、4096 视觉预算。[完整用法](docs/tui.md)。

直接启动 API：

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 180000 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 4 --lm-head-draft --default-max-tokens 65536 \
  --request-log-jsonl profiles/serve.jsonl --log-stats-interval-ms 1000 --no-thinking
```

DFlash7 改成容量 98304、chunk 1024、`--spec dflash --draft-tokens 7`，去掉
`--lm-head-draft`。输出上限仍受剩余上下文限制。
CLI 用 `build-v100/apps/ninfer`，相同推理参数加 `--prompt TEXT --max-new N`。
多卡主机可用 `--tp 2 --devices 4,6` 选其中两张；设置
`CUDA_VISIBLE_DEVICES=4,6` 后设备会重编号，此时填 `0,1`。
[测试用卡选择](tests/README.md#build-and-run)。

## 专题文档

| 文档 | 内容 |
|---|---|
| [TP4 / NVLink](docs/tp4.md) | 远端四卡、官方 NVFP4/native FP8、API 与 1Cat 对照 |
| [其他 V100 测评](docs/performance-v100.md) | 官方 NVFP4、Q4_K_M、P2P/staging、前缀缓存 |
| [RAM-KV / PP2 实验](docs/experiments.md) | 稀疏检索与启动时固定的分层流水线 |
| [替代模型 / 视觉](docs/alternatives.md) | Q4_K_M 转换、QUASAR 图片/视频入口 |
| [V100 实现说明](docs/v100-implementation.md) | kernel、通信和数值验证细节 |
| [上游继承数据](docs/performance-upstream.md) | 其他硬件、模型、并发和长上下文测评 |
| [文档索引](docs/README.md) | CLI、API、GUI、测试和维护参考 |

主推路线是 **张量并行 TP2**。PP2 是单独的启动时配置，不计入以上 TP2 成绩。

## 致谢与许可

本分支基于 [Neroued/ninfer](https://github.com/Neroued/ninfer)、RTX 3060 TP2 工作和
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)，并参考了
[plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo) 的 Volta NVFP4 路线。
Prefill 研究参考 [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。
DFlash2 参考 [Inco AI](https://inco.ai/blog/dflash2/) 和
[草稿 checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)；
[KVMem](https://github.com/kvmem/kvmem-llama.cpp) 为 RAM-KV 检索提供参考。
[Li3age](https://github.com/Li3age) 在 [PR #1](https://github.com/tuxKOH/ninfer-V100X2/pull/1)
贡献实验 TP2 视觉与可配置视觉预算。

模型来自 [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B)，QUASAR 来自
[QUASAR-QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)，
NInfer 转换来自 [MirkoCovizzi](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer)。
项目使用 [Apache-2.0](LICENSE)，相关声明见 [NOTICE](NOTICE)。
