# Windows 双 V100 性能诊断

2026-10-14，本机的主要性能差距是默认关闭 MTP，以及 GRID/WDDM、无 P2P 环境下
频繁小块跨卡交换的延迟。已测得跨卡路径的成本；尚未把它进一步拆分为驱动版本、
WDDM 调度、主机内存中转和平台拓扑各自的贡献，不能承诺升级驱动即可达到上游速度。

## 完整请求

两张 V100-SXM2 16GB，GRID 553.62、WDDM，PCIe 3.0 x16，无 P2P/NVLink。
Release 构建、CUDA 编译器/运行库 12.8、原生 SM70、CUDA Graph 开启。
模型为 Merkyor Coder390 W4A4-MTP-DFlash2 v3，身份 `qwen3.8-27b/nvfp4`。

提示词“从1数到300”，实际输入 19 tokens，容量 131072，INT8 group-64 KV，
prefill chunk 256，TP2，单请求，关闭 thinking。采样为 temperature 0.7、top-p 0.8、
top-k 20、presence penalty 1.5、seed 42，输出上限 2048，无前缀缓存命中。

| 模式 | 最终输出 tokens | 已提交 decode tok/s | 客户端请求耗时 |
|---|---:|---:|---:|
| 原传输，普通解码 | 1391 | 16.49 | 84.52 秒 |
| mapped-host，普通解码 | 1391 | 28.08 | 49.72 秒 |
| 原传输，临时开启 MTP3、optimized proposal head | 1391 | 57.07 | 24.58 秒 |
| mapped-host，临时开启 MTP3、optimized proposal head | 1391 | 94.68 | 14.87 秒 |

整数序列均严格为 1 到 300。普通解码修改前后的答案逐字节一致，已提交吞吐提高约
70.3%。decode 统计不含首个 prefill token 和模型加载。
MTP 修改前后答案也逐字节一致，吞吐提高约 65.9%；两次均接受 1044/1044 drafts，
共 348 轮。数数的接受率不能代表
一般问答，也不能消除[此前 MTP 严格行为门失败](windows-toolchain.md#mtp-尚未通过严格行为门)。
应用默认设置保持不变。

以上均为单次完整请求的前后对照，未建立跨次波动范围。94.68 tok/s 是本机短输入
数数的实验 MTP 成绩，不是已验证的一般问答速度，也不是上游 65K 占用的复现。
原始请求结果保存于本地 `build-tools/count300*`；冷却间隔不计入请求时间。

当前源码已构建为本地 `build-release/NInfer-Windows-V100X2-0.1.1/NInfer.exe`，
普通解码自动启用新传输。GitHub 已发布的 0.1.0 包仍是旧传输实现。

API 测速使用 `ninfer-serve` 和上述请求采样参数。自行观察同配置 CLI 输出可运行：

```powershell
$model = '.\models\Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer'
.\build-windows\apps\ninfer.exe $model --prompt '从1数到300' `
    --tp 2 --devices 0,1 --max-context 131072 --kv-capacity 131072 `
    --kv-dtype int8 --prefill-chunk 256 --no-thinking --max-new 2048 `
    --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --seed 42
```

实验 MTP 在末尾附加 `--spec mtp --draft-tokens 3 --lm-head-draft`。
它仍未通过下述严格行为门；便携应用默认不启用。

修改前的无分析器 CLI 对照使用相同设置、输出上限 128：decode 7.578 秒、16.76 tok/s。
因此低速在直接 CLI 推理中同样存在，GUI 和 HTTP 不是主要来源。

## 跨卡延迟证据

修改前普通解码的 64 层各有两次主干 all-reduce，共 128 次。隐藏向量有 5120 个 BF16
元素，每次每方向 10 KiB。无 P2P 时，跨卡 UVA D2D 经主机中转为 DtoH/HtoD；
实际活动记录每轮包含 512 次这种 10 KiB 拷贝，另有词表等传输。

CUPTI 局部跟踪保留了 18 个记录数一致的完整 Graph 轮次。按两卡时间线的区间并集
统计，单轮中位跨度约 61.66 ms，计算内核约 27.31 ms，全部计算/拷贝活动约
28.62 ms；约 33 ms 没有被记录的 GPU 活动。空档几乎都紧邻主机中转拷贝，
kernel-to-kernel 空档每轮合计约 0.36 ms。两卡重叠时间只计一次。

**采集限制：** Nsight Systems 2023.4 的 ETW 初始化要求管理员权限；直接 CUPTI
采集也未能完成整个请求，后续停顿原因尚未确定，诊断进程已终止。上述数据只作局部
归因，不能用采集请求总耗时报告吞吐，也不能把所有空档直接命名为 WDDM 调度。

为独立核对，关闭分析器，以 `tools/tp2/capture_probe.cu` 的三阶段事件/拷贝结构
建立 128 轮 CUDA Graph：每轮两侧各有缩放、10 KiB 拷贝和合并。预热 3 次、计时
15 次，轮次内部没有主机同步，CUDA Event 测量整张图。

| 诊断路径 | 128 轮 GPU 时间中位数 | 每轮平均 |
|---|---:|---:|
| GPU0 ↔ GPU1 | 35.782 ms | 279.54 µs |
| GPU0 上两条流，保持相同事件、拷贝和计算结构 | 1.257 ms | 9.82 µs |
| GPU0 ↔ GPU1，显式 mapped-host 原始字节传输 | 10.204 ms | 79.72 µs |

两侧初始值为 `10+j%7` 和 `100+j%7`，缩放后合并；三种路径的全部结果均与独立
精确期望 `55+j%7` 相等。约 34.52 ms 的差额与真实推理的传输相关空档相近，支持
跨卡交换延迟是主要瓶颈。该探针只测通信调度，不是单卡运行 27B 模型，差额也不等于
已经实现的整模型加速收益。诊断副本和结果保存在本地 `build-tools/count300/` 及其父目录。

传输字节本身的活动时间较短，主要成本在传输前后等待。CUDA Graph 已开启，
但没有消除这里的跨设备传输与依赖成本。

## 与上游的比较及后续方向

[继承的上游表格](https://github.com/Havensky-stack/windows-ninfer-V100X2/blob/bcfbce2e/README.zh-CN.md)
包含官方 NVFP4 v3、NVLink TP2、实际输入 65536 tokens 下的 MTP3 90.28 tok/s；
[性能参考](performance.md)还记录了关闭 P2P、短输入的 98.83 tok/s。上游同输入
P2P 开关对照只提高约 5.05%；这不能推广为本机通信代价也只有 5%。上游 Linux
非 P2P 路径与本机 GRID/WDDM 路径延迟不同。

上游 10 KiB、500 次主机同步通信微基准在关闭 P2P 时约 47.80 µs，本机此前约
1.43 ms。这是独立的主机同步测试，不能乘以 128 推算整模型时间；上面的 Graph
链测量更接近实际解码中的执行方式。

上述修改前基线未替换上游 `src/ops` 计算内核或双卡主干调度。官方 NVFP4 v3 的 MTP
权重驻留约 10.458 GiB/卡，本机为 10.46 GiB/卡，不能用明显更大的权重解释这组差距。
QUASAR 的 8.66 GiB/卡属于另一种包。上游的模型参数、输入、greedy 配置仍与本机
不同，现有结果不是严格的跨操作系统对照。

## Windows 小消息传输优化

Windows WDDM、SM70、TP2 且无直接 P2P 时，80 KiB 以内的 all-reduce 使用两块
启动时分配的 mapped pinned host mailbox。发布内核逐字复制原始数据并执行 system
fence，跨设备事件确认发布完成后，对端内核读入自己的 device staging。原有
pull-done 屏障保护下一轮复用；原有 BF16 求和内核和权重、KV 格式均不变。
大消息和 gather 继续使用原 DMA 路径；TCC、直接 P2P 和其他平台的路由不变。
这些 mailbox 由当前 Program 的通信资源拥有，Graph 重放期间不分配内存。

`ninfer_allreduce_test` 通过：1 byte、10 KiB 和 80 KiB 边界的独立原始字节检查，
包含非对齐地址、部分尾字和三次改变数据的 Graph 重放；既有 FP64 求和 oracle、
80 KiB 路由切换边界、guard、eager/Graph、连续 64 次 collective 和 gather 检查也通过。
单次主机同步的 10 KiB 微基准均值降为 578 µs；这是诊断值，不等于 Graph 内通信延迟。

4K 容量的 TP2 MTP Graph 实测占用约 26 MiB，超过此前 24 MiB 的规划，已将该小容量
拓扑的预算改为 32 MiB（20 MiB 双卡基础余量加 12 MiB greedy 拓扑）。保留启动时的
实际占用检查；128K 测速所用 94 MiB 预算不变。

修复预算后，真实 Coder390、4K 容量、MTP3/optimized 检查完成 Graph/eager 和采样
路线对照。普通 teacher-forced 对照为 0/64；MTP 仍为原来的 probe 1、位置 31，
target 2532 对 speculative 561、logit deficit 0.125（1/64）。严格测试仍返回失败，
没有放宽断言；本次传输优化未修复也未增加该已知行为差异。

该方案来自 NVIDIA 的 [CUDA 12.8 Zero Copy 建议](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html#zero-copy)：
独立显卡可直接访问映射的 pinned host memory，但适合一次性、合并访问的特定负载。
这里仍走 PCIe 和主机内存，并未启用 P2P，也不代表所有大小的传输都适合它。

## TCC 和 llama.cpp

[CUDA 12.8 Windows 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-microsoft-windows/index.html)
及 [NVIDIA SMI 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)说明：TCC 面向
计算、可降低内核启动延迟，但不能承担 Windows 显示/游戏图形渲染。用户已选择推理用
TCC、游戏恢复 WDDM，并接受重启；应在同负载软件对照后再测，不能把更换驱动或
切换 TCC 当成已证明的提速。当前 553.62 的命令选项应以本机 help 为准，不能套用
新版文档才有的自动重启控制选项。没有更改驱动、模式、功耗或时钟。

[llama.cpp 参数定义](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)
默认 split-mode 为 layer，按层拆分模型；它与 NInfer 每层做双卡张量并行的通信模式
不同，在跨卡延迟较高的平台上有优势的可能。用户 LM Studio 的实际拆分方式尚未确认。
当前没有相同模型、上下文占用和采样下的本机 llama.cpp 对照，不能宣称已超过它。
