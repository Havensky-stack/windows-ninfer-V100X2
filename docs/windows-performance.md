# Windows 双 V100 性能诊断

2026-10-14，本机的主要性能差距是默认关闭 MTP，以及 GRID/WDDM、无 P2P 环境下
频繁小块跨卡交换的延迟。已测得跨卡路径的成本；尚未把它进一步拆分为驱动版本、
WDDM 调度、主机内存中转和平台拓扑各自的贡献，不能承诺升级驱动即可达到上游速度。
同日切换双卡 TCC 后，短输入数数的普通解码、MTP3、DFlash7 分别测得
36.19、122.69、189.50 committed tok/s，较优化后的 WDDM 提高 23%–30%；
P2P 仍不可用，见下文 TCC 对照。这些结果不代表长上下文或一般问答速度。

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

## DFlash7 本机补测

2026-10-14，同一 Coder390 v3、GRID 553.62/WDDM 和 mapped-host 构建，
`--spec dflash --draft-tokens 7`，full proposal head。容量改为 **98304**，分块仍为
256，INT8 group-64 KV、TP2、Graph、19-token 数数提示、采样参数和 2048 输出上限
与前文相同。该容量与前文 MTP 的 131072 不同，不是完全同配置的后端对照。

单次完整请求输出 1391 tokens，已提交 decode **153.70 tok/s**，客户端耗时
**9.25 秒**；接受 **1217/1218** drafts（99.92%），174 轮。整数序列完整为 1..300。
相对普通解码，29 处换行变为空格，答案不逐字节相同。这个高度可预测的短输入请求
不能代表一般问答或长上下文速度。结果保存在本地 `build-tools/count300-dflash7/`。

DFlash 加载的权重合计 24.31 GiB，启动后主卡约余 270 MiB，显存余量很紧；本次
只验证该容量的短请求，不声明满容量稳定性。可复用前文 CLI 命令，将两个容量参数
都改为 `98304`，并追加 `--spec dflash --draft-tokens 7`；不要使用 MTP 的
`--lm-head-draft` 参数。

真实模型检查使用 `NINFER_V100X2_SPEC=dflash`、
`NINFER_V100X2_PROPOSAL_HEAD=full`，4K 容量。Graph/eager、greedy/sampling 路线
对照通过，但严格 teacher-forced 检查仍失败：probe 1、位置 31，target 2532 对
speculative 561，logit deficit 0.125，即 1/64；普通对照 0/64。这与 MTP 是同一处
差异，不能将 DFlash 标为已通过严格行为契约。两个验证提示的接受分别为 26/30 和
20/75，进一步说明数数的近 100% 接受率不适用于所有任务。应用默认设置不变。

## 并发测量

2026-10-14，同一 Windows/GRID/WDDM、双 V100、Coder390 v3、mapped-host 构建。
每路上限 4096 tokens，共享 KV 明确配置为 `4096 × 并发数`，INT8 group-64，
prefill chunk 256，关闭 thinking 和前缀复用。每档冷启动一个独立 server，再用线程屏障
同时发起 C 个“从1数到300”流式请求；每路实际输入 19 tokens、输出 1391 tokens，
temperature 0.7、top-p 0.8、top-k 20、presence penalty 1.5、seed 42，上限 2048。

总吞吐为所有请求的已提交 decode tokens（每路输出减去一个 prefill token）除以
从首个请求发出到最后一个请求结束的墙钟时间，包含预填充和调度时间，不含模型加载。
每路速度来自各请求的 decode 计时，因此不能简单用总吞吐除以 C 替代。
单次完整请求组，不是重复均值。后端间休息三分钟，不计入请求测量。

| 普通解码并发 | 总吞吐 tok/s | 每路 decode tok/s | 首字延迟范围 | 整组耗时 |
|---:|---:|---:|---:|---:|
| 1 | 28.28 | 28.40 | 0.19 s | 49.15 s |
| 2 | 60.37 | 30.46–30.47 | 0.17–0.40 s | 46.05 s |
| 4 | 109.52 | 27.85–27.86 | 0.17–0.81 s | 50.77 s |
| 8 | 180.23 | 23.25–23.27 | 0.23–1.83 s | 61.70 s |

所有普通解码请求均完整数到 300，前缀命中为零。各档日志确实观测到 decode batch
1/2/4/8 和对应活跃请求数，不是队列串行执行。8 路启动后主卡尚余约 2724 MiB。
引擎的活跃请求上限是 8，额外请求受有限 FIFO 队列约束；这不是硬件所有可能实现的
理论上限，也不证明 8 路可各自占满 96K/128K。启动器仍固定 1 路。

原始结果和每路答案保存在本地 `build-tools/concurrency-count300/`。

同条件 MTP3/optimized proposal head：

| MTP3 并发 | 总吞吐 tok/s | 每路 decode tok/s | 首字延迟范围 | 整组耗时 | 接受/草稿 |
|---:|---:|---:|---:|---:|---:|
| 1 | 93.09 | 94.22 | 0.17 s | 14.93 s | 1044/1044 |
| 2 | 157.77 | 80.89–80.93 | 0.19–0.43 s | 17.62 s | 2088/2088 |
| 4 | 150.42 | 38.59–38.72 | 0.17–1.02 s | 36.96 s | 4173/4176 |
| 8 | 207.88 | 27.16–27.40 | 0.17–2.67 s | 53.49 s | 8345/8352 |

MTP 各档也观测到对应的真实 1/2/4/8 decode batch，所有整数序列正确且没有前缀
命中；8 路启动后主卡约余 2134 MiB。数数接受率接近 100%，不能据此承诺一般问答
性能或消除此前的严格行为门失败。4 路总吞吐低于 2 路，说明增加并发不保证线性加速。

普通解码 8 路启动示例（多请求需由 API 客户端并行发出）：

```powershell
$model = '.\models\Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer'
.\build-windows\apps\ninfer-serve.exe $model --host 127.0.0.1 --port 8080 `
    --tp 2 --devices 0,1 --max-concurrency 8 --max-context 4096 --kv-capacity 32768 `
    --kv-dtype int8 --prefill-chunk 256 --no-thinking --no-prefix-reuse
```

实验 MTP3 追加 `--spec mtp --draft-tokens 3 --lm-head-draft`。

同条件 DFlash7/full proposal head：

| DFlash7 并发 | 总吞吐 tok/s | 每路 decode tok/s | 结果 |
|---:|---:|---:|---|
| 1 | 152.06 | 155.55 | 9.14 s 完成，首字 0.20 s，接受 1217/1218，整数序列正确 |
| 2 | — | — | 启动时 `dflash2_select_sharded: invalid tensor shape` |
| 4 | — | — | 同一张量形状错误 |
| 8 | — | — | 显存预算拒绝：运行区约需 3.09 GiB，可用约 2.56 GiB |

DFlash2 并发形状错误已定位到 `dflash_impl.h` 的 `propose_batch_impl`：
`selector_gate` 以 `[256,K*B]` 分配后直接传给要求 `[R,K,B]` 的 selector，B>1 时
第三维仍为 1，与 `[B]` anchors 和 `[K,B]` drafts 不匹配。TP1 分支的 logits/topk/gate
也以合并列数表示，不能仅删除 Op 的维度检查来解决。当前任务记录现有版本的测量和
原因，没有修改推理或放宽检查。2/4 路失败属于软件调用问题，8 路还存在独立的显存
约束；本次 DFlash7 只有 1 路获得成功结果，不能宣传其多并发速度。

因此在此次已测的 1/2/4/8 档中，最高总吞吐为实验 MTP3 的 8 路 207.88 tok/s，
普通解码 8 路为 180.23 tok/s。两者当前都能达到引擎的 8 活跃请求上限。
全部 31 个已完成请求都完整数到 300，但这不是一般问答/长上下文吞吐或数值质量认证；
MTP/DFlash 的既有严格行为门失败仍然有效。

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

切换前 WDDM 补查：CUDA Driver API `cuDeviceCanAccessPeer` 双向能力值均为 0；NVML 的双向
P2P read/write 状态均为 `IOH_TOPOLOGY_NOT_SUPPORTED`，PCI P2P 状态为
`NOT_SUPPORTED`，NVLink 链路均 inactive。当时当前和待生效驱动模式均为 WDDM。
在场设备的 Windows PnP 路径分别为 `PCIROOT(0)#PCI(0300)#PCI(0000)` 和
`PCIROOT(80)#PCI(0200)#PCI(0000)`；主板 JGINYUE X99-8D3/2.5G Server，
两颗 Xeon E5-2696 v3。两卡确实分属不同 PCIe 根。

[CUDA 12.8 P2P 文档](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)
以设备 peer-capability 查询和实际 PCIe/NVLink 拓扑为准。当前 NInfer 无法绕过这个
不支持结果启用真正 P2P；mapped-host 通信仍经过主机内存。

2026-10-14，使用原 GRID 553.62 的普通 `-dm 1` 成功设置两卡，重启后已核实
current/pending 均为 **TCC**。CUDA 双向 peer 查询 API 调用成功，但能力值仍为 **0**；
NVML 双向 read/write 仍为 **`IOH_TOPOLOGY_NOT_SUPPORTED`**，PCI 为
`NOT_SUPPORTED`，所有 NVLink 链路仍 inactive。结果保存于本地
`build-tools/driver-mode/p2p-after-tcc.json`。因此本机在当前驱动及固定拓扑下，
切 TCC 没有启用 P2P；不能将这个结果推广为所有跨根平台或 V100 永久不支持 P2P。
用户已明确无法换槽；若继续调查 NVLink，需要先核实 SXM2 底板/转接板型号和
实际互联布线，不能假设普通桥接器可用。没有尝试绕过驱动能力检查。

[CUDA 12.8 Windows 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-microsoft-windows/index.html)
及 [NVIDIA SMI 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)说明：TCC 面向
计算、可降低内核启动延迟，但不能承担 Windows 显示/游戏图形渲染。用户已选择推理用
TCC、游戏恢复 WDDM，并接受重启。当前 553.62 的命令选项应以本机 help 为准，不能套用
新版文档才有的自动重启控制选项。本次只切换驱动模式，没有更换驱动、CUDA、功耗或时钟。
正式切换入口已提交到源码：[启用 TCC](../tools/windows/enable-tcc.cmd)、
[恢复 WDDM](../tools/windows/restore-wddm.cmd)，共用
[`set-v100-mode.ps1`](../tools/windows/set-v100-mode.ps1)。脚本自动识别两张 V100，
通过 UAC 按实际查询到的 UUID 执行 `-dm 1/0`，随后需重启；没有写死本机 UUID。
只读查询使用 `-Mode Status`。切换日志保存在 `%LOCALAPPDATA%\NInferV100X2\driver-mode`。
也可在管理员终端对已核实的两卡分别执行 `nvidia-smi -i <GPU-UUID> -dm 0`；
脚本不自动重启电脑。WDDM → TCC 已实测；本轮未实际执行回切 WDDM。

### TCC 同负载对照

复用上述 Release 二进制和 Coder390 v3，TP2、Graph、INT8 group-64 KV、chunk 256，
同一“从1数到300”提示和采样参数。每次输入 19、输出 1391 tokens，无前缀缓存命中。
每种模式的请求和 Engine 配置均已与其 WDDM 记录核对相同；普通/MTP3 的容量为
131072，DFlash7 为 98304，容量不代表实际占用。各后端之间空闲至少三分钟。

| 模式 | 优化后 WDDM decode tok/s | TCC decode tok/s | 提升 | TCC 客户端耗时 |
|---|---:|---:|---:|---:|
| 普通解码 | 28.08 | 36.19 | 28.9% | 38.51 秒 |
| MTP3，optimized proposal head | 94.68 | 122.69 | 29.6% | 11.44 秒 |
| DFlash7，full proposal head | 153.70 | 189.50 | 23.3% | 7.43 秒 |

每种模式分别与对应 WDDM 答案逐字节一致，均完整数到 300；这不表示三个模式
彼此输出完全相同。MTP 接受 1044/1044 drafts、348 轮；DFlash 接受 1217/1218、
174 轮，均与各自 WDDM 相同。decode 吞吐为 1390 个已提交 decode tokens 除以
服务端 decode 时间；客户端耗时包括 prefill/传输但不含模型加载或冷却。
TCC 三次模型启动分别耗时 131.19、22.77、30.76 秒，第一项为重启后首次加载；
没有控制磁盘缓存状态，不能据此归因驱动模式对加载速度的影响。

这是当前产品自动路由的端到端模式对照：WDDM 使用小消息 mapped-host 优化，
TCC 使用现有非 P2P DMA 中转路径，不是固定通信实现后的纯驱动调度微基准。
TCC 没有启用 P2P；提升不能归因于 P2P，也未进一步拆分调度和传输贡献。
每个结果只有一次完整请求，未建立跨次波动范围。高度可预测的短输入数数结果
不能证明一般问答、85K 占用或相对 LM Studio 的收益；此前 MTP/DFlash 的
1/64 严格行为门失败仍有效，应用默认仍为普通解码。
本地结果为 `build-tools/count300-tcc/`、`count300-tcc-mtp/`、
`count300-tcc-dflash7/`，对照摘要为 `build-tools/driver-mode/tcc-comparison.json`。

### llama.cpp 对照边界

[llama.cpp 参数定义](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)
默认 split-mode 为 layer，按层拆分模型；它与 NInfer 每层做双卡张量并行的通信模式
不同，在跨卡延迟较高的平台上有优势的可能。用户 LM Studio 的实际拆分方式尚未确认。
当前没有相同模型、上下文占用和采样下的本机 llama.cpp 对照，不能宣称已超过它。
