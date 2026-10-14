# Windows 双 V100 工具链

普通用户优先使用[中文首页的便携应用](../README.md#小白使用下载解压双击)，无需安装编译环境。
启动器与发行打包说明见 [Windows 应用维护](windows-app.md)。本文面向自行编译和复查验证的用户。

本机测速与跨卡延迟归因见 [Windows 性能诊断](windows-performance.md)。

本 fork 的 Windows 工作以 `tuxKOH/ninfer-V100X2` 的双卡执行路径为基础，参考
[ww485000 的 Windows 单卡分支](https://github.com/ww485000/ninfer-windows-v100/tree/v100-sm70)
的平台适配。Windows CLI/API 已完成整个引擎的编译、链接和 `--help` 启动检查。
模型与回归验证范围见本文末尾；单卡分支的 TCC/32GB 性能结果不代表本机结果。

## 已验证组合

2026-10-14，本机 Windows 11、两张 V100-SXM2 16GB、两卡 PCIe 3.0 x16：

| 组件 | 选择 |
|---|---|
| NVIDIA 驱动 | GRID 553.62；初始验证为 WDDM，现已切换双卡 TCC，模式对照见性能诊断 |
| CUDA | 项目内的 CUDA 12.8.1 组件，NVCC 12.8.93、Runtime 12.8.90 |
| CPU 编译器 | MSVC 工具集 14.44.35207，编译器 19.44.35221 |
| 编译器安装位置 | 现有 Visual Studio 2026 内的 14.44 工具集；不使用默认的 14.50 |
| Windows SDK | 10.0.26100.0 |
| CMake / 生成器 | VS 自带 CMake 4.1.1-msvc1 / Ninja 1.12.1 |
| CUDA 编译目标 | 原生 `sm_70` 机器码，探针不依赖新 PTX 在旧驱动上即时编译 |
| 依赖管理 | VS 自带 vcpkg；固定 FFmpeg 8.1.2、curl 8.21.0、zlib 1.3.2 |

CUDA 12.8.1 的主机编译器检查接受 MSVC 19.44；无需
`--allow-unsupported-compiler` 或修改 CUDA 头文件。使用 Ninja，避免单卡参考项目
固定的 `Visual Studio 17 2022` 生成器与本机安装不匹配。

GRID 553.62 满足 CUDA 12.x 小版本兼容的最低 Windows 驱动门槛，但不等于支持
12.8 的所有新驱动功能。旧驱动对新 PTX 和新驱动 API 的限制仍需在实际引擎中验证。
依据：[CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)、
[小版本兼容限制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 准备与验证

在仓库根目录的 PowerShell 中执行：

```powershell
# 仅首次准备或恢复本地组件时需要；下载并校验 NVIDIA 官方组件。
.\tools\windows\setup-cuda.ps1

# 初始化进程内工具链、用 CMake/Ninja 编译并运行仓库现有 TP2 探针。
.\tools\windows\check-toolchain.ps1

# 后续在当前终端手动构建时，单独载入环境：
. .\tools\windows\enter-toolchain.ps1
```

## 编译产品和测试

```powershell
# 安装固定版本的 vcpkg 依赖并构建 CLI/API；默认最多四个并行任务。
.\tools\windows\build.ps1

# 启用测试时明确指定已有的 Python 3.11，不修改该 Python 环境。
.\tools\windows\build.ps1 -Tests -PythonExecutable 'C:\path\to\python311\python.exe'
.\tools\windows\test.ps1
```

依赖通过仓库 `vcpkg.json` 和 `x64-windows-ninfer` triplet 固定，使用动态 CRT、动态库、
Release 构建及 MSVC 14.44。输出在 `build-windows/apps`。vcpkg 的 app-local 部署会把
运行所需的依赖 DLL 放到程序目录；不要只拷贝 EXE。

持有兼容的本地模型后，可启动仅监听本机的文本服务：

```powershell
.\tools\windows\serve.ps1 -Artifact 'D:\Models\qwen3_8_27b_nvfp4.ninfer'
```

启动脚本使用双卡 `0,1`、INT8 KV、普通解码、单并发、16K 上下文、256-token 预填充分块。
用 `-Context` 和 `-PrefillChunk` 显式调整容量和工作区。先释放其他模型占用的
显存。该脚本不下载、转换模型，不停止其他服务，也不修改显卡模式。
本次模型的实际验证和容量限制见下文；其他模型须另外验证。

## 本次选择的 Coder390 模型

使用 [Merkyor 的 NVFP4-NInfer 仓库](https://modelscope.cn/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer)
中的明确文件：

`NVFP4-NInfer/W4A4/Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer`

这是 v3 容器，文件大小 23,773,675,264 字节，发布者的 SHA-256 为
`3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808`。
其名称虽含 W4A4，实际存储仍采用本引擎的混合 NVFP4/FP8/W8 配置；V100 使用已有的
解量化计算路径，不具备原生 NVFP4 Tensor Core 指令。模型通过明确的 metadata 名称
登记，并继续接受完整的张量布局验证；不改写下载的 packed 权重或前端资源。
独立聊天模板按原始资源读取；DFlash 的十组独立上下文 K/V 副本必须与 QKV 对应行的
量化码、FP16 尺度逐字节相同，验证后才复用已有的 QKV 行。

不要用仓库中名字含 `NInferV2-NOT-FOR-V3` 的文件替代它：这些旧 V100 包的前端和
张量清单与本 fork 不匹配。发布者的其他 GPU 速度和质量评分不代表本机验证结果。

本地文件放在忽略提交的 `models/` 目录，用 `serve.ps1 -Artifact` 显式指定。
INT8 KV + MTP 的 KV 预算约为每个 token 35,904 字节（两卡合计），128K 容量约
4.38 GiB，180,000 容量约 6.02 GiB，另有权重、双卡复制、工作区和 Graph 开销。
容量必须根据实际加载后的显存预算确认，不能从文件大小直接推导最大上下文。

### 上下文容量实测

建议日常使用普通解码、131,072 上下文、INT8 KV、256-token 预填充分块。已通过真实
OpenAI Chat Completions 请求，每卡启动后约有 1.80 GiB 余量：

```powershell
.\tools\windows\serve.ps1 -Artifact '.\models\Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer' -Context 131072
```

普通解码容量测试（其余设置相同）：

| 容量 | 结果 |
|---|---|
| 131,072 | API 问答通过；每卡约余 1.80 GiB |
| 200,000 | CLI 问答通过；每卡约余 420 MiB，运行区约 4.24 GiB |
| 220,000 | 显存预算预检查拒绝；需 4,981,152,512 字节，可用 4,979,431,936 字节/卡 |

因此普通解码已验证到 20 万容量，当前内存预算的边缘接近但不到 22 万；不建议贴着
该边缘运行。以上都是短请求容量验证，并非满容量长期运行测试。

上述模型在 TP2、INT8 KV、MTP3、优化 proposal head、CUDA Graph、单请求、关闭 Vision
的配置下，每卡权重驻留约 10.46 GiB。GRID/WDDM 报告的可分配预算不能简单按 16 GiB
减权重计算；本次权重之外的运行区预算约 4.26 GiB/卡。

| 容量 | 预填充分块 | 实测结果 |
|---|---|---|
| 4,096 | 256 | 加载并完成中文问答 |
| 180,000 | 256 | 加载并完成中文问答；启动后每卡仅余约 242/246 MiB |
| 180,000 | 1,024 | 显存预算预检查拒绝：运行区约需 4.48 GiB/卡 |
| 262,144 | 256 | 显存预算预检查拒绝：运行区约需 5.83 GiB/卡 |

180,000 是 MTP 配置已验证的容量，显存余量很紧，不是承诺的稳定极限。上述容量问答只有
28 个输入 token、7 个输出 token，不能证明填满长上下文后的吞吐量、质量或稳定性。
模型注册的原生位置上限为 262,144；它不等于本机能够容纳的 KV 容量。
游戏或其他 GPU 程序会改变可用预算，应进一步降低容量。

CUDA 组件位于 `build-tools/cuda-12.8.1`；下载缓存和探针构建目录已经被现有
`build-*/` 忽略规则覆盖。安装内容包括 NVCC、CUDART、CCCL、NVML 开发文件和 NVTX，
不是包含所有数学库及分析器的完整系统 Toolkit 安装。

工具链安装/构建脚本不安装或替换显示驱动，不更改显卡模式，不改注册表或用户/系统级环境变量。
系统 CUDA 12.4 保留；载入脚本仅改变当前 PowerShell 及子进程的环境。
不要在这个构建终端启动需要默认 CUDA 12.4 的其他开发任务。

单独的显卡模式工具位于 `tools/windows/enable-tcc.cmd`、`restore-wddm.cmd` 和
`set-v100-mode.ps1`，会请求管理员权限并修改两张 V100 的待生效模式；操作步骤、
重启要求和只读状态查询见 [README](../README.md#显卡模式推理用-tcc游戏用-wddm)。

## 验证结果及边界

`tools/tp2/capture_probe.cu` 经上述 C++20 工具链实际编译并运行成功：

- CUDA 查询双向 P2P 均为 0，使用驱动处理的非 P2P 跨卡拷贝。
- 两张卡的计算内核、跨卡事件及双向拷贝执行成功。
- 单个跨设备 Graph 包含 12 个计算节点和 6 个拷贝节点。
- Graph 实例化和重放通过；两个设备的完整输出与普通执行一致，两次重放一致。
- 探针的独立双 Graph 合并负对照按预期返回 `cudaErrorStreamCaptureMerge`，不属于失败。

产品 `ninfer.exe` 与 `ninfer-serve.exe` 已编译、链接，并通过 `--help` 启动检查。
平台适配覆盖 Windows 文件映射/并发偏移读取、RAM-KV 虚拟内存、Winsock、控制台与
日志，以及 MSVC 的 NVTX/utf8proc/移动构造和 Ninja 中文头文件依赖检测。
`tools/windows/test.ps1` 的九项定向回归全部通过，覆盖容器读取/物化、媒体解码、
OpenAI/Anthropic schema、服务选项/日志、双卡通信和 RAM-KV；另有模板/资源检查。

首轮双卡通信测试的数值和 guard 检查通过；10 KiB BF16 all-reduce 在每次都做主机同步的
500 次微基准中平均约 1.43 ms。继承的 100 us 计算驱动预算未达到。测试在 WDDM 下
保留延迟报告和全部正确性断言，将延迟单独列为诊断数据；Linux/TCC 仍保留原性能门槛。
这一微基准不代表 CUDA Graph 下的整模型 token 吞吐量。

Coder390 真实容器的 DFlash2 v3 load-plan 检查通过，包括 66 个投影的原始 payload 和
启用/禁用物化计划；这不是 DFlash 生成验证。独立聊天模板的八组 Jinja2 参考输出和
资源一致性检查通过。完整上游 frontend 测试依赖作者私有 Linux tokenizer 路径，未能
在本机完整运行；Windows 脚本仅运行可独立复现的模板/资源检查。

### MTP 尚未通过严格行为门

真实 Coder390、4K 容量、256 分块、MTP3/优化 proposal head：32 和 725 token 两个
提示各生成 32 token，Graph/eager 的输出和接受状态一致，采样/贪心切换检查通过。
但非投机逐位置 teacher-forced argmax 检查在 64 个位置中有 1 个不同，输出 logit
差为 0.125；普通解码对照为 0/64。原测试因此返回失败，未放宽判据。
这个结果不能认定 MTP 满足本项目的严格行为契约，也不证明源模型微调/量化质量。
因此 `serve.ps1` 默认关闭投机解码。上面的 MTP 容量数据只说明内存和短请求能够执行，
不构成对 MTP 正确性的认证；DFlash、Vision 和长期满容量请求也未做本机验收。

0.1.1 mapped-host 通信优化后的复核结果相同：Graph/eager 与采样路线检查通过，
普通对照 0/64；MTP 仍在 probe 1、位置 31 发生同一处 0.125-logit 差异。
4K MTP 的 Graph 预算已从 24 MiB 修正为 32 MiB，以容纳新增的传输内核节点。

复现实际模型检查（模型路径必须显式指定）：

```powershell
$env:NINFER_V100X2_ARTIFACT = (Resolve-Path 'models/Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer').Path
$env:NINFER_V100X2_PROPOSAL_HEAD = 'optimized'
.\build-windows\tests\ninfer_qwen3_8_27b_v100x2_real_test.exe
```

## 上游 Volta 更新的 Windows 集成验证

本次集成上游 `861633e8` / `756d1363`，保留 WDDM mapped-host 通信、Coder390 前端、
4K MTP Graph 的 32 MiB 预算、模式切换工具及中文启动器。新 API 中的 `device_count`
表示执行域设备数；本地通信优化已同步适配。系统 CUDA 12.4 和 GRID 553.62 未改变。

验证使用独立目录 `build-windows-upstream`，CUDA 12.8.93、MSVC 14.44、Release、SM70、
四路构建；共享原有 vcpkg/CUTLASS 依赖。CLI/API 构建和 `--help` 通过；12 项平台/API
测试通过，包括 public API、device、artifact reader/materialization、media decode、
allreduce、RAM-KV、CLI/serve options、request log 和 OpenAI/Anthropic schema。
Coder390 模板对照及启动器参数、持久化、下载测试通过。

新 W8 词表头检查覆盖 code/scale 的精确排列、独立 FP64 投影参考、Graph/eager 与内存
边界；NVFP4 输入投影、down projection 和 SwiGLU 对照独立 FP64 参考通过。
85K 实际占用、180K 容量、1/4/5/6 查询的 INT8 注意力完整 FP64 对照、Graph 重放与
页表边界检查也通过。这是算子验证，不是整模型长上下文性能或质量证明。
这些测试可在通常的 `build-windows` 目录复现：

```powershell
. ./tools/windows/enter-toolchain.ps1
cmake --build build-windows --parallel 4 --target ninfer_linear_w8_prepacked_sm70_test ninfer_nvfp4_input_sm70_test ninfer_attention_headlocal_test
ctest --test-dir build-windows --output-on-failure -j 1 -R '^ninfer_(linear_w8_prepacked_sm70|nvfp4_input_sm70)_test$'
./build-windows/tests/ninfer_attention_headlocal_test.exe --long-context-only
```

PP2、实验长注意力和 Q/K 均值 RAM-KV 保留上游的默认关闭状态，未加入 Windows GUI。
本轮测试在 TCC 下执行；不把它当作新版 WDDM 或 PP2 的整模型性能认证。

本次 TCC Coder390、4K 容量、DFlash7/full 的真实模型复核：Graph/eager 与
贪心/采样切换检查通过；严格 teacher-forced 门仍失败，同一 probe 1、位置 31，
目标 token 2532、投机 token 561，logit 差 0.125（1/64）。普通对照 0/64。
未放宽断言，也未认定 DFlash 完成严格资格验证。MTP 本轮未重跑整模型门，旧失败仍有效。

```powershell
$env:NINFER_V100X2_ARTIFACT = (Resolve-Path 'models/Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer').Path
$env:NINFER_V100X2_SPEC = 'dflash'
$env:NINFER_V100X2_PROPOSAL_HEAD = 'full'
./build-windows/tests/ninfer_qwen3_8_27b_v100x2_real_test.exe
```
