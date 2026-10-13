# NInfer Windows 双 V100

[English](README.en.md) · [下载 Windows 应用](https://github.com/Havensky-stack/windows-ninfer-V100X2/releases) · [使用与构建详情](docs/windows-toolchain.md)

在 **Windows 原生环境**中，用两张 V100 运行 Qwen3.8-27B。保留 GRID/WDDM 和现有游戏配置，无需切换 TCC，也无需替换系统 CUDA。

本项目是在上游推理引擎基础上的 **Windows 双卡适配与易用性发行版**，不是重新实现的模型或独立训练成果。当前发布为预览版，已验证 Windows 11 + 双 V100-SXM2 16GB + GRID 553.62。

## 本项目做了什么

- **原生 Windows CLI/API**：适配文件映射、并发偏移读取、RAM-KV 虚拟内存、网络、控制台和日志，保留上游 Engine/TP2/算子路线。
- **WDDM 小消息通信优化**：无 P2P 的双 V100 使用映射主机内存传输；“从1数到300”的普通解码从 16.49 提高到 28.08 tok/s，答案逐字节一致。条件和限制见[本机性能报告](docs/windows-performance.md)。
- **保留现有驱动与环境**：项目内独立 CUDA 12.8.1 + MSVC 14.44 工具链，不替换系统 CUDA 12.4；修复中文 MSVC 头文件依赖检测。
- **接入指定 Coder390 v3 模型**：验证原始权重布局、DFlash 上下文副本和独立聊天模板；不重新量化模型。八组模板输出与原始 Jinja 模板逐字节对照。
- **本机验证与公开限制**：双卡无 P2P/NVLink 的 Graph 探针、九项平台回归、真实 CLI/API 问答、上下文容量检查；明确保留 MTP 尚未通过的严格校验结果。
- **中文便携应用**：显卡检查、模型选择/断点下载/SHA-256 校验、一键启动/停止、单轮聊天和 API 地址复制；运行依赖随包提供。

## 上游与工作来源

| 来源 | 本项目继承或参考的工作 |
|---|---|
| [tuxKOH/ninfer-V100X2](https://github.com/tuxKOH/ninfer-V100X2) | **直接上游**：双 V100 执行路线、CUDA 算子、模型运行时、格式及工具。主要推理能力来自这里。 |
| [Neroued/ninfer](https://github.com/Neroued/ninfer) | 原始 NInfer C++/CUDA 推理引擎。 |
| [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) | 上游继承的双卡 TP2、跨设备 Graph 和 YaRN 工作。 |
| [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) | 上游整合的 Volta/V100 实现。 |
| [ww485000/ninfer-windows-v100 · v100-sm70](https://github.com/ww485000/ninfer-windows-v100/tree/v100-sm70) | Windows 平台适配参考，尤其文件映射及偏移 I/O；其单卡 TCC 结果不是本项目的双卡 WDDM 成绩。 |
| [Merkyor 的 NVFP4-NInfer 模型仓库](https://modelscope.cn/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer) | 本次验证的 Coder390 模型与容器来源；本项目不宣称其训练或量化成果。 |

感谢这些项目的工作。代码遵循 [Apache-2.0](LICENSE)，来源说明见 [NOTICE](NOTICE)。第三方组件及模型分别遵循其许可证。

## 小白使用：下载、解压、双击

1. 安装适合自己显卡的 NVIDIA 驱动。**本机验证的是 GRID 553.62 + WDDM**，不是要求所有机器盲目更换到该版本。
2. 从 [Releases](https://github.com/Havensky-stack/windows-ninfer-V100X2/releases) 下载 `NInfer-Windows-V100X2-0.1.0.zip`，完整解压，双击 **`NInfer.exe`**。不要在压缩包内直接运行。
3. 选择已有的兼容 `.ninfer` 模型；没有模型时点击 **“下载推荐模型”**。模型约 **23.8GB**，需预留至少 25GB 磁盘空间，支持取消后续传及完整性校验。
4. 选择上下文容量，点击 **“启动模型”**，等待服务就绪。建议从 **128K** 开始；显存紧张时选 16K。
5. 在界面提问，点击 **“发送单轮提问”**。每次问答独立；多轮聊天可使用自己的 OpenAI API 客户端。
6. 点击 **“停止 / 取消”**或正常关闭窗口，会停止本应用启动的服务并释放显存。

**无需安装 Python、CUDA Toolkit 或 Visual Studio。** CUDA runtime 与 VC++ 运行库随发布包提供，界面使用 Windows 自带的 .NET Framework 4.8。应用不安装驱动、不修改 TCC/WDDM、全局环境变量或其他服务。

![中文启动器](docs/images/windows-launcher.png)

运行范围：Windows 11 x64，NVIDIA 设备 `0,1` 为两张至少 16GB 的 V100。RX550 可继续负责显示输出。建议 64GB 以上系统内存；本机验证为 128GB，较小内存尚未验证。发布包未做代码签名，请核对同一 Release 的 `SHA256SUMS.txt`。

### 连接聊天客户端

启动成功后点击 **“复制 API 地址”**，在客户端选择 OpenAI 兼容接口：

- Base URL：启动器显示的 `http://127.0.0.1:端口/v1`。
- 模型名：`coder390`。
- API key：留空；客户端强制填写时可填 `local`。

服务仅监听本机。端口由启动器自动选择，避免占用已有服务的端口。

### 推荐模型

启动器下载的明确文件是：

`NVFP4-NInfer/W4A4/Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer`

来源为上表 Merkyor 仓库，SHA-256：

```text
3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808
```

仅支持本项目登记的模型。**不能直接加载普通 GGUF、safetensors 或任意改成 `.ninfer` 后缀的文件。** 不要选该仓库的 `NInferV2-NOT-FOR-V3` 旧包。

## 本机实测与限制

当前源码和本地 0.1.1 构建包含上述通信优化；已发布的 0.1.0 ZIP 不包含该改动。

Windows 11、2× V100-SXM2 16GB、GRID 553.62/WDDM、PCIe 3.0 ×16，**无 P2P、无 NVLink**。普通解码、INT8 KV、256-token 预填充分块、单请求、关闭 Vision：

| 上下文容量 | 验证结果 |
|---|---|
| 131,072 | API 问答通过，启动后每卡约剩 1.80 GiB。 |
| 200,000 | CLI 问答通过，每卡约剩 420 MiB，余量较小。 |
| 220,000 | 当前显存预算不足，启动前被拒绝。 |

这是**容量分配加短问答**，不是填满长上下文后的性能或质量证明。游戏及其他 GPU 程序会减少可用显存；容量包含输入历史与输出。

**启动器默认关闭 MTP。** MTP 的 Graph/eager 对照通过，但严格 teacher-forced argmax 校验在 64 个位置中有 1 个不同；普通解码对照为 0/64。原失败判据没有放宽。DFlash、Vision、长期满容量运行尚未在此 Windows 预览版完成验收。

上游 Linux 的 QUASAR/MTP/DFlash 速度与其他 GPU 成绩属于[上游项目](https://github.com/tuxKOH/ninfer-V100X2)，**不作为本项目 Windows 性能宣传**。详细验证见 [Windows 指南](docs/windows-toolchain.md)。

## 已有源码和构建产物时如何使用

在仓库根目录的 PowerShell 执行：

```powershell
.\tools\windows\serve.ps1 -Artifact '.\models\Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer' -Context 131072
```

默认监听 `http://127.0.0.1:8080/v1`，用 `-Port` 更换端口。模型 ID 可通过 `/v1/models` 查询。需要自己编译时阅读 [工具链与构建](docs/windows-toolchain.md)；构建便携应用和发布包见 [应用维护说明](docs/windows-app.md)。

日志及上次选择的模型路径保存在 `%LOCALAPPDATA%\NInferV100X2`。卸载时关闭启动器、删除解压目录即可；模型和设置可自行保留。
