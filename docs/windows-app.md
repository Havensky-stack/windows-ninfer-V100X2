# Windows 中文便携应用

用户操作见[中文首页](../README.md)。`NInfer.exe` 是一个薄 WinForms 启动器，通过本机
HTTP 调用既有 `ninfer-serve.exe`；没有第二套推理实现，也不改变模型数值路径。

## 运行和环境

- 64 位 Windows 11，系统自带 .NET Framework 4.8；设备 0、1 为至少 16GB 的 V100。
- 驱动由用户安装。检查 CUDA 12.x 最低 Windows 驱动门槛 528.33，本机实测 GRID 553.62。
  达到最低版本不等于其他驱动/硬件组合已通过本项目验证。
- ZIP 的 `bin/` 包含引擎及 FFmpeg/curl/zlib/OpenBLAS/VC++ DLL；CUDART 静态链接进引擎。
  普通用户无需安装编译器、Python、CUDA Toolkit 或系统 VC++ 安装程序。
- v0.1.2 默认普通解码、思考开启、INT8 KV、256 分块、128K 上下文、2048 输出上限。
- 可选实验 MTP（1–3 草稿、优化草稿头开关）、DFlash（1–7 草稿）；DFlash 选择时建议
  96K 容量并限制单请求，多并发已知错误未修复。MTP/DFlash 的 1/64 严格行为门失败仍有效。
- 上下文、输出上限、预填充分块、KV 格式、并发、端口、Graph 和前缀复用可调整；
  选项通过同一公共 server CLI 生效。普通/MTP 共享 KV 为每路容量乘并发，显存由引擎预算检查。
- 只监听回环地址，端口 0 自动选择，也可指定固定端口；已占用端口会报错。
  只停止启动器持有的子进程，不按进程名关闭其他服务。
- 只在用户点击下载并选定目录后下载明确登记的模型。单个 `.part` 文件保存进度，完成后
  核对精确大小和 SHA-256，才重命名为最终模型；已存在的最终模型不会被覆盖。
- 设置与日志在 `%LOCALAPPDATA%\NInferV100X2`，`settings.json` 保存模型路径和全部启动选项。
  内置聊天为单轮、非流式，与服务默认值共用输出上限，显示 `reasoning_content` 和最终回答；
  输出达到上限会提示截断。多轮及流式功能由外部 OpenAI 兼容客户端使用引擎既有接口。
- TCC/WDDM 按钮调用包内 `tools/` 的同一模式脚本，经 UAC 以管理员身份运行；
  服务运行时按钮禁用，脚本不自动重启。这里只请求模式切换，不安装驱动；回切 WDDM 尚未实测。

这是一份未签名的预览发行包，不是驱动安装器。目前未验证全新 Windows 安装或所有 GRID
版本；针对本机的具体证据和 MTP 限制见[工具链指南](windows-toolchain.md)。

## 构建与测试

先按工具链指南构建两个产品 EXE，然后：

```powershell
.\tools\windows\test-launcher.ps1
.\tools\windows\package.ps1 -SkipArchive
```

启动器使用 Windows 自带的 .NET Framework C# 编译器，不需要新装 .NET SDK/NuGet。
下载测试使用本机 HTTP 小文件，覆盖 Range 续传、服务端忽略 Range 后重启下载、错误哈希
拒绝发布、取消后续传；不会重复下载整个模型。原生 EXE 的应用清单启用 UTF-8 代码页，
以支持中文和空格模型路径；不会更改系统代码页。
选项测试覆盖实际 CLI 参数、DFlash 多并发拒绝、思考/回答显示和输出截断提示。

真实模型集成检查通过启动器的环境检测、服务启动、健康检查、Chat Completions 和停止路径：

```powershell
$app = (Resolve-Path '.\build-release\NInfer-Windows-V100X2-0.1.2\NInfer.exe').Path
$model = (Resolve-Path '.\models\Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer').Path
$log = Join-Path $PWD 'build-release\launcher-smoke.log'
$process = Start-Process $app -ArgumentList '--smoke', ('"' + $model + '"'), ('"' + $log + '"'), 'dflash' -WindowStyle Hidden -PassThru -Wait
if ($process.ExitCode -ne 0) { throw 'Launcher smoke failed; inspect the log.' }
```

`--render-ui <PNG路径>` 仅绘制启动器自己的控件以检查布局，不启动推理或截取用户桌面。
2026-10-14，v0.1.2 的真实 DFlash7 + thinking 集成检查通过，使用双卡 TCC、4K 容量、
512 输出上限，包含思考及最终回答；不是新的速度或模型质量验收。
2026-10-27，0.1.4 发布构建再次通过该真实 DFlash7 + thinking 集成检查，包含启动、
思考/回答解析和退出时停止子服务；仍不代表 DFlash 已通过严格数值/行为门。

## 发行打包

```powershell
.\tools\windows\package.ps1 -BuildDirectory build-windows-upstream -Version 0.1.4
.\tools\windows\package-ffmpeg-source.ps1

# 两个归档完成后生成本次发布的完整校验文件。
$archives = @('.\build-release\NInfer-Windows-V100X2-0.1.4.zip', '.\build-release\FFmpeg-8.1.2-source.zip')
$archives | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($_))"
} | Set-Content -LiteralPath .\build-release\SHA256SUMS.txt -Encoding ascii
```

输出到忽略提交的 `build-release/`。包含主 ZIP、FFmpeg 完整源代码归档和实际使用的
vcpkg port/补丁，以及两个 ZIP 的 SHA-256。依赖许可从所选构建的 CMakeCache 读取；
发布时把两个 ZIP 与 `SHA256SUMS.txt` 一起上传。
需要时可给源代码打包脚本显式传 `-SourceArchive` 和 `-PortDirectory`。

第三方许可在应用 `licenses/`，项目许可/来源说明在根目录。FFmpeg 动态链接，并提供
对应源码及替换 DLL 的重建说明；模型单独从发布者下载，不放进应用 ZIP 或 Git。
不要分发本机模型、日志、下载缓存、编译工具链或任何 GPU 驱动。
