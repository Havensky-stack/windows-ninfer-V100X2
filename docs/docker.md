# Docker

容器默认下载 [QUASAR NVFP4 v3](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer/tree/v3)
的 `qwen3_8_27b_nvfp4.ninfer`，完成后打开 TUI。按 `s` 加载模型并启动 API，
再按 `s` 停止；`q` 停止服务并退出。首次下载需要足够的磁盘空间。

## 启动

宿主机需要 Linux、可用的 NVIDIA 驱动、Docker Compose v2 和
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
镜像内置 CUDA 12.8，默认编译 SM70，面向两张 V100 16 GB；宿主机无需安装 CUDA 编译器。

```bash
docker compose build
docker compose run --rm --service-ports ninfer
```

请在交互终端使用 `compose run`。TUI 默认 QUASAR / TP2 / MTP4 / 180000 上下文，
Vision 和两个数值实验关闭。API 在容器内监听 `0.0.0.0:8080`，映射到宿主机 8080。
按 `s` 前可以编辑设备、容量、视觉等选项；[完整操作](tui.md)。

无需 Compose 的等价命令：

```bash
docker build -t ninfer:v100 .
docker run --rm -it --gpus all --ipc=host -p 8080:8080 \
  -v ninfer-models:/Models/ninfer-V100X2 \
  -v ninfer-data:/data ninfer:v100
```

构建并行度默认 2，可用 `NINFER_BUILD_JOBS=2 docker compose build` 调低；
直接构建时用 `--build-arg BUILD_JOBS=2`。

## 换模型

将 Hugging Face 仓库链接或文件链接传给 `--model`：

```bash
docker compose run --rm --service-ports ninfer \
  --model https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer \
  --profile quasar-mtp4
```

也接受 `owner/repository`、`/tree/版本`、`/resolve/版本/文件.ninfer` 和
`/blob/版本/文件.ninfer`。仓库内只有一个 `.ninfer` 时自动选择，否则用
`--filename 子目录/文件.ninfer` 指定；`--revision` 可覆盖链接中的版本。
自定义仓库未指定版本时使用 `main`，默认 QUASAR 链接使用 `v3`。

也可以通过环境变量设置：

```bash
NINFER_MODEL=https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer \
  docker compose run --rm --service-ports ninfer
```

私有或受限仓库可设置 `HF_TOKEN`，Compose 会传入容器；直接 `docker run` 时加
`-e HF_TOKEN`。普通 HTTP(S) `.ninfer` 文件链接也可作为 `--model`。
使用已有本地文件时，需要显式挂载：

```bash
docker run --rm -it --gpus all --ipc=host -p 8080:8080 \
  -v /Models/ninfer-V100X2:/Models/ninfer-V100X2 \
  -v ninfer-data:/data ninfer:v100 \
  --model /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer --profile q4-mtp3
```

下载入口只接受引擎已支持身份的 `.ninfer` 成品；原始 safetensors/GGUF 需要先转换。
换链接不会增加引擎支持的模型架构。入口会将下载路径传给 TUI，推理参数由保存的配置
或显式 `--profile` 决定；换模型时请核对 TP、MTP 和容量。最终支持范围由引擎检查。

## 缓存和配置

Compose 的两个命名卷保留下载、配置和日志，退出或重建容器不会删除：

| 容器路径 | 内容 |
|---|---|
| `/Models/ninfer-V100X2` | 模型和 Hugging Face 下载元数据 |
| `/data/config/ninfer/tui.json` | TUI 保存的非敏感设置 |
| `/data/logs/requests.jsonl` | 请求与吞吐日志 |

HF 下载支持断点续传，并检查对应版本的文件更新；完整文件会复用。
普通 HTTP 链接按完整 URL 分开缓存，支持服务端提供的断点续传，完成后直接复用，
同一 URL 的内容变化需删除对应缓存后重新下载。
`docker compose down -v` 会删除这些命名卷和其中的文件。

首次无终端下载可用：

```bash
docker compose run --rm -T ninfer --pull-only
```

其余参数转交 TUI，包括 `--profile`、`--no-settings`、`--settings`、`--host`、
`--log` 和 `--print-command`。例如先下载并查看实际服务命令：

```bash
docker compose run --rm -T ninfer --profile quasar-mtp4 --print-command
```

容器入口每次将模型、程序、监听地址和日志路径设为上述容器默认值；
显式传入的同名 TUI 参数优先，其余已保存设置保留。
若修改 TUI 中的容器监听端口，还需要相应修改 Docker 端口映射；
只改宿主机端口可用 `NINFER_PORT=8081 docker compose run --rm --service-ports ninfer`。
