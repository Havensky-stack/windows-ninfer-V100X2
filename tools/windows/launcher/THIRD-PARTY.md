# Binary distribution components

NInfer and this launcher use the repository's Apache-2.0 LICENSE and NOTICE.
The engine incorporates the in-tree utf8proc, nlohmann/json, cpp-httplib, CUTLASS and NVTX
components; their license files are included by the release packaging workflow.

FFmpeg 8.1.2 is dynamically linked, built without GPL/nonfree features by vcpkg.
The release includes a separate FFmpeg source archive, the exact vcpkg port/patches and build
instructions. Users may replace the FFmpeg DLLs with a compatible modified build; reverse
engineering for debugging modifications to the LGPL library is permitted by its license.
See ffmpeg.txt and ffmpeg.spdx.json for license and source identities.

curl 8.21.0 and zlib 1.3.2 are dynamically linked; copyright/license texts and source identities
are provided alongside this file. OpenBLAS 0.3.33 is dynamically linked for CPU Vision, built
without its threading feature; its BSD license and source identity are in openblas.txt and
openblas.spdx.json. The SM70 attention adaptation's license and author list are also included.
CUDA CUDART 12.8.90 is statically linked into the engine;
its redistributable terms are in NVIDIA-CUDA.txt. No NVIDIA driver is distributed.

Microsoft Visual C++ 14.44.35112 x64 runtime DLLs are included app-locally under Microsoft's
redistributable terms: https://visualstudio.microsoft.com/license-terms/
Windows and .NET Framework remain operating-system prerequisites.

Model weights are not included in this ZIP. The optional downloader fetches the published
Merkyor model directly from ModelScope. Its model card and terms remain at:
https://modelscope.cn/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer
