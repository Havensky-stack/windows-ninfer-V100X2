# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCH=70
ARG BUILD_JOBS=2
RUN apt-get update && apt-get install --yes --no-install-recommends \
        build-essential ca-certificates cmake git ninja-build pkg-config \
        libavcodec-dev libavformat-dev libavutil-dev libcurl4-openssl-dev libswscale-dev libopenblas-serial-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel ${BUILD_JOBS} --target ninfer ninfer-serve

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --yes --no-install-recommends \
        ca-certificates curl python3 python3-venv tini ncurses-term \
        libavcodec60 libavformat60 libavutil58 libcurl4t64 libswscale7 libopenblas0-serial \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir huggingface-hub==1.0.1

# Use the driver's libcuda supplied by the NVIDIA Container Toolkit.
RUN rm -rf /usr/local/cuda-12.8/compat /usr/local/cuda/compat

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
WORKDIR /opt/ninfer
COPY tools/v100/ninfer_tui.py tools/v100/gui_cache.py tools/v100/launcher_sampling.py ./tools/v100/
COPY docker/entrypoint.py ./docker/entrypoint.py
COPY LICENSE NOTICE ./

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    TERM=xterm-256color \
    NINFER_MODEL_DIR=/Models/ninfer-V100X2 \
    HF_HOME=/Models/ninfer-V100X2/.huggingface \
    XDG_CONFIG_HOME=/data/config

VOLUME ["/Models/ninfer-V100X2", "/data"]
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/venv/bin/python3", "/opt/ninfer/docker/entrypoint.py"]
CMD []
