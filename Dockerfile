# GPU-enabled Dockerfile for the CTZ benchmarking workflow.
# Requires host NVIDIA drivers and NVIDIA Container Toolkit.
# Build: docker build -t ctz_research:gpu .
# Run:   docker run --gpus all -it --rm -v "$PWD":/app ctz_research:gpu

FROM nvidia/cuda:12.2.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CTZ_REQUIRE_GPU=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 \
       python3.11-distutils \
       python3-pip \
       build-essential \
       git \
       libhdf5-dev \
       libnetcdf-dev \
       libopenblas-dev \
       liblapack-dev \
       gcc \
       wget \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt
RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

COPY . /app

CMD ["python3", "benchmark_pipeline.py"]
