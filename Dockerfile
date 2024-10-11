# Use a base image with CUDA 12.x to match the host's CUDA version
FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV CUDA_VERSION=12.2
ENV CUDNN_VERSION=8

WORKDIR /

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo ca-certificates git wget curl bash libgl1 libx11-6 \
    software-properties-common ffmpeg build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies except ctranslate2
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt \
    && rm /requirements.txt

# Build and install ctranslate2 from source
RUN git clone https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 \
    && cd /tmp/ctranslate2 \
    && mkdir build && cd build \
    && cmake -DWITH_CUDA=ON -DCUDA_ARCH_LIST="90" .. \
    && make -j$(nproc) install \
    && cd / && rm -rf /tmp/ctranslate2

# Copy and run script to fetch models
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

# Copy source code into image
COPY src .

# Set default command
CMD ["python", "-u", "/rp_handler.py"]
