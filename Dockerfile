# Use an explicit NVIDIA CUDA base runtime to leverage the T4 GPU
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies and Python 3.10
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up aliases so 'python' points to our installation
RUN ln -s /usr/bin/python3.10 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with explicit CUDA 12.1 support
RUN pip install --no-cache-dir torch==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install GPU-accelerated llama-cpp-python directly via pre-built wheels
RUN pip install --no-cache-dir llama-cpp-python==0.3.22 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Core Accelerated ML Stack
RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    ctranslate2==4.7.1 \
    TTS==0.22.0 \
    librosa \
    soundfile \
    scipy \
    numpy

# LLM / Transformers Support
RUN pip install --no-cache-dir \
    transformers==4.41.0 \
    tokenizers \
    safetensors

# Copy requirements and install remaining standard packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Files
COPY . .

# Persistent Storage Directory Configuration
RUN mkdir -p /data && chmod 777 /data
RUN chmod +x /app/start.sh

# Environment Configuration
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

EXPOSE 7860

CMD ["/app/start.sh"]