# ── Use Nvidia CUDA Base Image ───────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# ── Install Python 3.10 & System dependencies ────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default 'python' and 'pip'
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# ── Upgrade pip ─────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch CUDA 12.1 (Replaces the old +cpu wheels) ─
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Core ML stack ───────────────────────────────────
RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    ctranslate2==4.7.1 \
    TTS==0.22.0 \
    librosa \
    soundfile \
    scipy \
    numpy

# ── LLM / Transformers ──────────────────────────────
RUN pip install --no-cache-dir \
    transformers==4.41.0 \
    tokenizers \
    safetensors

# ── Compile llama-cpp-python with CUDA Acceleration ──
ENV FORCE_CMAKE=1
ENV LLAMA_CUDA=on
RUN pip install --force-reinstall --no-cache-dir llama-cpp-python

# ── App requirements ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app ────────────────────────────────────────
COPY . .

# ── Persistent Storage Directory Configuration ──────
RUN mkdir -p /data && chmod 777 /data
RUN chmod +x /app/start.sh

# ── Environment Configuration ───────────────────────
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

EXPOSE 7860

CMD ["/app/start.sh"]