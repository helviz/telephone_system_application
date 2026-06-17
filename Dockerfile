# ── Maintain exact Python 3.10 environment ──────────
FROM python:3.10-slim

WORKDIR /app

# ── Install CUDA runtime repositories & dependencies ──
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip and core compiler components ──────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel scikit-build-core

# ── PyTorch with CUDA 12.1 acceleration ──────────────
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Core ML Stack ───────────────────────────────────
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

# ── Install App packages from requirements.txt ───────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Compile llama-cpp-python with CUDA Support ───────
ENV FORCE_CMAKE=1
ENV LLAMA_CUDA=on
RUN pip install --force-reinstall --no-cache-dir llama-cpp-python==0.3.22

# ── Copy Application Files ──────────────────────────
COPY . .

# ── Persistent Storage Directory Configuration ──────
RUN mkdir -p /data && chmod 777 /data
RUN chmod +x /app/start.sh

# ── Environment Configuration ───────────────────────
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

EXPOSE 7860

CMD ["/app/start.sh"]