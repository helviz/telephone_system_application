# Use the official PyTorch development image with built-in CUDA 12.1 and build tools
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Prevent interactive prompts and configure model caches
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache

WORKDIR /app

# Install system dependencies required for telephony and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade foundational packaging infrastructure
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install llama-cpp-python using pre-built CUDA 12.1 wheels to avoid build timeouts/OOMs
RUN pip install llama-cpp-python==0.3.22 --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Install heavy inference model blocks (Excluding the broken/unused Coqui TTS package)
RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    ctranslate2==4.7.1 \
    librosa \
    soundfile \
    scipy \
    numpy \
    transformers==4.41.0 \
    tokenizers \
    safetensors

# Copy requirements.txt and install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining codebase files
COPY . .

# Set up runtime folders and broad write access privileges for Hugging Face Spaces
RUN mkdir -p /data /app/.cache && \
    chmod 777 /data /app/.cache && \
    chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]