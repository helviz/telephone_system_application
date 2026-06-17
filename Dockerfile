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

# ==============================================================================
# CRITICAL LINKER FIX: Replace Conda's outdated libstdc++.so.6 with the system's
# updated version containing GLIBCXX_3.4.30 so the pre-built CUDA wheels can link.
# ==============================================================================
RUN rm -f /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# Upgrade foundational packaging infrastructure
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install llama-cpp-python using pre-built CUDA 12.1 wheels to avoid build timeouts/OOMs
# NOTE: requirements.txt must NOT list llama_cpp_python — that CPU wheel would
# silently overwrite this CUDA build when requirements.txt installs below.
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
# (kept as its own layer so Docker can cache it independently of app code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check at build time: fail fast if llama-cpp-python ends up CPU-only
# or if the CUDA backend isn't actually compiled in.
RUN python -c "from llama_cpp import llama_cpp; print('GPU offload available:', llama_cpp.llama_supports_gpu_offload())"

# Copy the remaining codebase files
COPY . .

# Set up runtime folders and broad write access privileges for Hugging Face Spaces
RUN mkdir -p /data /app/.cache && \
    chmod 777 /data /app/.cache && \
    chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]