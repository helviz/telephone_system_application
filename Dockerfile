# Use the official PyTorch development image with built-in CUDA 12.1 and build tools
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Prevent interactive prompts and configure model caches
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache \
    CMAKE_ARGS="-DLLAMA_CUDA=on" \
    FORCE_CMAKE=1

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

# Compile llama-cpp-python natively from source using the container's NVCC compiler
RUN pip install --no-cache-dir llama-cpp-python==0.3.22

# Install heavy inference model blocks
RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    ctranslate2==4.7.1 \
    TTS==0.22.0 \
    librosa \
    soundfile \
    scipy \
    numpy \
    transformers==4.41.0 \
    tokenizers \
    safetensors

# Copy requirements.txt and remove the CPU wheel reference to prevent package collisions
COPY requirements.txt .
RUN sed -i '/llama_cpp_python/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the remaining codebase files
COPY . .

# Set up runtime folders and broad write access privileges for Hugging Face Spaces
RUN mkdir -p /data /app/.cache && \
    chmod 777 /data /app/.cache && \
    chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]