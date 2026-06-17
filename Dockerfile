# Maintain exact Python 3.10 environment
FROM python:3.10-slim

WORKDIR /app

# Install lightweight system requirements (no heavy build-essential or cmake)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Core ML Stack
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

# Install App packages and the pre-compiled llama-cpp wheel
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