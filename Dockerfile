FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface \
    HF_HUB_CACHE=/data/.cache/huggingface/hub \
    CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Build from source against actual CUDA 12.8 — no prebuilt wheel exists for cu128
RUN pip install --no-cache-dir llama-cpp-python==0.3.22 --no-binary llama-cpp-python

RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    ctranslate2==4.7.1 \
    librosa \
    soundfile \
    numpy \
    scipy==1.13.1 \
    transformers==4.44.2 \
    tokenizers \
    safetensors

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/.cache/huggingface /app/.cache && \
    chmod -R 777 /data /app/.cache && \
    chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]