FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg for audio decoding, libsndfile for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (keeps image size sane — ~800 MB vs 5 GB for CUDA)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    faster-whisper==1.0.3 \
    silero-vad \
    soundfile \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.30.1" \
    structlog==24.1.0 \
    "redis[asyncio]>=5.0.0" \
    websockets>=12.0 \
    audioop-lts; python_version>="3.13"

# audioop is built-in for Python ≤ 3.12 — only install the backport on 3.13+.
# The conditional above handles it; no extra step needed.

COPY main.py .

# Pre-download Silero VAD weights so the first request isn't slow
RUN python - <<'EOF'
import torch
torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False)
EOF

EXPOSE 8001
# Use --ws websockets to ensure WebSocket support is available
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--ws", "websockets"]
