FROM python:3.10

WORKDIR /app

# ── System dependencies ─────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ─────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch CPU (IMPORTANT: keep consistent with transformers) ─
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchaudio==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Core ML stack (from your conda file, cleaned) ─
RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    ctranslate2==4.7.1 \
    TTS==0.22.0 \
    librosa \
    soundfile \
    scipy \
    numpy

# ── LLM / Transformers (FIXED COMPATIBILITY) ─
RUN pip install --no-cache-dir \
    transformers==4.41.0 \
    tokenizers \
    safetensors

# ── App requirements ─
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app ─
COPY . .

# ── Persistent Storage Directory Configuration ──────
# Create the /data volume point used in database.py and set permissions
# so it can be safely written to by container runtime users.
RUN mkdir -p /data && chmod 777 /data

RUN chmod +x /app/start.sh

# ── Environment Configuration for Free Tier ──────────
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# Expose the standard port used by Hugging Face Spaces
EXPOSE 7860

# Run the startup script
CMD ["/app/start.sh"]