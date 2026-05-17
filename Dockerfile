FROM python:3.10-slim

WORKDIR /app

# 1. System deps: Added build-essential and python3-dev so llama-cpp can compile!
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip & install wheels to speed up compilation steps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Layer 1 Cache: Install CPU PyTorch explicitly first (Heavy download)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchaudio==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Layer 2 Cache: Install heavy audio/ML dependencies next
# (Now with the compiler installed, llama-cpp-python will compile successfully)
RUN pip install --no-cache-dir \
    TTS==0.22.0 \
    faster-whisper==1.0.3 \
    ctranslate2==4.4.0 \
    llama-cpp-python==0.3.1

# 5. Layer 3 Cache: Copy requirements.txt and install remaining lightweight packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy app source and finalize
COPY . .
RUN chmod +x /app/start.sh

# HF Spaces requires the app to listen on 0.0.0.0:7860
EXPOSE 7860

CMD ["bash", "./start.sh"]