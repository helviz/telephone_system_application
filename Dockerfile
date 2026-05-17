FROM python:3.10

WORKDIR /app

# Install deps INCLUDING pax-utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    pax-utils \
    binutils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchaudio==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    TTS==0.22.0 \
    faster-whisper==1.0.3 \
    llama-cpp-python==0.3.1

# IMPORTANT FIX
RUN find /usr/local/lib/python3.10/site-packages/ctranslate2 -name "*.so*" \
    -exec execstack -c {} \; || true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["bash", "./start.sh"]