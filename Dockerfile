FROM python:3.10-slim

WORKDIR /app

# System deps: audio libs + ffmpeg for faster-whisper/librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer caching)
COPY requirements.txt .

# Install CPU torch first from the PyTorch index, then the rest
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchaudio==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

RUN chmod +x /app/start.sh

# HF Spaces requires the app to listen on 0.0.0.0:7860
EXPOSE 7860

CMD ["bash", "./start.sh"]