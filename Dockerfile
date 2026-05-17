FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install cloudflared
RUN curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
    && dpkg -i cloudflared.deb \
    && rm cloudflared.deb

COPY environment.yml .

# Fix potential rootless permission issues during conda env creation
RUN chmod -R 777 /opt/conda

# Build the Conda environment and clean up cache to save space
RUN conda env create -f environment.yml && conda clean -afy

# Prepend the 'voice' environment to PATH
ENV PATH /opt/conda/envs/voice/bin:$PATH

COPY . .

# Ensure the entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]