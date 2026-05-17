FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN chmod -R 777 /opt/conda

RUN conda env create -f environment.yml && conda clean -afy

ENV PATH /opt/conda/envs/voice/bin:$PATH

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["bash", "/app/start.sh"]