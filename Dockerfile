FROM docker.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /chatter

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python package manager and dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --locked --no-editable 

COPY docker_init.py ./

RUN /chatter/.venv/bin/python docker_init.py

COPY fatterbox ./fatterbox

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["/chatter/.venv/bin/python", "-m", "fatterbox"]
