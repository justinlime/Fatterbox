# Use Python 3.11 slim image as base
# NOTE: Build does NOT require NVIDIA/CUDA - only runtime does
FROM python:3.11-slim

# Set working directory
WORKDIR /chatter

# Install system dependencies (no CUDA needed for build)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for offline model caching
ENV HF_HOME=/chatter/.cache/huggingface
ENV TRANSFORMERS_CACHE=/chatter/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/chatter/.cache/huggingface/datasets
ENV TORCH_HOME=/chatter/.cache/torch
ENV NLTK_DATA=/chatter/.cache/nltk_data

# Create cache directories
RUN mkdir -p ${HF_HOME} ${TORCH_HOME} ${NLTK_DATA}

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python package manager and dependencies
RUN pip install --no-cache-dir uv && \
    uv venv --python /usr/local/bin/python && \
    uv sync --locked --no-editable

COPY docker_init.py ./

# Run initialization script to download all models (CPU mode - no CUDA needed)
# Models are downloaded on CPU but will be loaded on GPU at runtime if available
RUN /chatter/.venv/bin/python docker_init.py

COPY src/ ./

# Set environment variable to disable online lookups at runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Expose ports
EXPOSE 5002 10200

# Run the application
# To use with GPU at runtime, run with: docker run --gpus all ...
CMD ["/chatter/.venv/bin/python", "main.py"]