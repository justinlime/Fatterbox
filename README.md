# Fatterbox Usage Guide

Fatterbox is a Wyoming protocol and OpenAPI-compatible wrapper for Chatterbox TTS with voice cloning support.

## Overview

Fatterbox is built on [rsxdalv's optimized Chatterbox implementation](https://github.com/rsxdalv/chatterbox/tree/faster), exposing both Wyoming protocol and OpenAPI endpoints with streaming support for minimal time-to-first-word latency. The streaming architecture splits text into sentence chunks and generates audio progressively, allowing playback to begin before the entire text is synthesized.

## Requirements

- Docker with NVIDIA GPU support ([install nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- NVIDIA GPU with CUDA capability
- Voice reference files (.wav format)

## Quick Start

1. **Prepare voice files**: Place `.wav` files in a `voices` directory. Each file becomes a voice (e.g., `Jake.wav` → voice name "Jake").

2. **Pull the prebuilt image** (or build your own with `docker build -t fatterbox .`):
```bash
docker pull docker.io/justinlime/fatterbox:v0.1.0
```

3. **Run the container**:
```bash
docker run --gpus all \
  -v ./voices:/chatter/voices \
  -p 10200:10200 \
  -p 8000:8000 \
  docker.io/justinlime/fatterbox:v0.1.0
```

## Servers

Two servers run simultaneously:

- **Wyoming protocol**: `tcp://0.0.0.0:10200` (Home Assistant integration)
- **OpenAPI REST**: `http://0.0.0.0:8000` (OpenAI-compatible)

## Configuration

Configure via environment variables (all prefixed with `FATTERBOX_`):

### Server Configuration
```bash
FATTERBOX_WYOMING_HOST=0.0.0.0
FATTERBOX_WYOMING_PORT=10200
FATTERBOX_OPENAPI_HOST=0.0.0.0
FATTERBOX_OPENAPI_PORT=8000
FATTERBOX_VOICES_DIR=./voices
```

### Model Configuration
```bash
FATTERBOX_DEVICE=cuda              # cuda or cpu
FATTERBOX_DTYPE=bf16               # float32, fp16, bf16 (bf16 recommended)
FATTERBOX_BACKEND=cudagraphs-manual # fastest option
```

**VRAM Usage:**
- FP32: ~4.5 GB
- FP16/BF16: ~3.5 GB (recommended)

BF16 offers the best balance of speed and memory efficiency on modern GPUs.

### Generation Parameters
```bash
FATTERBOX_EXAGGERATION=0.5      # Emotional expressiveness (0.0-2.0)
FATTERBOX_CFG_WEIGHT=0.5        # Voice adherence (0.0-1.0)
FATTERBOX_TEMPERATURE=0.8       # Randomness (0.05-5.0)
FATTERBOX_SEED=0                # Random seed (0=random)
FATTERBOX_TOP_P=1.0             # Nucleus sampling (0.0-1.0)
FATTERBOX_MIN_P=0.0             # Min probability (0.0-1.0)
FATTERBOX_MAX_NEW_TOKENS=4096   # Max audio tokens (~25 per second)
FATTERBOX_N_TIMESTEPS=10        # Diffusion steps
FATTERBOX_FLOW_CFG_SCALE=1.0    # Mel decoder CFG scale
FATTERBOX_DEBUG=false           # Enable debug logging
```

### Example with custom settings:
```bash
docker run --gpus all \
  -v ./voices:/chatter/voices \
  -p 10200:10200 \
  -p 8000:8000 \
  -e FATTERBOX_DTYPE=bf16 \
  -e FATTERBOX_EXAGGERATION=0.7 \
  -e FATTERBOX_CFG_WEIGHT=0.4 \
  fatterbox
```

## API Usage

### OpenAPI Endpoint
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test.",
    "voice": "Jake"
  }' \
  --output speech.wav
```

### List Available Voices
```bash
curl http://localhost:8000/v1/voices
```

## Wyoming Protocol

Compatible with Home Assistant's Wyoming protocol. Configure in Home Assistant using:
- Host: `<docker-host-ip>`
- Port: `10200`

## Performance Tips

- Use `bf16` dtype (recommended) for best balance of speed and VRAM efficiency
- RTX 30xx/40xx GPUs have native BF16 support for optimal performance
- Use `cudagraphs-manual` backend (default) for fastest generation
- Lower `EXAGGERATION` and `CFG_WEIGHT` for more expressive speech
- Single-sentence streaming provides lowest latency
