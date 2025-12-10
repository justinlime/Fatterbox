# Fatterbox TTS server

A lightweight, production-ready TTS server that supports **OpenAI-compatible API** and **Wyoming Protocol** for voice synthesis. Built for easy deployment with Docker and minimal configuration.

---

## 🐳 Docker Image

Use the pre-built Docker image from Docker Hub:

```bash
docker pull docker.io/justinlime/fatterbox:latest
```

> This image includes all dependencies, optimized for GPU/CPU inference and ready to run.

---

## 🚀 Quick Start

### Run the Container
```bash
docker run -p 5002:5002 -p 10200:10200 \
  -e HOST=0.0.0.0 \
  -e OPENAI_PORT=5002 \
  -e WYOMING_PORT=10200 \
  -e DEVICE=cuda \
  -e AUDIO_PROMPT=/audio/prompt.wav \
  -e LANGUAGE_ID=en \
  -e STREAM=true \
  -e EXAGGERATION=0.5 \
  -e TEMPERATURE=0.8 \
  -e MIN_P=0.05 \
  -e TOP_P=1.0 \
  -e REPETITION_PENALTY=1.2 \
  -e DTYPE=bfloat16 \
  -v /path/to/audio/prompt.wav:/audio/prompt.wav \
  --name chatterbox-tts \
  docker.io/justinlime/fatterbox:latest
```

> Replace `/path/to/audio/prompt.wav` with the actual path to your audio prompt file.

---

## 🛠️ Environment Variables

| Variable | Description | Default |
|--------|-------------|--------|
| `HOST` | Server IP | `0.0.0.0` |
| `OPENAI_PORT` | OpenAI API port | `5002` |
| `WYOMING_PORT` | Wyoming protocol port | `10200` |
| `DEBUG` | Enable debug logs | `false` |
| `DEVICE` | `cpu`, `cuda`, or `mps` | `cpu` |
| `STREAM` | Enable streaming | `true` |
| `LANGUAGE_ID` | Language ID | `en` |
| `AUDIO_PROMPT` | Path to audio prompt | (required) |
| `EXAGGERATION` | Voice expression | `0.5` |
| `TEMPERATURE` | Sampling temperature | `0.8` |
| `MIN_P` | Nucleus sampling | `0.05` |
| `TOP_P` | Top-p sampling | `1.0` |
| `REPETITION_PENALTY` | Repetition control | `1.2` |
| `DTYPE` | Model precision | `bfloat16` |

---

## 🚀 Usage

### OpenAI-Compatible API
```bash
curl -X POST http://localhost:5002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test.",
    "voice": "alloy",
    "response_format": "mp3"
  }'
```

### Wyoming Protocol
Connect via:
```bash
wyoming-client --server tcp://localhost:10200 --text "Hello world"
```

---

## 🧩 Features

- ✅ OpenAI API compatibility
- ✅ Wyoming Protocol support
- ✅ Voice cloning with audio prompt
- ✅ Multi-lingual support
- ✅ GPU acceleration (CUDA/MPS)
- ✅ Mixed precision (bfloat16/float16)
- ✅ Streaming & batch inference
- ✅ Low-latency design

---

## 🧩 Model Requirements

- Chatterbox TTS model (pretrained)
- Audio prompt file (for voice cloning)
- Supported languages: `en`, `de`, `fr`, `es`, etc.

---

## 🧪 Testing

### Test API
```bash
curl -X POST http://localhost:5002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox jumps over the lazy dog.",
    "voice": "nova",
    "response_format": "mp3"
  }' > output.mp3
```

### Test Wyoming
```bash
wyoming-client --server tcp://localhost:10200 --text "Hello world"
```

---

## 🧩 Build & Run

### Use Pre-Built Image
```bash
docker pull docker.io/justinlime/fatterbox:latest
docker run -p 5002:5002 -p 10200:10200 -e DEVICE=cuda -v /path/to/audio/prompt.wav:/audio/prompt.wav docker.io/justinlime/fatterbox:latest
```
