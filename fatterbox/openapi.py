"""FastAPI REST API server for Chatterbox TTS."""
import asyncio
import logging
import struct
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from .utils import split_text

_LOGGER = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    """Request model for TTS generation (OpenAPI compatible)."""
    input: str = Field(..., description="Text to synthesize", alias="text")
    voice: Optional[str] = Field(None, description="Voice name (from voices directory)")
    model: Optional[str] = Field("tts-1", description="Model identifier (for compatibility)")
    response_format: Optional[str] = Field("wav", description="Audio format (only wav supported)")
    speed: Optional[float] = Field(1.0, description="Playback speed (not implemented)")
    
    class Config:
        populate_by_name = True  # Allow both 'input' and 'text' field names


class VoiceInfo(BaseModel):
    """Voice information model."""
    name: str
    path: str


class ServerInfo(BaseModel):
    """Server information model."""
    name: str = "Fatterbox TTS API"
    version: str = "1.0.0"
    sample_rate: int
    voices: list[VoiceInfo]


def create_api(model, voices: dict, voices_dir: Path) -> FastAPI:
    """Create FastAPI application with OpenAPI-compatible endpoints."""
    app = FastAPI(
        title="Fatterbox TTS API",
        description="OpenAPI-compatible Chatterbox TTS with voice cloning and streaming",
        version="1.0.0"
    )
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/v1/info", response_model=ServerInfo)
    async def get_info():
        """Get server information and available voices."""
        voice_list = [
            VoiceInfo(name=name, path=path)
            for name, path in voices.items()
        ]
        return ServerInfo(
            sample_rate=model.sr,
            voices=voice_list
        )
    
    @app.get("/v1/voices")
    async def get_voices():
        """Get list of available voices."""
        return {"voices": list(voices.keys())}
    
    @app.post("/v1/audio/speech")
    async def synthesize(request: TTSRequest):
        """
        Synthesize speech from text (OpenAPI-compatible endpoint).
        Automatically streams audio using chunked transfer encoding and sentence splitting.
        
        Compatible with OpenAI's /v1/audio/speech endpoint.
        """
        # Accept both 'input' and 'text' field names for compatibility
        text = request.input
        
        _LOGGER.info(f"TTS request: '{text[:50]}...' with voice: {request.voice}")
        
        # Get audio prompt path
        audio_prompt_path = voices.get(request.voice) if request.voice else None
        
        if request.voice and not audio_prompt_path:
            raise HTTPException(status_code=404, detail=f"Voice '{request.voice}' not found")
        
        async def generate_chunks():
            """Generate audio chunks using sentence splitting (streaming)."""
            try:
                # Split text into chunks for streaming (same as Wyoming handler)
                chunks = split_text(text)
                
                # Send WAV header first
                header = _create_wav_header(model.sr)
                yield header
                
                start_time = time.time()
                first_chunk = True
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    chunk_start = time.time()
                    
                    # Generate audio for this chunk
                    audio = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _generate_audio_sync,
                        model,
                        chunk,
                        audio_prompt_path
                    )
                    
                    chunk_time = time.time() - chunk_start
                    
                    if first_chunk:
                        ttfa = time.time() - start_time
                        _LOGGER.info(f"First chunk generated (TTFA: {ttfa:.2f}s)")
                        first_chunk = False
                    
                    _LOGGER.info(f"Chunk {i+1}/{len(chunks)}: {chunk_time:.2f}s")
                    
                    # Convert to PCM bytes and yield
                    audio_bytes = (audio * 32767).numpy().astype("int16").tobytes()
                    yield audio_bytes
                
                total_time = time.time() - start_time
                _LOGGER.info(f"Streaming complete - Total time: {total_time:.2f}s")
            
            except Exception as e:
                _LOGGER.error(f"Error during streaming synthesis: {e}", exc_info=True)
                raise
            finally:
                # Clean up CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return StreamingResponse(
            generate_chunks(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "Cache-Control": "no-cache",
            }
        )
    
    return app


def _generate_audio_sync(model, text: str, audio_prompt_path: str = None) -> torch.Tensor:
    """Generate audio synchronously (same logic as Wyoming handler)."""
    with torch.no_grad():
        # Get backend and generation params from model
        backend = getattr(model, '_wyoming_backend', 'cudagraphs-manual')
        gen_params = getattr(model, '_wyoming_gen_params', {})
        
        # Performance parameters only
        t3_params = {
            "benchmark_t3": True,
            "generate_token_backend": backend,
            "skip_when_1": True,
        }
        
        # Add seed if specified
        if gen_params.get("seed"):
            t3_params["seed"] = gen_params["seed"]
        
        # Generate with or without voice cloning
        if audio_prompt_path:
            wav = model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=gen_params.get("exaggeration", 0.5),
                cfg_weight=gen_params.get("cfg_weight", 0.5),
                t3_params=t3_params,
            )
        else:
            # Use default voice if no prompt provided
            wav = model.generate(
                text,
                exaggeration=gen_params.get("exaggeration", 0.5),
                cfg_weight=gen_params.get("cfg_weight", 0.5),
                t3_params=t3_params,
            )
        
        # Move to CPU
        result = wav.squeeze().cpu()
    
    # Synchronize and cleanup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    return result


def _create_wav_header(sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Create a minimal WAV header for streaming (data size will be unknown)."""
    # RIFF header
    riff = b'RIFF'
    # File size (unknown for streaming, use max value)
    file_size = struct.pack('<I', 0xFFFFFFFF - 8)
    wave = b'WAVE'
    
    # fmt subchunk
    fmt = b'fmt '
    fmt_size = struct.pack('<I', 16)  # PCM
    audio_format = struct.pack('<H', 1)  # PCM
    num_channels = struct.pack('<H', channels)
    sample_rate_bytes = struct.pack('<I', sample_rate)
    byte_rate = struct.pack('<I', sample_rate * channels * sample_width)
    block_align = struct.pack('<H', channels * sample_width)
    bits_per_sample = struct.pack('<H', sample_width * 8)
    
    # data subchunk
    data = b'data'
    data_size = struct.pack('<I', 0xFFFFFFFF)  # Unknown size for streaming
    
    return (riff + file_size + wave +
            fmt + fmt_size + audio_format + num_channels + 
            sample_rate_bytes + byte_rate + block_align + bits_per_sample +
            data + data_size)
