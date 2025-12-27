#!/usr/bin/env python3
"""
Wyoming Protocol TTS Server for Chatterbox-Turbo
Home Assistant compatible real-time text-to-speech service
"""
import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime
import librosa
import soundfile as sf
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize
from wyoming.info import Describe

# Model constants
MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64

_LOGGER = logging.getLogger(__name__)


class RepetitionPenaltyLogitsProcessor:
    """Apply repetition penalty to prevent model from repeating tokens"""
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"penalty must be positive float, got {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


class ChatterboxTTS:
    """Chatterbox-Turbo TTS inference engine"""
    
    def __init__(self, dtype: str = "fp32", voices_dir: str = "./voices", use_cuda: bool = False):
        self.dtype = dtype
        self.voices_dir = Path(voices_dir)
        self.voices: Dict[str, np.ndarray] = {}
        self.tokenizer = None
        self.use_cuda = use_cuda
        
        # Configure execution providers for ONNX
        if self.use_cuda:
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            _LOGGER.info("CUDA acceleration enabled")
        else:
            self.providers = ['CPUExecutionProvider']
            _LOGGER.info("Using CPU execution")
        
        # ONNX sessions
        self.speech_encoder_session = None
        self.embed_tokens_session = None
        self.language_model_session = None
        self.cond_decoder_session = None
        
    def download_model(self, name: str) -> str:
        """Download ONNX model from HuggingFace"""
        suffix = "" if self.dtype == "fp32" else "_quantized" if self.dtype == "q8" else f"_{self.dtype}"
        filename = f"{name}{suffix}.onnx"
        _LOGGER.info(f"Downloading {filename}...")
        graph = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename)
        hf_hub_download(MODEL_ID, subfolder="onnx", filename=f"{filename}_data")
        return graph
    
    def load_models(self):
        """Load all ONNX models and tokenizer"""
        _LOGGER.info("Loading models...")
        
        # Download models
        conditional_decoder_path = self.download_model("conditional_decoder")
        speech_encoder_path = self.download_model("speech_encoder")
        embed_tokens_path = self.download_model("embed_tokens")
        language_model_path = self.download_model("language_model")
        
        # Create ONNX sessions
        _LOGGER.info("Creating ONNX inference sessions...")
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.speech_encoder_session = onnxruntime.InferenceSession(
            speech_encoder_path, 
            sess_options=sess_options,
            providers=self.providers
        )
        self.embed_tokens_session = onnxruntime.InferenceSession(
            embed_tokens_path,
            sess_options=sess_options,
            providers=self.providers
        )
        self.language_model_session = onnxruntime.InferenceSession(
            language_model_path,
            sess_options=sess_options,
            providers=self.providers
        )
        self.cond_decoder_session = onnxruntime.InferenceSession(
            conditional_decoder_path,
            sess_options=sess_options,
            providers=self.providers
        )
        
        # Log which provider is actually being used
        for name, session in [
            ("speech_encoder", self.speech_encoder_session),
            ("embed_tokens", self.embed_tokens_session),
            ("language_model", self.language_model_session),
            ("cond_decoder", self.cond_decoder_session)
        ]:
            provider = session.get_providers()[0]
            _LOGGER.debug(f"{name} using: {provider}")
        
        # Load tokenizer
        _LOGGER.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        _LOGGER.info("Models loaded successfully")
    
    def load_voices(self):
        """Load all voice reference audio files from voices directory"""
        if not self.voices_dir.exists():
            _LOGGER.warning(f"Voices directory not found: {self.voices_dir}")
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            return
        
        _LOGGER.info(f"Loading voices from {self.voices_dir}...")
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        
        for audio_file in self.voices_dir.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                try:
                    # Load audio and resample to model sample rate
                    audio_values, _ = librosa.load(str(audio_file), sr=SAMPLE_RATE)
                    audio_values = audio_values[np.newaxis, :].astype(np.float32)
                    
                    # Use filename without extension as voice name
                    voice_name = audio_file.stem
                    self.voices[voice_name] = audio_values
                    
                    _LOGGER.info(f"Loaded voice: {voice_name} ({audio_file.name})")
                except Exception as e:
                    _LOGGER.error(f"Failed to load {audio_file.name}: {e}")
        
        if not self.voices:
            _LOGGER.warning("No voices loaded. Add .wav files to voices directory.")
    
    def synthesize(
        self,
        text: str,
        voice_name: str,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.2
    ) -> np.ndarray:
        """
        Synthesize speech from text using specified voice
        
        Args:
            text: Text to synthesize
            voice_name: Name of voice to use (must exist in self.voices)
            max_new_tokens: Maximum tokens to generate
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Audio waveform as numpy array
        """
        if voice_name not in self.voices:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {list(self.voices.keys())}")
        
        _LOGGER.info(f"Synthesizing with voice '{voice_name}': {text[:50]}...")
        
        # Get reference audio for this voice
        audio_values = self.voices[voice_name]
        
        # Tokenize text
        input_ids = self.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        
        # Generation loop
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        
        for i in range(max_new_tokens):
            inputs_embeds = self.embed_tokens_session.run(None, {"input_ids": input_ids})[0]
            
            if i == 0:
                # First iteration: encode reference audio
                ort_speech_encoder_input = {"audio_values": audio_values}
                cond_emb, prompt_token, speaker_embeddings, speaker_features = \
                    self.speech_encoder_session.run(None, ort_speech_encoder_input)
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
                
                # Initialize cache and LLM inputs
                batch_size, seq_len, _ = inputs_embeds.shape
                past_key_values = {
                    inp.name: np.zeros(
                        [batch_size, NUM_KV_HEADS, 0, HEAD_DIM],
                        dtype=np.float16 if inp.type == 'tensor(float16)' else np.float32
                    )
                    for inp in self.language_model_session.get_inputs()
                    if "past_key_values" in inp.name
                }
                attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
                position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1).repeat(batch_size, axis=0)
            
            # Run language model
            logits, *present_key_values = self.language_model_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_key_values,
            ))
            
            # Apply repetition penalty and sample next token
            logits = logits[:, -1, :]
            next_token_logits = repetition_penalty_processor(generate_tokens, logits)
            input_ids = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens = np.concatenate((generate_tokens, input_ids), axis=-1)
            
            # Check for stop token
            if (input_ids.flatten() == STOP_SPEECH_TOKEN).all():
                break
            
            # Update for next iteration
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
            position_ids = position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]
        
        # Decode audio from tokens
        speech_tokens = generate_tokens[:, 1:-1]
        silence_tokens = np.full((speech_tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
        speech_tokens = np.concatenate([prompt_token, speech_tokens, silence_tokens], axis=1)
        
        wav = self.cond_decoder_session.run(None, dict(
            speech_tokens=speech_tokens,
            speaker_embeddings=speaker_embeddings,
            speaker_features=speaker_features,
        ))[0].squeeze(axis=0)
        
        _LOGGER.info(f"Synthesized {len(wav) / SAMPLE_RATE:.2f}s of audio")
        return wav


class ChatterboxEventHandler(AsyncEventHandler):
    """Handle Wyoming protocol events for TTS"""
    
    def __init__(
        self,
        tts_engine: ChatterboxTTS,
        wyoming_info: Info,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tts_engine = tts_engine
        self.wyoming_info_event = wyoming_info.event()
        self.default_voice = next(iter(tts_engine.voices.keys())) if tts_engine.voices else None
    
    async def handle_event(self, event: Event) -> bool:
        """Process incoming Wyoming events"""
        if Describe.is_type(event.type):
            # Send server info in response to describe request
            await self.write_event(self.wyoming_info_event)
            return True
        
        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            _LOGGER.debug(f"Received synthesis request: {synthesize.text}")
            
            # Determine which voice to use
            voice_name = synthesize.voice.name if synthesize.voice else self.default_voice
            if not voice_name:
                _LOGGER.error("No voice specified and no default voice available")
                return False
            
            try:
                # Send audio start event immediately for low latency perception
                await self.write_event(
                    AudioStart(
                        rate=SAMPLE_RATE,
                        width=2,  # 16-bit = 2 bytes
                        channels=1
                    ).event()
                )
                
                # Synthesize audio in background thread
                wav = await asyncio.to_thread(
                    self.tts_engine.synthesize,
                    synthesize.text,
                    voice_name
                )
                
                # Convert to 16-bit PCM
                wav_int16 = (wav * 32767).astype(np.int16)
                
                # Stream audio in small chunks for responsive playback
                chunk_size = 1024  # Small chunks for lower latency
                for i in range(0, len(wav_int16), chunk_size):
                    chunk = wav_int16[i:i + chunk_size]
                    await self.write_event(
                        AudioChunk(
                            rate=SAMPLE_RATE,
                            width=2,
                            channels=1,
                            audio=chunk.tobytes()
                        ).event()
                    )
                
                # Send audio stop event
                await self.write_event(AudioStop().event())
                _LOGGER.debug("Audio streaming complete")
                
            except Exception as e:
                _LOGGER.error(f"Synthesis failed: {e}", exc_info=True)
                return False
        
        return True


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Wyoming Chatterbox TTS Server")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0, env: HOST)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "10200")),
        help="Port to bind to (default: 10200, env: PORT)"
    )
    parser.add_argument(
        "--voices-dir",
        default=os.environ.get("VOICES_DIR", "./voices"),
        help="Directory containing voice reference audio files (default: ./voices, env: VOICES_DIR)"
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("DTYPE", "fp32"),
        choices=["fp32", "fp16", "q8", "q4", "q4f16"],
        help="Model data type (default: fp32, env: DTYPE)"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=os.environ.get("CUDA", "").lower() in ("1", "true", "yes"),
        help="Use CUDA GPU acceleration (default: False, env: CUDA)"
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO, env: LOG_LEVEL)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    _LOGGER.info("=" * 60)
    _LOGGER.info("Wyoming Chatterbox TTS Server")
    _LOGGER.info("=" * 60)
    
    # Check CUDA availability if requested
    if args.cuda:
        cuda_available = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        if not cuda_available:
            _LOGGER.warning("CUDA requested but not available. Install onnxruntime-gpu for CUDA support.")
            _LOGGER.warning("Falling back to CPU execution.")
        else:
            _LOGGER.info("CUDA is available and will be used for acceleration")
    
    # Initialize TTS engine
    tts_engine = ChatterboxTTS(
        dtype=args.dtype,
        voices_dir=args.voices_dir,
        use_cuda=args.cuda
    )
    
    try:
        tts_engine.load_models()
        tts_engine.load_voices()
    except Exception as e:
        _LOGGER.error(f"Failed to initialize TTS engine: {e}", exc_info=True)
        return 1
    
    if not tts_engine.voices:
        _LOGGER.error("No voices available. Cannot start server.")
        return 1
    
    # Build Wyoming info
    voices = [
        TtsVoice(
            name=name,
            description=f"Chatterbox voice: {name}",
            attribution=Attribution(
                name="Resemble AI",
                url="https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX"
            ),
            installed=True,
            version="1.0.0",
            languages=["en"]  # Chatterbox supports English
        )
        for name in tts_engine.voices.keys()
    ]
    
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="chatterbox-turbo",
                description="Resemble AI Chatterbox Turbo TTS",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX"
                ),
                installed=True,
                version="1.0.0",
                voices=voices
            )
        ]
    )
    
    # Start Wyoming server
    _LOGGER.info(f"Starting server on {args.host}:{args.port}")
    _LOGGER.info(f"Available voices: {', '.join(tts_engine.voices.keys())}")
    
    server = AsyncServer.from_uri(f"tcp://{args.host}:{args.port}")
    
    # Use partial to bind custom parameters, leaving reader/writer for Wyoming
    await server.run(partial(
        ChatterboxEventHandler,
        tts_engine,
        wyoming_info
    ))
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down...")
        sys.exit(0)
