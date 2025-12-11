"""
Chatterbox TTS Server
OpenAI-compatible API + Wyoming Protocol support with Multi-Voice Support
"""
import argparse
import gc
import io
import logging
import os
import time
from functools import partial
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import asyncio
import pysbd
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from flask import Flask, Response, jsonify, render_template, request, send_file

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

from wyoming.info import Info, TtsProgram, TtsVoice, Attribution
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event


# ============================================================================
# Voice Management
# ============================================================================

class VoiceLoader:
    """Loads and manages multiple voice prompts from directory with runtime discovery"""
    
    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus'}
    OPENAI_VOICE_NAMES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    def __init__(self, voices_dir: str, default_voice: Optional[str] = None):
        self.voices_dir = Path(voices_dir)
        self.voices: Dict[str, str] = {}
        self.default_voice_name = default_voice
        self.last_scan_time = 0
        self.scan_lock = Lock()
        self.polling_thread = None
        self.should_poll = False
        
        # Initial scan
        if not self.voices_dir.exists():
            raise ValueError(f"Voices directory does not exist: {voices_dir}")
        if not self.voices_dir.is_dir():
            raise ValueError(f"Voices path is not a directory: {voices_dir}")
        
        self._scan_voices_directory()
        
        if not self.voices:
            raise ValueError(f"No audio files found in voices directory: {voices_dir}")
        
        # Set default voice
        if self.default_voice_name:
            if self.default_voice_name not in self.voices:
                logging.warning(f"Default voice '{self.default_voice_name}' not found, using first available")
                self.default_voice_name = list(self.voices.keys())[0]
        else:
            self.default_voice_name = list(self.voices.keys())[0]
        
        logging.info(f"Default voice set to: {self.default_voice_name}")
        logging.info(f"OpenAI-compatible voice names mapped to: {self.default_voice_name}")
    
    def _scan_voices_directory(self):
        """Scan directory and load all audio files as voices"""
        with self.scan_lock:
            voices_before = set(self.voices.keys())
            
            # Scan for audio files
            audio_files = []
            for ext in self.SUPPORTED_AUDIO_FORMATS:
                audio_files.extend(self.voices_dir.glob(f"*{ext}"))
                audio_files.extend(self.voices_dir.glob(f"*{ext.upper()}"))
            
            # Update voices dictionary
            new_voices = {}
            for audio_file in sorted(audio_files):
                voice_name = audio_file.stem
                new_voices[voice_name] = str(audio_file.absolute())
            
            # Detect changes
            voices_after = set(new_voices.keys())
            added = voices_after - voices_before
            removed = voices_before - voices_after
            
            if added:
                logging.info(f"Added voices: {sorted(added)}")
            if removed:
                logging.info(f"Removed voices: {sorted(removed)}")
            
            self.voices = new_voices
            self.last_scan_time = time.time()
            
            if not voices_before and self.voices:
                # Initial scan
                logging.info(f"Loaded {len(self.voices)} voices: {sorted(self.voices.keys())}")
    
    def _polling_loop(self):
        """Background thread that polls the directory for changes every 1 second"""
        logging.info("Voice polling started (interval: 1.0s)")
        while self.should_poll:
            time.sleep(1.0)
            if self.should_poll:  # Check again after sleep
                try:
                    self._scan_voices_directory()
                except Exception as e:
                    logging.error(f"Error during voice directory polling: {e}")
    
    def start_polling(self):
        """Start the background polling thread"""
        if self.polling_thread is None or not self.polling_thread.is_alive():
            self.should_poll = True
            self.polling_thread = Thread(target=self._polling_loop, daemon=True)
            self.polling_thread.start()
            logging.info("Voice directory polling enabled")
    
    def stop_polling(self):
        """Stop the background polling thread"""
        self.should_poll = False
        if self.polling_thread is not None:
            self.polling_thread.join(timeout=2.0)
            logging.info("Voice directory polling stopped")
    
    def get_voice_path(self, voice_name: str) -> Optional[str]:
        """Get audio prompt path for a voice name"""
        # Check if it's an OpenAI voice name
        if voice_name in self.OPENAI_VOICE_NAMES:
            voice_name = self.default_voice_name
        
        return self.voices.get(voice_name)
    
    def get_default_voice_path(self) -> str:
        """Get default voice audio prompt path"""
        return self.voices[self.default_voice_name]
    
    def list_voices(self) -> List[str]:
        """Get list of all available voice names"""
        return sorted(self.voices.keys())
    
    def get_all_voices_with_openai(self) -> List[str]:
        """Get all voices including OpenAI-compatible names"""
        actual_voices = self.list_voices()
        return sorted(set(actual_voices + self.OPENAI_VOICE_NAMES))


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Central configuration management"""
    def __init__(self, args):
        self.host = args.host
        self.openai_port = args.openai_port
        self.wyoming_port = args.wyoming_port
        self.debug = args.debug
        self.device = self._validate_device(args.device)
        self.stream = args.stream
        self.language_id = args.language_id or "en"
        
        # Voice management with polling
        self.voice_loader = VoiceLoader(
            voices_dir=args.voices_dir,
            default_voice=args.default_voice
        )
        
        # Generation parameters
        self.exaggeration = args.exaggeration
        self.temperature = args.temperature
        self.cfg_weight = args.cfg_weight
        self.min_p = args.min_p
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.dtype = args.dtype
        
        # T3 optimization params
        # FIXED: Disabled skip_when_1 to prevent CUDA graph capture errors
        # The skip_when_1 optimization causes "operation not permitted when stream is capturing"
        # errors because it tries to compare top_p tensor during graph capture
        self.t3_params = {
            "initial_forward_pass_backend": "eager",
            "generate_token_backend": "cudagraphs-manual",
            "stride_length": 4,
            "skip_when_1": False,  # Changed from True to prevent graph capture errors
        }
    
    @staticmethod
    def _validate_device(device: str) -> str:
        """Validate and fallback device selection"""
        if "cuda" in device and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if "mps" in device and not torch.backends.mps.is_available():
            logging.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return device


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model loading, optimization, and device management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[ChatterboxMultilingualTTS] = None
        self.inference_lock = Lock()
        self.generation_count = 0
        self.cached_conds: Dict[str, any] = {}
        self.current_audio_prompt = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Load and optimize the model"""
        logging.info("Loading Chatterbox TTS model...")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.config.device)
        
        if "cuda" in self.config.device:
            self._apply_cuda_optimizations()
            self._warmup()
    
    def _apply_cuda_optimizations(self):
        """Apply CUDA-specific optimizations"""
        # T3 dtype optimization
        if self.config.dtype != "float32":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            logging.info(f"Optimizing T3 model with dtype: {self.config.dtype}")
            self.model.t3.to(dtype=dtype_map[self.config.dtype])
            self.model.conds.t3.to(dtype=dtype_map[self.config.dtype])
            torch.cuda.empty_cache()
        
        # Torch compile
        if hasattr(torch, 'compile'):
            try:
                logging.info("Compiling model with torch.compile...")
                self.model.t3 = torch.compile(
                    self.model.t3,
                    mode="reduce-overhead",
                    fullgraph=True
                )
            except Exception as e:
                logging.warning(f"Compilation failed: {e}")
        
        # Hardware-specific optimizations
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("TF32 enabled for Ampere+ GPU")
        
        torch.backends.cudnn.benchmark = True
    
    def _warmup(self):
        """Warmup generations for CUDA graph initialization"""
        logging.info("Running warmup generation...")
        try:
            for text in ["Warmup generation for CUDA graph initialization.", 
                        "Second warmup for full speed."]:
                self.model.generate(text,
                                    language_id=self.config.language_id,
                                    t3_params=self.config.t3_params)
            logging.info("Warmup complete!")
        except Exception as e:
            logging.warning(f"Warmup failed: {e}")
    
    def cleanup_after_generation(self):
        """Cleanup after generation"""
        # Periodic cache clearing to prevent VRAM usage from growing indefinitely
        self.generation_count += 1
        if self.generation_count >= 5 and "cuda" in self.config.device:
            self.generation_count = 0
            torch.cuda.empty_cache()
            gc.collect()
            logging.debug("CUDA cache cleared")
    
    def prepare_voice_conditionals(self, audio_prompt_path: str):
        """Load or retrieve cached voice conditionals"""
        if audio_prompt_path == self.current_audio_prompt:
            logging.debug("Using existing conditionals")
            return
        
        voice_conds = self.cached_conds.get(audio_prompt_path)
        
        if voice_conds:
            logging.debug("Using cached conditionals")
            self.model.conds = voice_conds
        else:
            logging.info(f"Generating new conditionals for: {Path(audio_prompt_path).name}")
            self.model.prepare_conditionals(audio_prompt_path)
            self.cached_conds[audio_prompt_path] = self.model.conds
        
        self.current_audio_prompt = audio_prompt_path
    
    def generate(self, text: str, **kwargs) -> torch.Tensor:
        """Generate audio with thread safety"""
        with self.inference_lock:
            return self.model.generate(text, **kwargs)
    
    def generate_batch(self, texts: List[str], **kwargs) -> List[torch.Tensor]:
        """Generate audio batch if supported"""
        if len(texts) > 1 and hasattr(self.model, 'generate_batch'):
            logging.info(f"Using batch generation for {len(texts)} sentences")
            with self.inference_lock:
                return self.model.generate_batch(texts, **kwargs)
        
        # Fallback to sequential
        return [self.generate(text, **kwargs) for text in texts if text.strip()]


# ============================================================================
# Audio Processing
# ============================================================================

class AudioProcessor:
    """Handles audio processing and format conversion"""
    
    MIMETYPES = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/L16",
    }
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.segmenter = pysbd.Segmenter(language="en", clean=True)
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if len(text) <= 150:
            return [text]
        return self.segmenter.segment(text)
    
    @staticmethod
    def process_waveform(wav_tensor: torch.Tensor) -> torch.Tensor:
        """Clamp waveform to prevent clipping"""
        waveform = wav_tensor.squeeze(0).cpu()
        return torch.clamp(waveform, -1.0, 1.0)
    
    def to_pcm16(self, waveform: torch.Tensor) -> bytes:
        """Convert waveform to PCM16 bytes"""
        waveform_int16 = (waveform * 32767).to(torch.int16)
        return waveform_int16.numpy().tobytes()
    
    def save_audio(self, waveform: torch.Tensor, format: str) -> io.BytesIO:
        """
        Save audio waveform to a BytesIO buffer in the requested format.
        Uses raw PCM16 for 'pcm', pydub (ffmpeg) for everything else.
        """

        buffer = io.BytesIO()

        # Normalize tensor: [-1, 1] → int16 numpy
        waveform = waveform.cpu().numpy()
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

        int16_audio = (waveform * 32767).astype(np.int16).squeeze()

        # --- RAW PCM (streaming format) ---
        if format == "pcm":
            buffer.write(int16_audio.tobytes())
            buffer.seek(0)
            return buffer

        # --- Convert tensor → pydub AudioSegment ---
        # AudioSegment expects raw PCM bytes
        segment = AudioSegment(
            int16_audio.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1,
        )

        # --- Format normalization ---
        export_format = format
        codec_params = {}

        # Opus must be inside an OGG container
        if format == "opus":
            export_format = "ogg"
            codec_params["codec"] = "libopus"

        # AAC should be inside an M4A container
        elif format == "aac":
            export_format = "mp4"
            codec_params["codec"] = "aac"

        # FLAC/WAV/MP3 can export normally
        elif format in ("mp3", "wav", "flac"):
            pass
        else:
            raise ValueError(f"Unsupported output format: {format}")

        # --- Export through pydub/ffmpeg ---
        segment.export(buffer, format=export_format, **codec_params)
        buffer.seek(0)
        return buffer


# ============================================================================
# Flask API Server
# ============================================================================

class FlaskServer:
    """OpenAI-compatible API server"""
    
    def __init__(self, model_manager: ModelManager, audio_processor: AudioProcessor, config: Config):
        self.app = Flask(__name__)
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.config = config
        
        self._register_routes()
    
    def _register_routes(self):
        """Register Flask routes"""
        self.app.route("/v1/audio/speech", methods=["POST"])(self.openai_tts)
        self.app.route("/v1/models", methods=["GET"])(self.list_models)
        self.app.route("/v1/voices", methods=["GET"])(self.list_voices)
        # OpenWebUI compat
        self.app.route("/v1/audio/models", methods=["GET"])(self.list_models)
        self.app.route("/v1/audio/voices", methods=["GET"])(self.list_voices)
    
    def openai_tts(self):
        """OpenAI-compatible TTS endpoint"""
        payload = request.get_json(force=True)
        
        # Extract parameters
        text = payload.get("input", "")
        voice = payload.get("voice", self.config.voice_loader.default_voice_name)
        # OpenAI uses "speed" parameter, but we map it to cfg_weight
        # If "speed" is provided, use it; otherwise use default cfg_weight
        cfg_weight = payload.get("speed", self.config.cfg_weight)
        stream = payload.get("stream", self.config.stream)
        fmt = payload.get("response_format", "pcm").lower()
        
        if not text:
            return jsonify({"error": "Missing input in request"}), 400
        
        # Get audio prompt (polling handles discovery automatically)
        audio_prompt_path = self.config.voice_loader.get_voice_path(voice)
        if not audio_prompt_path:
            available_voices = self.config.voice_loader.get_all_voices_with_openai()
            return jsonify({
                "error": f"Voice '{voice}' not found",
                "available_voices": available_voices
            }), 400
        
        # Prepare model
        self.model_manager.prepare_voice_conditionals(audio_prompt_path)
        
        # Generation kwargs
        kwargs = {
            "language_id": self.config.language_id,
            "exaggeration": self.config.exaggeration,
            "temperature": self.config.temperature,
            "cfg_weight": cfg_weight,
            "min_p": self.config.min_p,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "t3_params": self.config.t3_params,
        }
        
        logging.info(f"OpenAI {'streaming' if stream else 'non-streaming'} request: text='{text[:50]}...', voice='{voice}', cfg_weight={cfg_weight}")
        
        if stream:
            return self._stream_response(text, fmt, kwargs)
        else:
            return self._full_response(text, fmt, kwargs)
    
    def _stream_response(self, text: str, fmt: str, kwargs: dict):
        """Generate streaming response"""
        if fmt not in ['pcm', 'mp3']:
            return jsonify({"error": f"Streaming only supports 'pcm' and 'mp3' formats"}), 400
        
        def generate_stream():
            try:
                sentences = self.audio_processor.split_sentences(text)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    wav_tensor = self.model_manager.generate(sentence, **kwargs)
                    waveform = self.audio_processor.process_waveform(wav_tensor)
                    
                    if fmt == 'pcm':
                        yield self.audio_processor.to_pcm16(waveform)
                    elif fmt == 'mp3':
                        buffer = self.audio_processor.save_audio(waveform, 'mp3')
                        yield buffer.getvalue()
            finally:
                self.model_manager.cleanup_after_generation()
        
        mimetype = self.audio_processor.MIMETYPES[fmt]
        return Response(generate_stream(), mimetype=mimetype)
    
    def _full_response(self, text: str, fmt: str, kwargs: dict):
        """Generate full response"""
        if fmt not in self.audio_processor.MIMETYPES:
            return jsonify({"error": f"Unsupported format: {fmt}"}), 400
        
        try:
            sentences = self.audio_processor.split_sentences(text)
            audio_chunks = self.model_manager.generate_batch(sentences, **kwargs)
            
            final_audio = torch.cat(audio_chunks, dim=-1) if len(audio_chunks) > 1 else audio_chunks[0]
            waveform = self.audio_processor.process_waveform(final_audio)
            
            buffer = self.audio_processor.save_audio(waveform, fmt)
            mimetype = self.audio_processor.MIMETYPES[fmt]
            
            return send_file(buffer, mimetype=mimetype)
        finally:
            self.model_manager.cleanup_after_generation()
    
    def list_models(self):
        """List available models"""
        return jsonify({
            "object": "list",
            "data": [
                {"id": "tts-1", "object": "model", "created": 1677610600, "owned_by": "openai"},
                {"id": "tts-1-hd", "object": "model", "created": 1677610600, "owned_by": "openai"}
            ]
        })
    
    def list_voices(self):
        """List available voices"""
        actual_voices = self.config.voice_loader.list_voices()
        all_voices = self.config.voice_loader.get_all_voices_with_openai()
        default_voice = self.config.voice_loader.default_voice_name
        
        return jsonify({
            "voices": all_voices,
            "actual_voices": actual_voices,
            "openai_voices": VoiceLoader.OPENAI_VOICE_NAMES,
            "default_voice": default_voice,
            "openai_mapping": f"OpenAI voices (alloy, echo, etc.) map to '{default_voice}'",
            "count": len(actual_voices)
        })
    
    def run(self):
        """Run Flask server"""
        self.app.run(
            host=self.config.host,
            port=self.config.openai_port,
            debug=self.config.debug,
            use_reloader=False
        )


# ============================================================================
# Wyoming Protocol Server
# ============================================================================

class WyomingEventHandler(AsyncEventHandler):
    """Handles Wyoming protocol events for TTS with parallel prefetching"""
    
    def __init__(self, wyoming_info: Info, model_manager: ModelManager, 
                 audio_processor: AudioProcessor, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.config = config
        
        # Audio format
        self.sample_rate = 24000  # Chatterbox outputs at 24kHz
        self.sample_width = 2
        self.channels = 1
        self.chunk_size = 2048  # Larger chunks for better throughput
        
        # Thread pool for parallel generation (max 3 concurrent as per wyoming_openai)
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Streaming state
        self.streaming_text_buffer = []
        self.streaming_voice = None
        self.is_streaming = False
    
    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming protocol events"""
        if event.type == "describe":
            logging.info("Received describe event")
            # Refresh Wyoming info to include any newly added voices
            updated_info = self._get_updated_wyoming_info()
            await self.write_event(updated_info.event())
            return True
        
        # Streaming protocol
        if SynthesizeStart.is_type(event.type):
            synth_start = SynthesizeStart.from_event(event)
            self.streaming_voice = synth_start.voice.name if synth_start.voice else self.config.voice_loader.default_voice_name
            self.streaming_text_buffer = []
            self.is_streaming = True
            logging.info(f"Wyoming TTS streaming started with voice '{self.streaming_voice}'")
            return True
        
        if SynthesizeChunk.is_type(event.type):
            synth_chunk = SynthesizeChunk.from_event(event)
            self.streaming_text_buffer.append(synth_chunk.text)
            logging.debug(f"Received text chunk: '{synth_chunk.text}'")
            return True
        
        if SynthesizeStop.is_type(event.type):
            logging.info("Wyoming TTS streaming stopped")
            full_text = "".join(self.streaming_text_buffer)
            await self._generate_and_stream_audio(full_text, self.streaming_voice or self.config.voice_loader.default_voice_name)
            await self.write_event(SynthesizeStopped().event())
            
            self.streaming_text_buffer = []
            self.streaming_voice = None
            self.is_streaming = False
            return True
        
        # Legacy protocol
        if Synthesize.is_type(event.type):
            if self.is_streaming:
                logging.warning("Ignoring legacy Synthesize during streaming session")
                return True
            
            synthesize = Synthesize.from_event(event)
            voice_name = synthesize.voice.name if synthesize.voice else self.config.voice_loader.default_voice_name
            logging.info(f"Wyoming TTS request (legacy): '{synthesize.text}' with voice '{voice_name}'")
            await self._generate_and_stream_audio(synthesize.text, voice_name)
            return True
        
        logging.warning(f"Unexpected event type: {event.type}")
        return True
    
    def _get_updated_wyoming_info(self) -> Info:
        """Get updated Wyoming info with current voices"""
        return Info(
            tts=[TtsProgram(
                name="chatterbox",
                version="1.0.0",
                description="Chatterbox TTS with voice cloning",
                attribution=Attribution(
                    name="Chatterbox",
                    url="https://github.com/chatterbox-tts/chatterbox"
                ),
                installed=True,
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=voice_name,
                        version="1.0.0",
                        description=f"Chatterbox voice: {voice_name}",
                        attribution=Attribution(name="Chatterbox", url=""),
                        installed=True,
                        languages=[self.config.language_id]
                    )
                    for voice_name in self.config.voice_loader.list_voices()
                ]
            )]
        )
    
    async def _generate_and_stream_audio(self, text: str, voice_name: str):
        """Generate audio with parallel prefetching and stream via Wyoming protocol"""
        loop = asyncio.get_event_loop()
        
        # Get voice audio prompt (polling handles discovery automatically)
        audio_prompt_path = self.config.voice_loader.get_voice_path(voice_name)
        if not audio_prompt_path:
            logging.error(f"Voice '{voice_name}' not found, using default")
            audio_prompt_path = self.config.voice_loader.get_default_voice_path()
        
        # Prepare voice conditionals in executor to not block
        await loop.run_in_executor(
            self.executor, self.model_manager.prepare_voice_conditionals, audio_prompt_path
        )
        
        # Generation kwargs - uses default cfg_weight from config
        kwargs = {
            "language_id": self.config.language_id,
            "exaggeration": self.config.exaggeration,
            "temperature": self.config.temperature,
            "cfg_weight": self.config.cfg_weight,
            "min_p": self.config.min_p,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "t3_params": self.config.t3_params,
        }
        
        logging.info(f"Wyoming request: text='{text[:50]}...', voice='{voice_name}', cfg_weight={self.config.cfg_weight}")
        
        # Send audio start
        await self.write_event(AudioStart(
            rate=self.sample_rate,
            width=self.sample_width,
            channels=self.channels
        ).event())
        
        # Split into sentences
        sentences = [s for s in self.audio_processor.split_sentences(text) if s.strip()]
        
        if not sentences:
            await self.write_event(AudioStop().event())
            return
        
        # Use parallel prefetching with async queue for low latency
        # This mimics wyoming_openai's approach of prefetching up to 3 requests
        await self._parallel_generate_and_stream(sentences, kwargs, loop)
        
        # Send audio stop
        await self.write_event(AudioStop().event())
        await loop.run_in_executor(self.executor, self.model_manager.cleanup_after_generation)
        logging.info("Finished Wyoming TTS generation")
    
    async def _parallel_generate_and_stream(self, sentences: List[str], kwargs: dict, loop):
        """Generate sentences in parallel (up to 3) and stream sequentially"""
        # Queue to hold generated audio futures
        futures = []
        
        # Submit up to 3 generation tasks at a time
        for i, sentence in enumerate(sentences):
            # Wait if we have 3 tasks in flight
            if len(futures) >= 3:
                # Pop and stream the first completed task
                wav_tensor = await futures.pop(0)
                await self._stream_audio_chunk(wav_tensor)
            
            # Submit next generation task
            logging.debug(f"Submitting generation for sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
            
            # Create a wrapper function that properly unpacks kwargs
            def generate_with_kwargs(text=sentence, params=kwargs):
                return self.model_manager.generate(text, **params)
            
            future = loop.run_in_executor(self.executor, generate_with_kwargs)
            futures.append(future)
        
        # Stream remaining futures in order
        for future in futures:
            wav_tensor = await future
            await self._stream_audio_chunk(wav_tensor)
    
    async def _stream_audio_chunk(self, wav_tensor: torch.Tensor):
        """Stream a single audio chunk"""
        waveform = self.audio_processor.process_waveform(wav_tensor)
        audio_bytes = self.audio_processor.to_pcm16(waveform)
        
        # Stream in larger chunks for better throughput, minimal delay
        for i in range(0, len(audio_bytes), self.chunk_size):
            chunk = audio_bytes[i:i + self.chunk_size]
            await self.write_event(AudioChunk(
                rate=self.sample_rate,
                width=self.sample_width,
                channels=self.channels,
                audio=chunk
            ).event())
            
            # Small delay to prevent overwhelming the client
            if i + self.chunk_size < len(audio_bytes):
                await asyncio.sleep(0.0001)


class WyomingServer:
    """Wyoming protocol server"""
    
    def __init__(self, model_manager: ModelManager, audio_processor: AudioProcessor, config: Config):
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.config = config
    
    def _create_wyoming_info(self) -> Info:
        """Create Wyoming info with available voices"""
        return Info(
            tts=[TtsProgram(
                name="chatterbox",
                version="1.0.0",
                description="Chatterbox TTS with voice cloning",
                attribution=Attribution(
                    name="Chatterbox",
                    url="https://github.com/chatterbox-tts/chatterbox"
                ),
                installed=True,
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=voice_name,
                        version="1.0.0",
                        description=f"Chatterbox voice: {voice_name}",
                        attribution=Attribution(name="Chatterbox", url=""),
                        installed=True,
                        languages=[self.config.language_id]
                    )
                    for voice_name in self.config.voice_loader.list_voices()
                ]
            )]
        )
    
    async def run(self):
        """Run Wyoming server"""
        wyoming_info = self._create_wyoming_info()
        logging.info(f"Wyoming server starting on port {self.config.wyoming_port}")
        
        await AsyncServer.from_uri(f"tcp://0.0.0.0:{self.config.wyoming_port}").run(
            partial(WyomingEventHandler, wyoming_info, self.model_manager, 
                   self.audio_processor, self.config)
        )


# ============================================================================
# Utilities
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for Chatterbox with Wyoming protocol support and multi-voice support."
    )
    
    # Server arguments
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--openai-port", type=int, default=int(os.getenv("OPENAI_PORT", os.getenv("PORT", "5002"))))
    parser.add_argument("--wyoming-port", type=int, default=int(os.getenv("WYOMING_PORT", "10200")))
    parser.add_argument("--debug", action="store_true", default=os.getenv("DEBUG", "").lower() == "true")
    parser.add_argument("--device", type=str, default=os.getenv("DEVICE", "cpu"))
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction,
                       default=os.getenv("STREAM", "").lower() == "true")
    parser.add_argument("--language-id", type=str, default=os.getenv("LANGUAGE_ID", "en"))
    
    # Voice configuration
    parser.add_argument("--voices-dir", type=str, default=os.getenv("VOICES_DIR"),
                       help="Directory containing voice audio files (wav, mp3, flac, etc.)")
    parser.add_argument("--default-voice", type=str, default=os.getenv("DEFAULT_VOICE"),
                       help="Default voice name for OpenAI-compatible voice names (alloy, echo, etc.)")
    
    # Generation arguments
    parser.add_argument("--exaggeration", type=float, default=float(os.getenv("EXAGGERATION", "0.5")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.8")))
    parser.add_argument("--cfg-weight", type=float, default=float(os.getenv("CFG_WEIGHT", "1.0")),
                       help="Classifier-free guidance weight (default: 1.0)")
    parser.add_argument("--min-p", type=float, default=float(os.getenv("MIN_P", "0.05")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "1.0")))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.getenv("REPETITION_PENALTY", "1.2")))
    parser.add_argument("--dtype", type=str, default=os.getenv("DTYPE", os.getenv("T3_DTYPE", "bfloat16")),
                       choices=["float32", "float16", "bfloat16"])
    
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.debug else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize components
    config = Config(args)
    model_manager = ModelManager(config)
    audio_processor = AudioProcessor()
    
    # Start voice directory polling
    config.voice_loader.start_polling()
    
    # Start Flask server in background thread
    openapi_server = FlaskServer(model_manager, audio_processor, config)
    flask_thread = Thread(target=openapi_server.run, daemon=True)
    flask_thread.start()
    
    logging.info(f"Flask server started on {config.host}:{config.openai_port}")
    logging.info(f"Starting Wyoming server on port {config.wyoming_port}...")
    
    # Run Wyoming server
    try:
        wyoming_server = WyomingServer(model_manager, audio_processor, config)
        asyncio.run(wyoming_server.run())
    except KeyboardInterrupt:
        logging.info("Shutting down servers...")
        config.voice_loader.stop_polling()


if __name__ == "__main__":
    main()
