"""
Chatterbox TTS Server
OpenAI-compatible API + Wyoming Protocol support
"""
import argparse
import gc
import io
import logging
import os
from functools import partial
from threading import Lock, Thread
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import asyncio
import pysbd
import torch
import torchaudio
from flask import Flask, Response, jsonify, render_template, request, send_file

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

from wyoming.info import Info, TtsProgram, TtsVoice, Attribution
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event


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
        self.audio_prompt = args.audio_prompt
        
        # Generation parameters
        self.exaggeration = args.exaggeration
        self.temperature = args.temperature
        self.min_p = args.min_p
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.dtype = args.dtype
        
        # T3 optimization params
        self.t3_params = {
            "initial_forward_pass_backend": "eager",
            "generate_token_backend": "cudagraphs-manual",
            "stride_length": 4,
            "skip_when_1": True,
        }
        
        # Voice mapping
        self.voice_map = {
            "alloy": self.audio_prompt,
            "echo": self.audio_prompt,
            "fable": self.audio_prompt,
            "onyx": self.audio_prompt,
            "nova": self.audio_prompt,
            "shimmer": self.audio_prompt,
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
            logging.info("Generating new conditionals")
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
        """Save audio to buffer in specified format"""
        buffer = io.BytesIO()
        
        if format == 'pcm':
            buffer.write(self.to_pcm16(waveform))
        else:
            waveform_2d = waveform.unsqueeze(0)
            format_args = {"format": format}
            
            if format == 'opus':
                format_args = {"format": "ogg", "encoding": "opus"}
            elif format == 'aac':
                format_args = {"format": "mp4", "encoding": "aac"}
            
            torchaudio.save(buffer, waveform_2d, self.sample_rate, **format_args)
        
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
    
    def openai_tts(self):
        """OpenAI-compatible TTS endpoint"""
        payload = request.get_json(force=True)
        
        # Extract parameters
        text = payload.get("input", "")
        voice = payload.get("voice", "alloy")
        cfg_weight = payload.get("speed", 1.0)
        stream = payload.get("stream", self.config.stream)
        fmt = payload.get("response_format", "mp3" if not stream else "pcm").lower()
        
        if not text:
            return jsonify({"error": "Missing input in request"}), 400
        
        # Get audio prompt
        audio_prompt_path = self.config.voice_map.get(voice, self.config.audio_prompt)
        if not audio_prompt_path:
            return jsonify({"error": f"Voice '{voice}' not configured"}), 400
        
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
            await self.write_event(self.wyoming_info.event())
            return True
        
        # Streaming protocol
        if SynthesizeStart.is_type(event.type):
            synth_start = SynthesizeStart.from_event(event)
            self.streaming_voice = synth_start.voice.name if synth_start.voice else "alloy"
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
            await self._generate_and_stream_audio(full_text, self.streaming_voice or "alloy")
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
            voice_name = synthesize.voice.name if synthesize.voice else "alloy"
            logging.info(f"Wyoming TTS request (legacy): '{synthesize.text}' with voice '{voice_name}'")
            await self._generate_and_stream_audio(synthesize.text, voice_name)
            return True
        
        logging.warning(f"Unexpected event type: {event.type}")
        return True
    
    async def _generate_and_stream_audio(self, text: str, voice_name: str):
        """Generate audio with parallel prefetching and stream via Wyoming protocol"""
        loop = asyncio.get_event_loop()
        
        # Get voice audio prompt
        audio_prompt_path = self.config.voice_map.get(voice_name, self.config.audio_prompt)
        if not audio_prompt_path:
            logging.error("No audio prompt configured")
            return
        
        # Prepare voice conditionals in executor to not block
        await loop.run_in_executor(
            self.executor, self.model_manager.prepare_voice_conditionals, audio_prompt_path
        )
        
        # Generation kwargs - MUST match OpenAI endpoint exactly
        kwargs = {
            "language_id": self.config.language_id,
            "exaggeration": self.config.exaggeration,
            "temperature": self.config.temperature,
            "cfg_weight": 1.0,  # Match OpenAI default
            "min_p": self.config.min_p,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "t3_params": self.config.t3_params,
        }
        
        logging.info(f"Wyoming request: text='{text[:50]}...', voice='{voice_name}', cfg_weight=1.0")
        
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
                    for voice_name in self.config.voice_map.keys()
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
        description="OpenAI-compatible TTS server for Chatterbox with Wyoming protocol support."
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
    parser.add_argument("--audio-prompt", type=str, default=os.getenv("AUDIO_PROMPT"))
    
    # Generation arguments
    parser.add_argument("--exaggeration", type=float, default=float(os.getenv("EXAGGERATION", "0.5")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.8")))
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
    
    # Start Flask server in background thread
    flask_server = FlaskServer(model_manager, audio_processor, config)
    flask_thread = Thread(target=flask_server.run, daemon=True)
    flask_thread.start()
    
    logging.info(f"Flask server started on {config.host}:{config.openai_port}")
    logging.info(f"Starting Wyoming server on port {config.wyoming_port}...")
    
    # Run Wyoming server
    try:
        wyoming_server = WyomingServer(model_manager, audio_processor, config)
        asyncio.run(wyoming_server.run())
    except KeyboardInterrupt:
        logging.info("Shutting down servers...")


if __name__ == "__main__":
    main()
