import argparse
import gc
import io
import os
import random
import numpy as np
import pysbd
import torch
from flask import Flask, request, send_file, jsonify, render_template, render_template_string, Response
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import torchaudio
from threading import Lock
import asyncio
import logging
from functools import partial

# Wyoming protocol imports
from wyoming.info import Info, TtsProgram, TtsVoice, Attribution
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

# Set up Flask app
app = Flask(__name__)

# Global lock for model inference to prevent concurrent CUDA graph issues
inference_lock = Lock()

parser = argparse.ArgumentParser(description="OpenAI-compatible TTS server for Chatterbox with Wyoming protocol support.")

# Server arguments
parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host for the server.")
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5002")), help="Port for the Flask server.")
parser.add_argument("--wyoming-port", type=int, default=int(os.getenv("WYOMING_PORT", "10200")), help="Port for the Wyoming protocol server.")
parser.add_argument("--debug", action="store_true", default=os.getenv("DEBUG", "").lower() == "true", help="Run the server in debug mode.")
parser.add_argument("--device", type=str, default=os.getenv("DEVICE", "cpu"), help="Device to run server on. Options: cpu, cuda, cuda:0, cuda:1, mps")
parser.add_argument("--low_vram", action=argparse.BooleanOptionalAction, default=os.getenv("LOW_VRAM", "").lower() == "true", help="Whether to unload model to cpu when not generating.")
parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=os.getenv("STREAM", "").lower() == "true", help="Enable audio streaming sentence by sentence.")
parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_PATH"), help="Path to a local directory containing model checkpoints.")
parser.add_argument("--language_id", type=str, default=os.getenv("LANGUAGE_ID", "en"), help="Two letter language code: Arabic (ar), Danish (da), German (de), Greek (el), English (en), Spanish (es), Finnish (fi), French (fr), Hebrew (he), Hindi (hi), Italian (it), Japanese (ja), Korean (ko), Malay (ms), Dutch (nl), Norwegian (no), Polish (pl), Portuguese (pt), Russian (ru), Swedish (sv), Swahili (sw), Turkish (tr), Chinese (zh)")
parser.add_argument("--audio_prompt", type=str, default=os.getenv("AUDIO_PROMPT"), help="Default audio prompt path for voice cloning.")

# Chatterbox generation arguments with reasonable defaults
parser.add_argument("--exaggeration", type=float, default=float(os.getenv("EXAGGERATION", "0.5")), help="Exaggeration level (0.5 is neutral).")
parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.8")), help="Sampling temperature.")
parser.add_argument("--min-p", type=float, default=float(os.getenv("MIN_P", "0.05")), help="min_p for nucleus sampling.")
parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "1.0")), help="top_p for nucleus sampling (1.0 disables it).")
parser.add_argument("--repetition-penalty", type=float, default=float(os.getenv("REPETITION_PENALTY", "1.2")), help="Repetition penalty.")

# T3 optimization argument
parser.add_argument("--t3-dtype", type=str, default=os.getenv("T3_DTYPE", "bfloat16"), choices=["float32", "float16", "bfloat16"], help="Data type for T3 model (bfloat16 recommended for most modern GPUs).")

args = parser.parse_args()
segmenter = pysbd.Segmenter(language="en", clean=True)
current_audio_prompt_path = None
cached_conds = {}

# OpenAI voice name mapping to audio prompt paths
VOICE_MAP = {
    "alloy": args.audio_prompt,
    "echo": args.audio_prompt,
    "fable": args.audio_prompt,
    "onyx": args.audio_prompt,
    "nova": args.audio_prompt,
    "shimmer": args.audio_prompt,
}

DEVICE = args.device
if "cuda" in DEVICE:
    if not torch.cuda.is_available():
        DEVICE = "cpu"
if "mps" in DEVICE:
    if not torch.backends.mps.is_available():
        DEVICE = "cpu"
        
LANGUAGE = args.language_id if args.language_id else "en"

def load_chatterbox_tts_model(device):
    tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return tts_model

def t3_to(model: ChatterboxTTS, dtype):
    """Convert T3 model to specified dtype for optimization"""
    model.t3.to(dtype=dtype)
    model.conds.t3.to(dtype=dtype)
    torch.cuda.empty_cache()
    return model

# Load the Chatterbox model
print(f"Loading Chatterbox TTS...")
chatterbox_model = load_chatterbox_tts_model(DEVICE)

# Apply T3 dtype optimization if on CUDA
if "cuda" in DEVICE and args.t3_dtype != "float32":
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    target_dtype = dtype_map[args.t3_dtype]
    print(f"Optimizing T3 model with dtype: {args.t3_dtype}")
    t3_to(chatterbox_model, target_dtype)

CURRENT_DEVICE = DEVICE

generation_count = 0

# Hardcoded T3 optimization params for best performance
T3_PARAMS = {
    "initial_forward_pass_backend": "eager",
    "generate_token_backend": "cudagraphs-manual",
    "stride_length": 4,
    "skip_when_1": True,
}

print(f"T3 optimization params: {T3_PARAMS}")

# Warmup generation if using CUDA
if "cuda" in DEVICE:
    print("Running warmup generation...")
    try:
        # First warmup to initialize CUDA graphs
        _ = chatterbox_model.generate(
            "Warmup generation for CUDA graph initialization.",
            t3_params=T3_PARAMS
        )
        # Second generation at full speed
        _ = chatterbox_model.generate(
            "Second warmup for full speed.",
            t3_params=T3_PARAMS
        )
        print("Warmup complete!")
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}")

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def handle_vram_change(desired_device: str):
    global chatterbox_model, CURRENT_DEVICE
    if torch.cuda.is_available():
        if "cuda" in desired_device:
            if "cuda" not in CURRENT_DEVICE:
                if chatterbox_model:
                    del chatterbox_model
                gc.collect()
                chatterbox_model = load_chatterbox_tts_model(desired_device)
                
                # Re-apply T3 optimization after loading
                if args.t3_dtype != "float32":
                    dtype_map = {
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                    }
                    t3_to(chatterbox_model, dtype_map[args.t3_dtype])
                
                CURRENT_DEVICE = desired_device
                print(f"Switched ChatterboxTTS model to {desired_device}.")

        elif "cpu" in desired_device:
            if "cpu" not in CURRENT_DEVICE:
                del chatterbox_model
                torch.cuda.empty_cache()
                gc.collect()
                chatterbox_model = None
                CURRENT_DEVICE = desired_device
                print("Unloaded ChatterboxTTS model")

if args.low_vram and "cuda" in DEVICE:
    handle_vram_change("cpu")

def _cleanup():
    """Handles VRAM and garbage collection after a generation task."""
    if args.low_vram and "cuda" in DEVICE:
        handle_vram_change("cpu")

    # Even if low vram is off, clear cache after 5 generations to prevent VRAM usage from infinitely increasing
    global generation_count
    generation_count += 1
    if generation_count >= 5 and not args.low_vram and "cuda" in DEVICE:
        generation_count = 0
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared after 5 generations.")

def split_sentences(input_text: str) -> list:
    if len(input_text) <= 150:
        return [input_text]
    return segmenter.segment(input_text)

def get_voice_conds_for_audio_prompt(audio_prompt_path):
    global chatterbox_model
    # Check if voice conds already exist
    voice_conds = cached_conds.get(str(audio_prompt_path))
    # If not, add them to our cache
    if not voice_conds:
        print("Cached conditionals not found, generating new conditionals.")
        chatterbox_model.prepare_conditionals(audio_prompt_path)
        cached_conds[str(audio_prompt_path)] = chatterbox_model.conds
    else:
        print("Cached conditionals found; reusing.")
        chatterbox_model.conds = voice_conds

@app.route("/")
def index():
    return render_template(
        "index.html",
    )

@app.route("/v1/audio/speech", methods=["POST"])
def openai_tts():
    if args.low_vram and "cuda" in DEVICE:
        handle_vram_change(DEVICE)

    payload = request.get_json(force=True)
    
    text = payload.get("input", "")
    voice = payload.get("voice", "alloy")
    model = payload.get("model", "tts-1")
    cfg_weight = payload.get("speed", 0.5)
    stream = payload.get("stream", args.stream)

    if not text:
        return jsonify({"error":"Missing input in request"}), 400

    # Map OpenAI voice names to audio prompt paths
    audio_prompt_path = VOICE_MAP.get(voice, args.audio_prompt)
    
    # If no audio_prompt configured and voice not in map, return error
    if audio_prompt_path is None:
        return jsonify({"error": f"Voice '{voice}' not configured. Please set --audio_prompt or use a mapped voice."}), 400

    # Prepare conditionals for new wav if needed, otherwise recycle conditionals
    global current_audio_prompt_path
    if audio_prompt_path != current_audio_prompt_path or args.low_vram:
        get_voice_conds_for_audio_prompt(audio_prompt_path)
        current_audio_prompt_path = audio_prompt_path
    
    kwargs = dict(
        exaggeration=args.exaggeration,
        temperature=args.temperature,
        cfg_weight=cfg_weight,
        min_p=args.min_p,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        t3_params=T3_PARAMS,
    )
    # Add language_id only if not English
    if LANGUAGE != "en":
        kwargs["language_id"] = LANGUAGE
    
    # Streaming
    if stream:
        fmt = payload.get("response_format", "pcm").lower()
        
        # Only PCM and MP3 are supported for chunked streaming
        if fmt not in ['pcm', 'mp3']:
            return jsonify({"error": f"Streaming is only supported for 'pcm' and 'mp3' formats. Requested: {fmt}"}), 400
        
        print(f"Streaming Request: text='{text}', voice='{voice}', model='{model}', speed='{cfg_weight}', format='{fmt}'")

        def generate_stream():
            try:
                sentences = split_sentences(text)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    print(f"Streaming sentence: {sentence}")
                    # Always use same manual seed for consistency in generation
                    set_seed(12345)
                    
                    # Use lock to prevent concurrent CUDA graph access
                    with inference_lock:
                        wav_tensor = chatterbox_model.generate(
                            sentence,
                            language_id="en",
                            **kwargs
                        )
                    waveform_cpu = wav_tensor.squeeze(0).cpu()

                    if fmt == 'pcm':
                        waveform_int16 = (waveform_cpu * 32767).to(torch.int16)
                        yield waveform_int16.numpy().tobytes()
                    elif fmt == 'mp3':
                        buffer = io.BytesIO()
                        waveform_2d = waveform_cpu.unsqueeze(0)
                        torchaudio.save(buffer, waveform_2d, chatterbox_model.sr, format="mp3")
                        yield buffer.getvalue()
            finally:
                _cleanup()
                print("Finished streaming response.")

        mimetype = "audio/L16" if fmt == 'pcm' else "audio/mpeg"
        return Response(generate_stream(), mimetype=mimetype)

    # Non-Streaming
    else:
        fmt = payload.get("response_format", "mp3").lower()
        print(f"Request: text='{text}', voice='{voice}', model='{model}', speed='{cfg_weight}', format='{fmt}'")

        sentences = split_sentences(text)
        audio_chunks = []

        for sentence in sentences:
            # Always use same manual seed for consistency in generation
            set_seed(12345) 

            # Use lock to prevent concurrent CUDA graph access
            with inference_lock:
                # Call generate with unpacked kwargs
                wav_tensor = chatterbox_model.generate(
                    sentence,
                    language_id="en",
                    **kwargs
                )
            audio_chunks.append(wav_tensor)
        
        final_audio = torch.cat(audio_chunks, dim=-1) if len(audio_chunks) > 1 else audio_chunks[0]
        waveform_cpu = final_audio.squeeze(0).cpu()
        
        mimetypes = {
            "wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg",
            "aac": "audio/aac", "flac": "audio/flac", "pcm": "audio/L16",
        }

        if fmt not in mimetypes:
            return jsonify({"error": f"Unsupported format: {fmt}"}), 400

        mimetype = mimetypes[fmt]
        buffer = io.BytesIO()

        if fmt == 'pcm':
            waveform_int16 = (waveform_cpu * 32767).to(torch.int16)
            buffer.write(waveform_int16.numpy().tobytes())
        else:
            waveform_2d = waveform_cpu.unsqueeze(0)
            format_args = {"format": fmt}
            if fmt == 'opus':
                format_args = {"format": "ogg", "encoding": "opus"}
            elif fmt == 'aac':
                format_args = {"format": "mp4", "encoding": "aac"}
            torchaudio.save(buffer, waveform_2d, chatterbox_model.sr, **format_args)

        buffer.seek(0)
        _cleanup()
        return send_file(buffer, mimetype=mimetype)

@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    """
    Return a list of available models, for OpenAI client compatibility.
    """
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1677610600,
                "owned_by": "openai"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1677610600,
                "owned_by": "openai"
            }
        ]
    })


# ============================================================================
# Wyoming Protocol Implementation
# ============================================================================

class ChatterboxWyomingEventHandler(AsyncEventHandler):
    """Handles Wyoming protocol events for TTS"""
    
    def __init__(self, wyoming_info: Info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.sample_rate = 22050  # Chatterbox default sample rate
        self.sample_width = 2  # 16-bit audio
        self.channels = 1  # Mono
        
        # Streaming state
        self.streaming_text_buffer = []
        self.streaming_voice = None
        self.is_streaming = False
        
    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming protocol events"""
        # Handle describe events - Home Assistant queries server capabilities
        if event.type == "describe":
            logging.info("Received describe event, sending info")
            await self.write_event(self.wyoming_info.event())
            return True
        
        # New streaming TTS protocol
        if SynthesizeStart.is_type(event.type):
            synth_start = SynthesizeStart.from_event(event)
            voice_name = synth_start.voice.name if synth_start.voice else "alloy"
            logging.info(f"Wyoming TTS streaming started with voice '{voice_name}'")
            
            # Reset streaming state
            self.streaming_text_buffer = []
            self.streaming_voice = voice_name
            self.is_streaming = True
            return True
        
        if SynthesizeChunk.is_type(event.type):
            synth_chunk = SynthesizeChunk.from_event(event)
            logging.info(f"Received text chunk: '{synth_chunk.text}'")
            self.streaming_text_buffer.append(synth_chunk.text)
            return True
        
        if SynthesizeStop.is_type(event.type):
            logging.info("Wyoming TTS streaming stopped, generating audio")
            
            # Combine all text chunks
            full_text = "".join(self.streaming_text_buffer)
            voice_name = self.streaming_voice or "alloy"
            
            # Generate and stream audio
            await self._generate_and_stream_audio(full_text, voice_name)
            
            # Send stopped event
            await self.write_event(SynthesizeStopped().event())
            
            # Reset state
            self.streaming_text_buffer = []
            self.streaming_voice = None
            self.is_streaming = False
            
            return True
        
        # Legacy non-streaming protocol (for backward compatibility)
        # Only handle this if we're not in a streaming session
        if Synthesize.is_type(event.type):
            if self.is_streaming:
                logging.warning("Ignoring legacy Synthesize event during streaming session")
                return True
                
            synthesize = Synthesize.from_event(event)
            
            # Extract voice name from SynthesizeVoice object
            voice_name = synthesize.voice.name if synthesize.voice else "alloy"
            logging.info(f"Wyoming TTS request (legacy): '{synthesize.text}' with voice '{voice_name}'")
            
            # Generate and stream audio
            await self._generate_and_stream_audio(synthesize.text, voice_name)
            
            return True
        
        logging.warning(f"Unexpected event type: {event.type}")
        return True
    
    async def _generate_and_stream_audio(self, text: str, voice_name: str):
        """Generate audio and stream it back via Wyoming protocol"""
        # Load model to device if needed
        if args.low_vram and "cuda" in DEVICE:
            await asyncio.get_event_loop().run_in_executor(
                None, handle_vram_change, DEVICE
            )
        
        # Get voice audio prompt
        audio_prompt_path = VOICE_MAP.get(voice_name, args.audio_prompt)
        
        if audio_prompt_path is None:
            logging.error("No audio prompt configured")
            return
        
        # Prepare conditionals
        global current_audio_prompt_path
        if audio_prompt_path != current_audio_prompt_path or args.low_vram:
            await asyncio.get_event_loop().run_in_executor(
                None, get_voice_conds_for_audio_prompt, audio_prompt_path
            )
            current_audio_prompt_path = audio_prompt_path
        
        # Generate audio
        kwargs = dict(
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            cfg_weight=0.5,
            min_p=args.min_p,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            t3_params=T3_PARAMS,
        )
        
        if LANGUAGE != "en":
            kwargs["language_id"] = LANGUAGE
        
        # Send audio start event
        await self.write_event(
            AudioStart(
                rate=self.sample_rate,
                width=self.sample_width,
                channels=self.channels
            ).event()
        )
        
        # Split into sentences for streaming
        sentences = split_sentences(text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            logging.info(f"Generating sentence: {sentence}")
            
            # Generate audio in executor to not block event loop
            def generate_audio():
                set_seed(12345)
                with inference_lock:
                    return chatterbox_model.generate(
                        sentence,
                        language_id="en",
                        **kwargs
                    )
            
            wav_tensor = await asyncio.get_event_loop().run_in_executor(
                None, generate_audio
            )
            
            # Convert to PCM16
            waveform_cpu = wav_tensor.squeeze(0).cpu()
            waveform_int16 = (waveform_cpu * 32767).to(torch.int16)
            audio_bytes = waveform_int16.numpy().tobytes()
            
            # Send audio in chunks (Wyoming typically uses chunks)
            chunk_size = 1024
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                await self.write_event(
                    AudioChunk(
                        rate=self.sample_rate,
                        width=self.sample_width,
                        channels=self.channels,
                        audio=chunk
                    ).event()
                )
        
        # Send audio stop event
        await self.write_event(AudioStop().event())
        
        # Cleanup
        await asyncio.get_event_loop().run_in_executor(None, _cleanup)
        logging.info("Finished Wyoming TTS generation")


async def wyoming_server_main():
    """Main function for Wyoming server"""
    # Create Wyoming info with available voices
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="chatterbox",
                version="1.0.0",
                description="Chatterbox TTS with voice cloning",
                attribution=Attribution(
                    name="Chatterbox",
                    url="https://github.com/chatterbox-tts/chatterbox"
                ),
                installed=True,
                supports_synthesize_streaming=True,  # Enable new streaming protocol
                voices=[
                    TtsVoice(
                        name=voice_name,
                        version="1.0.0",
                        description=f"Chatterbox voice: {voice_name}",
                        attribution=Attribution(name="Chatterbox", url=""),
                        installed=True,
                        languages=[LANGUAGE]
                    )
                    for voice_name in VOICE_MAP.keys()
                ]
            )
        ]
    )

    logging.info(f"Wyoming server starting on port {args.wyoming_port}")

    # Run server with properly structured handler factory
    await AsyncServer.from_uri(f"tcp://0.0.0.0:{args.wyoming_port}").run(
        partial(ChatterboxWyomingEventHandler, wyoming_info)
    )


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.debug else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start Flask server in a separate thread
    from threading import Thread
    
    def run_flask():
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print(f"Flask server started on {args.host}:{args.port}")
    print(f"Starting Wyoming server on port {args.wyoming_port}...")
    
    # Run Wyoming server
    try:
        asyncio.run(wyoming_server_main())
    except KeyboardInterrupt:
        print("\nShutting down servers...")

if __name__ == "__main__":
    main()
