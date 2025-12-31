"""Wyoming event handler for Chatterbox TTS."""
import asyncio
import logging
import time

import torch
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (Synthesize, SynthesizeChunk, SynthesizeStart,
                         SynthesizeStop, SynthesizeStopped)

from .utils import Colors, split_text

_LOGGER = logging.getLogger(__name__)


class ChatterboxEventHandler(AsyncEventHandler):
    """Wyoming event handler for Chatterbox TTS."""
    
    def __init__(self, wyoming_info: Info, model, voices: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.voices = voices
        self.sample_rate = model.sr  # Use model's native sample rate
        self.client_id = id(self)
        
        # Streaming state
        self.is_streaming = False  # Boolean flag to track streaming mode
        self._streaming_text = None
        self._streaming_voice = None
    
    async def handle_event(self, event) -> bool:
        """Handle Wyoming protocol events."""
        # Send info on Describe event
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        # Handle streaming TTS (new protocol)
        if SynthesizeStart.is_type(event.type):
            synth_start = SynthesizeStart.from_event(event)
            self.is_streaming = True  # Set streaming flag
            self._streaming_text = ""  # Initialize empty, text comes in chunks
            self._streaming_voice = synth_start.voice
            _LOGGER.info(f"{Colors.CYAN}[Client {self.client_id}] Mode: STREAMING{Colors.RESET}")
            return True

        if SynthesizeChunk.is_type(event.type):
            if self.is_streaming:
                synth_chunk = SynthesizeChunk.from_event(event)
                self._streaming_text += synth_chunk.text
                _LOGGER.debug(f"[Client {self.client_id}] SynthesizeChunk received: '{synth_chunk.text}'")
            return True

        if SynthesizeStop.is_type(event.type):
            if self.is_streaming:
                _LOGGER.info(f"[Client {self.client_id}] SynthesizeStop received, synthesizing accumulated text: '{self._streaming_text[:50]}...'")
                await self._synthesize_text(self._streaming_text, self._streaming_voice)

                # Send SynthesizeStopped event to signal synthesis is complete
                await self.write_event(SynthesizeStopped().event())

                # Reset streaming state
                self.is_streaming = False
                self._streaming_text = None
                self._streaming_voice = None
            return True

        # Handle non-streaming TTS (old protocol - for backwards compatibility)
        if Synthesize.is_type(event.type):
            if self.is_streaming:
                # Ignore since this is only sent for compatibility reasons
                _LOGGER.debug(f"[Client {self.client_id}] Ignoring Synthesize event (streaming mode active)")
                return True

            synthesize = Synthesize.from_event(event)
            _LOGGER.info(f"{Colors.MAGENTA}[Client {self.client_id}] Mode: STANDARD{Colors.RESET}")
            await self._synthesize_text(synthesize.text, synthesize.voice)
            return True

        return True

    async def _synthesize_text(self, text: str, voice_spec):
        """Generate and stream speech."""
        voice_name = voice_spec.name if voice_spec else None

        _LOGGER.info(f"[Client {self.client_id}] Synthesizing: '{text[:50]}...' with voice: {voice_name}")

        # Get the audio prompt path for voice cloning
        audio_prompt_path = self.voices.get(voice_name) if voice_name else None

        if voice_name and not audio_prompt_path:
            _LOGGER.warning(f"Voice '{voice_name}' not found, using default")

        # Split on sentence boundaries for streaming
        chunks = split_text(text)

        # Track timing
        synthesis_start = time.time()
        first_chunk_sent = False

        try:
            # Send audio start event
            await self.write_event(
                AudioStart(
                    rate=self.sample_rate,
                    width=2,  # 16-bit audio
                    channels=1
                ).event()
            )

            # Generate and stream each chunk
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                # Check if connection is still alive before expensive generation
                if self.writer.is_closing():
                    _LOGGER.info(f"[Client {self.client_id}] Connection closed, aborting synthesis at chunk {i+1}/{len(chunks)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return

                # Time this chunk's synthesis
                chunk_start = time.time()

                # Generate audio for this chunk (runs in thread pool to avoid blocking)
                audio = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self._generate_audio, 
                    chunk, 
                    audio_prompt_path
                )

                chunk_synth_time = time.time() - chunk_start

                # Check again after generation in case client disconnected during generation
                if self.writer.is_closing():
                    _LOGGER.info(f"[Client {self.client_id}] Connection closed during generation of chunk {i+1}/{len(chunks)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return

                # Convert to bytes and send
                audio_bytes = (audio * 32767).numpy().astype("int16").tobytes()

                # Send audio chunk WITHOUT TIMESTAMP for immediate streaming
                await self.write_event(
                    AudioChunk(
                        rate=self.sample_rate,
                        width=2,
                        channels=1,
                        audio=audio_bytes
                    ).event()
                )

                # Log first chunk sent
                if not first_chunk_sent:
                    ttfa = time.time() - synthesis_start
                    _LOGGER.info(f"{Colors.GREEN}[Client {self.client_id}] First chunk sent (TTFA: {ttfa:.2f}s){Colors.RESET}")
                    first_chunk_sent = True

                # Log chunk timing
                _LOGGER.info(f"{Colors.YELLOW}[Client {self.client_id}] Chunk {i+1}/{len(chunks)}: {chunk_synth_time:.2f}s{Colors.RESET}")
            
            # Send audio stop event WITHOUT TIMESTAMP
            await self.write_event(AudioStop().event())

            # Log completion with total time
            total_time = time.time() - synthesis_start
            _LOGGER.info(f"{Colors.GREEN}[Client {self.client_id}] Complete - Total time: {total_time:.2f}s{Colors.RESET}")
        
        except ConnectionResetError:
            _LOGGER.info(f"[Client {self.client_id}] Client disconnected during synthesis (normal - user likely interrupted)")
            # Clean up CUDA cache since we generated partial audio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BrokenPipeError:
            _LOGGER.info(f"[Client {self.client_id}] Connection broken during synthesis (normal - client closed connection)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            _LOGGER.error(f"[Client {self.client_id}] Unexpected error during synthesis: {e}", exc_info=True)
            # Try to send audio stop if connection is still alive
            try:
                if not self.writer.is_closing():
                    await self.write_event(AudioStop().event())
            except Exception:
                pass  # Connection already dead, ignore
            # Clean up CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _generate_audio(self, text: str, audio_prompt_path: str = None) -> torch.Tensor:
        """Generate audio using Chatterbox (synchronous)."""
        # Use no_grad to prevent memory accumulation from autograd
        with torch.no_grad():
            # Get backend and generation params from model (set during initialization)
            backend = getattr(self.model, '_wyoming_backend', 'cudagraphs-manual')
            gen_params = getattr(self.model, '_wyoming_gen_params', {})

            t3_params = {
                "benchmark_t3": True,
                "generate_token_backend": backend,
                "skip_when_1": True,  # Skip Top P when it's 1.0
                # Add generation parameters
                "temperature": gen_params.get("temperature", 0.8),
                "top_p": gen_params.get("top_p", 1.0),
                "min_p": gen_params.get("min_p", 0.0),
                "max_new_tokens": gen_params.get("max_new_tokens", 4096),
            }

            # Add seed if specified (non-zero)
            if gen_params.get("seed"):
                t3_params["seed"] = gen_params["seed"]

            # Flow matching parameters
            s3gen_params = {
                "n_timesteps": gen_params.get("n_timesteps", 10),
                "cfg_scale": gen_params.get("flow_cfg_scale", 1.0),
            }

            # Generate with or without voice cloning
            if audio_prompt_path:
                wav = self.model.generate(
                    text, 
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=gen_params.get("exaggeration", 0.5),
                    cfg_weight=gen_params.get("cfg_weight", 0.5),
                    t3_params=t3_params,
                    s3gen_params=s3gen_params,
                )
            else:
                # Use default voice if no prompt provided
                default_voice = next(iter(self.voices.values()), None)
                if default_voice:
                    wav = self.model.generate(
                        text, 
                        audio_prompt_path=default_voice,
                        exaggeration=gen_params.get("exaggeration", 0.5),
                        cfg_weight=gen_params.get("cfg_weight", 0.5),
                        t3_params=t3_params,
                        s3gen_params=s3gen_params,
                    )
                else:
                    # Fallback: generate without voice cloning
                    wav = self.model.generate(
                        text,
                        exaggeration=gen_params.get("exaggeration", 0.5),
                        cfg_weight=gen_params.get("cfg_weight", 0.5),
                        t3_params=t3_params,
                        s3gen_params=s3gen_params,
                    )

            # Move to CPU immediately to free VRAM - numpy conversion happens on CPU anyway
            result = wav.squeeze().cpu()

        # Synchronize to ensure all CUDA operations are complete before cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return result
