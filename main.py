#!/usr/bin/env python3
"""
Wyoming Protocol wrapper for Chatterbox Turbo TTS
Provides voice cloning with automatic voice discovery from a directory.
"""
import argparse
import asyncio
import logging
import re
from pathlib import Path
from functools import lru_cache, partial

import torch
import torchaudio as ta
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_write_event
from wyoming.info import Describe

from chatterbox.tts_turbo import ChatterboxTurboTTS

_LOGGER = logging.getLogger(__name__)


class ChatterboxTurboEventHandler(AsyncEventHandler):
    """Wyoming event handler for Chatterbox Turbo TTS."""
    
    def __init__(self, wyoming_info: Info, model, voices: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.voices = voices
        self.sample_rate = model.sr  # Use model's native sample rate
        self.client_id = id(self)
    
    async def handle_event(self, event) -> bool:
        """Handle Wyoming protocol events."""
        # Send info on Describe event
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True
        
        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            await self._synthesize(synthesize)
            return True
        
        return True
    
    async def _synthesize(self, synthesize: Synthesize):
        """Generate speech and stream audio back to client."""
        text = synthesize.text
        voice_name = synthesize.voice.name if synthesize.voice else None
        
        _LOGGER.info(f"[Client {self.client_id}] Synthesizing: '{text[:50]}...' with voice: {voice_name}")
        
        # Get the audio prompt path for voice cloning
        audio_prompt_path = self.voices.get(voice_name) if voice_name else None
        
        if voice_name and not audio_prompt_path:
            _LOGGER.warning(f"Voice '{voice_name}' not found, using default")
        
        # Since Chatterbox Turbo doesn't support true streaming, we chunk the input
        # Split on sentence boundaries for more natural breaks
        chunks = self._split_text(text)
        
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
            
            _LOGGER.debug(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:30]}...'")
            
            # Generate audio for this chunk (runs in thread pool to avoid blocking)
            audio = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._generate_audio, 
                chunk, 
                audio_prompt_path
            )
            
            # Convert to bytes and send
            audio_bytes = (audio * 32767).numpy().astype("int16").tobytes()
            
            await self.write_event(
                AudioChunk(
                    rate=self.sample_rate,
                    width=2,
                    channels=1,
                    audio=audio_bytes
                ).event()
            )
        
        # Send audio stop event
        await self.write_event(AudioStop().event())
        _LOGGER.info(f"[Client {self.client_id}] Synthesis complete")
    
    def _generate_audio(self, text: str, audio_prompt_path: str = None) -> torch.Tensor:
        """Generate audio using Chatterbox Turbo (synchronous)."""
        # Generate with or without voice cloning
        if audio_prompt_path:
            wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
        else:
            # Use default voice if no prompt provided
            default_voice = next(iter(self.voices.values()), None)
            if default_voice:
                wav = self.model.generate(text, audio_prompt_path=default_voice)
            else:
                # Fallback: generate without voice cloning (if model supports it)
                wav = self.model.generate(text)
        
        return wav.squeeze()  # Remove batch dimension
    
    @staticmethod
    def _split_text(text: str, max_sentences: int = 3) -> list[str]:
        """
        Split text into chunks for pseudo-streaming.
        Splits on sentence boundaries (punctuation), grouping a few sentences per chunk.
        """
        # Split by sentence-ending punctuation (.!?) keeping the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have no sentences, return the original text
        if not sentences:
            return [text]
        
        # Group sentences into chunks of max_sentences
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = ' '.join(sentences[i:i + max_sentences])
            chunks.append(chunk)
        
        return chunks


def load_voices(voices_dir: Path) -> dict:
    """Scan voices directory and return mapping of voice name -> file path."""
    voices = {}
    
    if not voices_dir.exists():
        _LOGGER.warning(f"Voices directory does not exist: {voices_dir}")
        return voices
    
    # Find all .wav files in the voices directory
    for wav_file in voices_dir.glob("*.wav"):
        # Extract voice name from filename (e.g., "Jake.wav" -> "Jake")
        voice_name = wav_file.stem
        voices[voice_name] = str(wav_file)
        _LOGGER.info(f"Loaded voice: {voice_name} from {wav_file.name}")
    
    if not voices:
        _LOGGER.warning("No voice files found in voices directory")
    
    return voices


@lru_cache(maxsize=1)
def load_model(device: str = "cuda"):
    """Load Chatterbox Turbo model (cached)."""
    _LOGGER.info(f"Loading Chatterbox Turbo model on {device}...")
    
    # Patch watermarker if perth module is not available or broken
    try:
        import perth
        if perth.PerthImplicitWatermarker is None:
            raise ImportError("Perth watermarker not properly initialized")
    except (ImportError, AttributeError) as e:
        _LOGGER.warning(f"Perth watermarking not available ({e}), disabling watermarking")
        # Create a dummy watermarker class that mimics the Perth API
        class DummyWatermarker:
            def apply_watermark(self, audio, *args, **kwargs):
                """Pass-through watermark that returns audio unchanged."""
                return audio
            
            def __call__(self, audio, *args, **kwargs):
                return audio
        
        # Monkey-patch it into the module
        import sys
        if 'perth' not in sys.modules:
            import types
            sys.modules['perth'] = types.ModuleType('perth')
        sys.modules['perth'].PerthImplicitWatermarker = lambda: DummyWatermarker()
    
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    _LOGGER.info("Model loaded successfully")
    return model


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wyoming Chatterbox Turbo TTS Server")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200", help="Server URI")
    parser.add_argument("--voices-dir", type=Path, default=Path("./voices"), 
                       help="Directory containing voice reference .wav files")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to run model on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Load model
    model = load_model(args.device)
    
    # Create voices directory if it doesn't exist
    args.voices_dir.mkdir(parents=True, exist_ok=True)
    
    # Load voices
    voices = load_voices(args.voices_dir)
    
    # Create Wyoming info (using model's native sample rate)
    wyoming_info = create_info(args.voices_dir, model.sr)
    
    # Create server from URI
    _LOGGER.info(f"Starting Wyoming server on {args.uri}")
    server = AsyncServer.from_uri(args.uri)
    
    # Run server with handler factory
    await server.run(
        partial(
            ChatterboxTurboEventHandler,
            wyoming_info,
            model,
            voices
        )
    )


def create_info(voices_dir: Path, sample_rate: int) -> Info:
    """Create Wyoming Info with available voices."""
    voices = []
    
    # Discover voices from directory
    if voices_dir.exists():
        for wav_file in voices_dir.glob("*.wav"):
            voice_name = wav_file.stem
            voices.append(
                TtsVoice(
                    name=voice_name,
                    description=f"Cloned voice from {wav_file.name}",
                    attribution=Attribution(
                        name="Chatterbox Turbo",
                        url="https://github.com/resemble-ai/chatterbox"
                    ),
                    installed=True,
                    version="1.0",  # Required by Wyoming protocol
                    languages=["en"],  # Adjust based on your needs
                )
            )
    
    return Info(
        tts=[
            TtsProgram(
                name="chatterbox-turbo",
                description="Chatterbox Turbo TTS with voice cloning",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://github.com/resemble-ai/chatterbox"
                ),
                installed=True,
                version="1.0",  # Required by Wyoming protocol
                voices=voices,
            )
        ]
    )


if __name__ == "__main__":
    asyncio.run(main())
