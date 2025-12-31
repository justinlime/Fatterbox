"""Voice discovery and Wyoming info generation."""
import logging
from pathlib import Path

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

_LOGGER = logging.getLogger(__name__)


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


def create_wyoming_info(voices_dir: Path, sample_rate: int) -> Info:
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
                        name="Chatterbox",
                        url="https://github.com/resemble-ai/chatterbox"
                    ),
                    installed=True,
                    version="1.0",
                    languages=["en"],
                )
            )
    
    return Info(
        tts=[
            TtsProgram(
                name="chatterbox",
                description="Chatterbox TTS with voice cloning",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://github.com/resemble-ai/chatterbox"
                ),
                installed=True,
                version="1.0",
                voices=voices,
                supports_synthesize_streaming=True,  # Enable streaming support!
            )
        ]
    )
