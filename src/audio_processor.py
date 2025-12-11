import io
from typing import List

import numpy as np
import pysbd
import torch
from pydub import AudioSegment


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
