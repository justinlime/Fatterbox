"""Fatterbox - Wyoming Protocol wrapper for Chatterbox TTS."""

__version__ = "1.0.0"

from .handler import ChatterboxEventHandler
from .model import load_model
from .voices import create_wyoming_info, load_voices

__all__ = [
    "ChatterboxEventHandler",
    "load_model",
    "load_voices",
    "create_wyoming_info",
]
