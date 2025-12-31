#!/usr/bin/env python3
"""
Wyoming Protocol wrapper for Chatterbox TTS
Entry point and CLI setup.
"""
import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path

from wyoming.server import AsyncServer

from .handler import ChatterboxEventHandler
from .model import load_model
from .voices import load_voices, create_wyoming_info

_LOGGER = logging.getLogger(__name__)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wyoming Chatterbox TTS Server")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200", help="Server URI")
    parser.add_argument("--voices-dir", type=Path, default=Path("./voices"), 
                       help="Directory containing voice reference .wav files")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to run model on")
    parser.add_argument("--dtype", default="float32",
                       choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
                       help="Model precision (bf16 recommended for RTX 30xx/40xx)")
    parser.add_argument("--backend", default="cudagraphs-manual",
                       choices=["cudagraphs-manual", "cudagraphs", "eager", "inductor"],
                       help="Generation backend (cudagraphs-manual is fastest)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Load model
    model = load_model(args.device, args.dtype)
    
    # Store backend preference on model for use in generation
    model._wyoming_backend = args.backend
    
    # Create voices directory if it doesn't exist
    args.voices_dir.mkdir(parents=True, exist_ok=True)
    
    # Load voices
    voices = load_voices(args.voices_dir)
    
    # Create Wyoming info (using model's native sample rate)
    wyoming_info = create_wyoming_info(args.voices_dir, model.sr)
    
    # Create server from URI
    _LOGGER.info(f"Starting Wyoming server on {args.uri}")
    server = AsyncServer.from_uri(args.uri)
    
    # Run server with handler factory
    await server.run(
        partial(
            ChatterboxEventHandler,
            wyoming_info,
            model,
            voices
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
