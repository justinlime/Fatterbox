#!/usr/bin/env python3
"""
Wyoming Protocol wrapper for Chatterbox TTS
Entry point and CLI setup.
"""
import argparse
import asyncio
import logging
import os
from functools import partial
from pathlib import Path

from wyoming.server import AsyncServer

from .handler import ChatterboxEventHandler
from .model import load_model
from .voices import load_voices, create_wyoming_info

_LOGGER = logging.getLogger(__name__)


def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable with fallback to default."""
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def get_env_int(key: str, default: int) -> int:
    """Get int from environment variable with fallback to default."""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


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
    
    # TTS Generation Parameters
    parser.add_argument("--exaggeration", type=float, 
                       default=get_env_float("FATTERBOX_EXAGGERATION", 0.5),
                       help="Emotional expressiveness (0.0=flat, 0.5=normal, 2.0=exaggerated). Default: 0.5")
    parser.add_argument("--cfg-weight", type=float, 
                       default=get_env_float("FATTERBOX_CFG_WEIGHT", 0.5),
                       help="Voice adherence/pacing control (0.0-1.0, lower=expressive, higher=literal). Default: 0.5")
    parser.add_argument("--temperature", type=float, 
                       default=get_env_float("FATTERBOX_TEMPERATURE", 0.8),
                       help="Randomness/creativity (0.05-5.0, higher=more varied). Default: 0.8")
    parser.add_argument("--seed", type=int, 
                       default=get_env_int("FATTERBOX_SEED", 0),
                       help="Random seed for reproducibility (0=random). Default: 0")
    parser.add_argument("--top-p", type=float, 
                       default=get_env_float("FATTERBOX_TOP_P", 1.0),
                       help="Nucleus sampling top-p (0.0-1.0). Default: 1.0")
    parser.add_argument("--min-p", type=float, 
                       default=get_env_float("FATTERBOX_MIN_P", 0.0),
                       help="Nucleus sampling min-p (0.0-1.0). Default: 0.0")
    parser.add_argument("--max-new-tokens", type=int, 
                       default=get_env_int("FATTERBOX_MAX_NEW_TOKENS", 4096),
                       help="Max audio tokens (25≈1sec, max 4096). Default: 4096")
    parser.add_argument("--n-timesteps", type=int, 
                       default=get_env_int("FATTERBOX_N_TIMESTEPS", 10),
                       help="Diffusion steps for flow matching. Default: 10")
    parser.add_argument("--flow-cfg-scale", type=float, 
                       default=get_env_float("FATTERBOX_FLOW_CFG_SCALE", 1.0),
                       help="CFG scale for mel decoder. Default: 1.0")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Log generation parameters
    _LOGGER.info(f"Generation parameters:")
    _LOGGER.info(f"  exaggeration: {args.exaggeration}")
    _LOGGER.info(f"  cfg_weight: {args.cfg_weight}")
    _LOGGER.info(f"  temperature: {args.temperature}")
    _LOGGER.info(f"  seed: {args.seed}")
    _LOGGER.info(f"  top_p: {args.top_p}")
    _LOGGER.info(f"  min_p: {args.min_p}")
    
    # Load model
    model = load_model(args.device, args.dtype)
    
    # Store backend preference and generation params on model
    model._wyoming_backend = args.backend
    model._wyoming_gen_params = {
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "temperature": args.temperature,
        "seed": args.seed if args.seed > 0 else None,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "max_new_tokens": args.max_new_tokens,
        "n_timesteps": args.n_timesteps,
        "flow_cfg_scale": args.flow_cfg_scale,
    }
    
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
