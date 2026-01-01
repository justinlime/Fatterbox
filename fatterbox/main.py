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

import uvicorn
from wyoming.server import AsyncServer

from .openapi import create_api
from .handler import ChatterboxEventHandler
from .model import load_model
from .voices import load_voices, create_wyoming_info
from .utils import get_env_str, get_env_float, get_env_int, get_env_bool

_LOGGER = logging.getLogger(__name__)


async def run_wyoming_server(args, model, voices, wyoming_info):
    """Run Wyoming protocol server."""
    _LOGGER.info(f"Starting Wyoming server on tcp://{args.wyoming_host}:{args.wyoming_port}")
    uri = f"tcp://{args.wyoming_host}:{args.wyoming_port}"
    server = AsyncServer.from_uri(uri)
    
    await server.run(
        partial(
            ChatterboxEventHandler,
            wyoming_info,
            model,
            voices
        )
    )


async def run_fastapi_server(args, model, voices):
    """Run FastAPI REST API server."""
    app = create_api(model, voices, args.voices_dir)
    
    host = args.openapi_host
    port = args.openapi_port
    
    _LOGGER.info(f"Starting OpenAPI server on {host}:{port}")
    _LOGGER.info(f"OpenAPI endpoints:")
    _LOGGER.info(f"  - POST http://{host}:{port}/v1/audio/speech (streaming)")
    _LOGGER.info(f"  - GET  http://{host}:{port}/v1/voices")
    _LOGGER.info(f"  - GET  http://{host}:{port}/v1/info")
    _LOGGER.info(f"API documentation: http://{host}:{port}/docs")
    
    # Run uvicorn server
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info" if not args.debug else "debug"
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """Main entry point - runs both Wyoming and OpenAPI servers by default."""
    parser = argparse.ArgumentParser(description="Fatterbox: Wyoming + OpenAPI Chatterbox TTS Server")
    
    # Wyoming server options
    parser.add_argument("--wyoming-host", 
                       default=get_env_str("FATTERBOX_WYOMING_HOST", "0.0.0.0"),
                       help="Wyoming server host (default: 0.0.0.0, env: FATTERBOX_WYOMING_HOST)")
    parser.add_argument("--wyoming-port", type=int, 
                       default=get_env_int("FATTERBOX_WYOMING_PORT", 10200),
                       help="Wyoming server port (default: 10200, env: FATTERBOX_WYOMING_PORT)")
    
    # OpenAPI server options
    parser.add_argument("--openapi-host", 
                       default=get_env_str("FATTERBOX_OPENAPI_HOST", "0.0.0.0"),
                       help="OpenAPI server host (default: 0.0.0.0, env: FATTERBOX_OPENAPI_HOST)")
    parser.add_argument("--openapi-port", type=int, 
                       default=get_env_int("FATTERBOX_OPENAPI_PORT", 8000),
                       help="OpenAPI server port (default: 8000, env: FATTERBOX_OPENAPI_PORT)")
    
    # Common options
    parser.add_argument("--voices-dir", type=Path, 
                       default=Path(get_env_str("FATTERBOX_VOICES_DIR", "./voices")),
                       help="Directory containing voice reference .wav files (default: ./voices, env: FATTERBOX_VOICES_DIR)")
    parser.add_argument("--device", 
                       default=get_env_str("FATTERBOX_DEVICE", "cuda"),
                       choices=["cuda", "cpu"],
                       help="Device to run model on (default: cuda, env: FATTERBOX_DEVICE)")
    parser.add_argument("--dtype", 
                       default=get_env_str("FATTERBOX_DTYPE", "float32"),
                       choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
                       help="Model precision (bf16 recommended for RTX 30xx/40xx) (default: float32, env: FATTERBOX_DTYPE)")
    parser.add_argument("--backend", 
                       default=get_env_str("FATTERBOX_BACKEND", "cudagraphs-manual"),
                       choices=["cudagraphs-manual", "cudagraphs", "eager", "inductor"],
                       help="Generation backend (cudagraphs-manual is fastest) (default: cudagraphs-manual, env: FATTERBOX_BACKEND)")
    
    # TTS Generation Parameters
    parser.add_argument("--exaggeration", type=float, 
                       default=get_env_float("FATTERBOX_EXAGGERATION", 0.5),
                       help="Emotional expressiveness (0.0=flat, 0.5=normal, 2.0=exaggerated) (default: 0.5, env: FATTERBOX_EXAGGERATION)")
    parser.add_argument("--cfg-weight", type=float, 
                       default=get_env_float("FATTERBOX_CFG_WEIGHT", 0.5),
                       help="Voice adherence/pacing control (0.0-1.0, lower=expressive, higher=literal) (default: 0.5, env: FATTERBOX_CFG_WEIGHT)")
    parser.add_argument("--temperature", type=float, 
                       default=get_env_float("FATTERBOX_TEMPERATURE", 0.8),
                       help="Randomness/creativity (0.05-5.0, higher=more varied) (default: 0.8, env: FATTERBOX_TEMPERATURE)")
    parser.add_argument("--seed", type=int, 
                       default=get_env_int("FATTERBOX_SEED", 0),
                       help="Random seed for reproducibility (0=random) (default: 0, env: FATTERBOX_SEED)")
    parser.add_argument("--top-p", type=float, 
                       default=get_env_float("FATTERBOX_TOP_P", 1.0),
                       help="Nucleus sampling top-p (0.0-1.0) (default: 1.0, env: FATTERBOX_TOP_P)")
    parser.add_argument("--min-p", type=float, 
                       default=get_env_float("FATTERBOX_MIN_P", 0.0),
                       help="Nucleus sampling min-p (0.0-1.0) (default: 0.0, env: FATTERBOX_MIN_P)")
    parser.add_argument("--max-new-tokens", type=int, 
                       default=get_env_int("FATTERBOX_MAX_NEW_TOKENS", 4096),
                       help="Max audio tokens (25≈1sec, max 4096) (default: 4096, env: FATTERBOX_MAX_NEW_TOKENS)")
    parser.add_argument("--n-timesteps", type=int, 
                       default=get_env_int("FATTERBOX_N_TIMESTEPS", 10),
                       help="Diffusion steps for flow matching (default: 10, env: FATTERBOX_N_TIMESTEPS)")
    parser.add_argument("--flow-cfg-scale", type=float, 
                       default=get_env_float("FATTERBOX_FLOW_CFG_SCALE", 1.0),
                       help="CFG scale for mel decoder (default: 1.0, env: FATTERBOX_FLOW_CFG_SCALE)")
    
    parser.add_argument("--debug", 
                       action="store_true",
                       default=get_env_bool("FATTERBOX_DEBUG", False),
                       help="Enable debug logging (env: FATTERBOX_DEBUG)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    
    _LOGGER.info("="*60)
    _LOGGER.info("Fatterbox TTS Server - Wyoming + OpenAPI")
    _LOGGER.info("="*60)
    
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
    
    # Run both servers concurrently
    _LOGGER.info("="*60)
    await asyncio.gather(
        run_wyoming_server(args, model, voices, wyoming_info),
        run_fastapi_server(args, model, voices)
    )


if __name__ == "__main__":
    asyncio.run(main())
