#!/usr/bin/env python3
"""
Pre-download Chatterbox TTS model for offline Docker usage.
This script is run during Docker image build to cache all model files.
"""
import logging
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_LOGGER = logging.getLogger(__name__)


def patch_watermarker():
    """Patch watermarker if perth module is not available."""
    try:
        import perth
        if perth.PerthImplicitWatermarker is None:
            raise ImportError("Perth watermarker not properly initialized")
    except (ImportError, AttributeError) as e:
        _LOGGER.warning(f"Perth watermarking not available ({e}), disabling watermarking")
        
        class DummyWatermarker:
            def apply_watermark(self, audio, *args, **kwargs):
                return audio
            
            def __call__(self, audio, *args, **kwargs):
                return audio
        
        import types
        if 'perth' not in sys.modules:
            sys.modules['perth'] = types.ModuleType('perth')
        sys.modules['perth'].PerthImplicitWatermarker = lambda: DummyWatermarker()


def main():
    """Download and cache the Chatterbox TTS model."""
    _LOGGER.info("="*60)
    _LOGGER.info("Fatterbox Docker Initialization")
    _LOGGER.info("Pre-downloading Chatterbox TTS model...")
    _LOGGER.info("="*60)
    
    # Patch watermarker before importing chatterbox
    patch_watermarker()
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        # Detect available device (CPU during Docker build, usually)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LOGGER.info(f"Using device: {device}")
        
        # Load model (this triggers download and caching)
        _LOGGER.info("Loading Chatterbox TTS model...")
        model = ChatterboxTTS.from_pretrained(device=device)
        _LOGGER.info("✓ Model loaded successfully")
        
        # Do a test generation to ensure everything is cached
        _LOGGER.info("Performing test generation to verify model...")
        with torch.no_grad():
            test_audio = model.generate(
                "Testing model initialization.", 
                t3_params={"generate_token_backend": "eager"}  # Use eager for compatibility
            )
        _LOGGER.info("✓ Test generation successful")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        _LOGGER.info("="*60)
        _LOGGER.info("✓ Initialization complete!")
        _LOGGER.info("Model is cached and ready for offline use")
        _LOGGER.info("="*60)
        
    except Exception as e:
        _LOGGER.error(f"Failed to initialize model: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
