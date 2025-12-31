"""Model loading and initialization for Chatterbox TTS."""
import logging
from functools import lru_cache

import torch
from chatterbox.tts import ChatterboxTTS

_LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model(device: str = "cuda", dtype: str = "float32"):
    """Load Chatterbox TTS model (cached)."""
    _LOGGER.info(f"Loading Chatterbox TTS model on {device} with dtype {dtype}...")
    
    # Patch watermarker if perth module is not available or broken
    _patch_watermarker()
    
    # Load model
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Convert to specified dtype if not float32
    if dtype.lower() not in ["float32", "fp32"]:
        _convert_model_dtype(model, dtype)
    
    # Warmup for cudagraphs (recommended by Chatterbox docs)
    if device == "cuda":
        _warmup_model(model)
    
    _LOGGER.info("Model loaded successfully")
    return model


def _patch_watermarker():
    """Patch watermarker if perth module is not available or broken."""
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


def _convert_model_dtype(model, dtype: str):
    """Convert model to specified dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    
    if torch_dtype:
        _LOGGER.info(f"Converting model to {torch_dtype}...")
        # Use the recommended t3_to pattern from Chatterbox docs
        model.t3.to(dtype=torch_dtype)
        model.conds.t3.to(dtype=torch_dtype)
        torch.cuda.empty_cache()
        _LOGGER.info(f"Model converted to {torch_dtype}")


def _warmup_model(model):
    """Warmup model with cudagraphs optimization."""
    _LOGGER.info("Warming up model with cudagraphs optimization...")
    # First warmup call sets up the cudagraph
    model.generate(
        "Warmup for cudagraphs optimization.", 
        t3_params={"generate_token_backend": "cudagraphs-manual"}
    )
    # Second call runs at full speed
    model.generate(
        "Second warmup for full cudagraphs speed.",
        t3_params={"generate_token_backend": "cudagraphs-manual"}
    )
    _LOGGER.info("Warmup complete - cudagraphs ready")
