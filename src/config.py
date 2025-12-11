import logging
from typing import Optional

import torch

from voice_loader import VoiceLoader


class Config:
    """Central configuration management"""
    def __init__(self, args):
        self.host = args.host
        self.openai_port = args.openai_port
        self.wyoming_port = args.wyoming_port
        self.debug = args.debug
        self.device = self._validate_device(args.device)
        self.stream = args.stream
        self.language_id = args.language_id or "en"
        
        # Voice management with polling
        self.voice_loader = VoiceLoader(
            voices_dir=args.voices_dir,
            default_voice=args.default_voice
        )
        
        # Generation parameters
        self.exaggeration = args.exaggeration
        self.temperature = args.temperature
        self.cfg_weight = args.cfg_weight
        self.min_p = args.min_p
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.dtype = args.dtype
        
        # T3 optimization params
        # FIXED: Disabled skip_when_1 to prevent CUDA graph capture errors
        # The skip_when_1 optimization causes "operation not permitted when stream is capturing"
        # errors because it tries to compare top_p tensor during graph capture
        self.t3_params = {
            "initial_forward_pass_backend": "eager",
            "generate_token_backend": "cudagraphs-manual",
            "stride_length": 4,
            "skip_when_1": False,  # Changed from True to prevent graph capture errors
        }
    
    @staticmethod
    def _validate_device(device: str) -> str:
        """Validate and fallback device selection"""
        if "cuda" in device and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if "mps" in device and not torch.backends.mps.is_available():
            logging.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return device
