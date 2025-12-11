import gc
import logging
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from config import Config


class ModelManager:
    """Manages model loading, optimization, and device management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[ChatterboxMultilingualTTS] = None
        self.inference_lock = Lock()
        self.generation_count = 0
        self.cached_conds: Dict[str, any] = {}
        self.current_audio_prompt = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Load and optimize the model"""
        logging.info("Loading Chatterbox TTS model...")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.config.device)
        
        if "cuda" in self.config.device:
            self._apply_cuda_optimizations()
            self._warmup()
    
    def _apply_cuda_optimizations(self):
        """Apply CUDA-specific optimizations"""
        # T3 dtype optimization
        if self.config.dtype != "float32":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            logging.info(f"Optimizing T3 model with dtype: {self.config.dtype}")
            self.model.t3.to(dtype=dtype_map[self.config.dtype])
            self.model.conds.t3.to(dtype=dtype_map[self.config.dtype])
            torch.cuda.empty_cache()
        
        # Torch compile
        if hasattr(torch, 'compile'):
            try:
                logging.info("Compiling model with torch.compile...")
                self.model.t3 = torch.compile(
                    self.model.t3,
                    mode="reduce-overhead",
                    fullgraph=True
                )
            except Exception as e:
                logging.warning(f"Compilation failed: {e}")
        
        # Hardware-specific optimizations
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("TF32 enabled for Ampere+ GPU")
        
        torch.backends.cudnn.benchmark = True
    
    def _warmup(self):
        """Warmup generations for CUDA graph initialization"""
        logging.info("Running warmup generation...")
        try:
            for text in ["Warmup generation for CUDA graph initialization.", 
                        "Second warmup for full speed."]:
                self.model.generate(text,
                                    language_id=self.config.language_id,
                                    t3_params=self.config.t3_params)
            logging.info("Warmup complete!")
        except Exception as e:
            logging.warning(f"Warmup failed: {e}")
    
    def cleanup_after_generation(self):
        """Cleanup after generation"""
        # Periodic cache clearing to prevent VRAM usage from growing indefinitely
        self.generation_count += 1
        if self.generation_count >= 5 and "cuda" in self.config.device:
            self.generation_count = 0
            torch.cuda.empty_cache()
            gc.collect()
            logging.debug("CUDA cache cleared")
    
    def prepare_voice_conditionals(self, audio_prompt_path: str):
        """Load or retrieve cached voice conditionals"""
        if audio_prompt_path == self.current_audio_prompt:
            logging.debug("Using existing conditionals")
            return
        
        voice_conds = self.cached_conds.get(audio_prompt_path)
        
        if voice_conds:
            logging.debug("Using cached conditionals")
            self.model.conds = voice_conds
        else:
            logging.info(f"Generating new conditionals for: {Path(audio_prompt_path).name}")
            self.model.prepare_conditionals(audio_prompt_path)
            self.cached_conds[audio_prompt_path] = self.model.conds
        
        self.current_audio_prompt = audio_prompt_path
    
    def generate(self, text: str, **kwargs) -> torch.Tensor:
        """Generate audio with thread safety"""
        with self.inference_lock:
            return self.model.generate(text, **kwargs)
    
    def generate_batch(self, texts: List[str], **kwargs) -> List[torch.Tensor]:
        """Generate audio batch if supported"""
        if len(texts) > 1 and hasattr(self.model, 'generate_batch'):
            logging.info(f"Using batch generation for {len(texts)} sentences")
            with self.inference_lock:
                return self.model.generate_batch(texts, **kwargs)
        
        # Fallback to sequential
        return [self.generate(text, **kwargs) for text in texts if text.strip()]
