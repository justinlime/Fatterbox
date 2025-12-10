"""
Docker initialization script to pre-download all models and dependencies
Run during Docker build (CPU-only) to enable fully offline operation

NOTE: This runs WITHOUT CUDA during Docker build, but the cached models
will be loaded on GPU at runtime if available.
"""
import os
import sys
import logging
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# Monkey-patch torch.load to force CPU during build
# This works around a bug in Chatterbox where it doesn't use map_location
# ============================================================================

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that forces map_location='cpu'"""
    # Only apply patch if map_location is not already specified
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
        logging.debug(f"Patched torch.load to use CPU (no CUDA during build)")
    return _original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = _patched_torch_load
logging.info("Applied torch.load patch to force CPU map_location")

def download_chatterbox_models():
    """Download Chatterbox TTS models on CPU (CUDA not available during build)"""
    logging.info("Downloading Chatterbox TTS models (CPU mode)...")
    
    # Verify we're in CPU mode
    if torch.cuda.is_available():
        logging.warning("CUDA detected during build - this is unexpected!")
    
    model = None
    
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        
        # MUST use device="cpu" since CUDA is not available during build
        logging.info("Loading model on CPU to trigger downloads...")
        model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        logging.info("✓ Chatterbox multilingual model downloaded and cached")
        
        # Note: Skipping test generation during build because the model
        # requires CUDA for some operations. The important part (downloading
        # model files) is complete. Model will be tested at runtime with GPU.
        logging.info("✓ Model files cached - ready for runtime use")
        
    except Exception as e:
        logging.error(f"Failed to download Chatterbox models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup to free memory
        if model is not None:
            del model

def download_pkuseg_data():
    """Download pkuseg tokenizer data used by Chatterbox for Chinese"""
    logging.info("Downloading pkuseg data...")
    try:
        import pkuseg
        
        # Download default model
        logging.info("Initializing pkuseg...")
        seg = pkuseg.pkuseg()
        
        # Trigger actual download by processing text
        test_text = "测试文本"
        result = seg.cut(test_text)
        logging.info(f"✓ pkuseg data downloaded (test result: {result})")
        
    except ImportError:
        logging.warning("pkuseg not installed - skipping (only needed for Chinese)")
    except Exception as e:
        logging.warning(f"pkuseg download failed (may not be needed): {e}")
        # Don't exit - pkuseg might not be needed for all languages

def download_nltk_data():
    """Download NLTK data used by pysbd"""
    logging.info("Downloading NLTK data for pysbd...")
    try:
        import nltk
        
        # pysbd uses these datasets
        packages_to_download = ['punkt', 'punkt_tab']
        
        for package in packages_to_download:
            try:
                logging.info(f"Downloading {package}...")
                nltk.download(package, quiet=False)
                logging.info(f"✓ Downloaded {package}")
            except Exception as e:
                logging.warning(f"Could not download {package}: {e}")
        
    except ImportError:
        logging.warning("NLTK not installed - skipping")
    except Exception as e:
        logging.warning(f"NLTK download warning: {e}")

def test_sentence_segmentation():
    """Test pysbd to ensure all data is cached"""
    logging.info("Testing sentence segmentation...")
    try:
        import pysbd
        
        # Test multiple languages to cache everything
        test_cases = [
            ("en", "Hello world. This is a test. It should work offline."),
            ("en", "Dr. Smith went to the store. He bought milk."),
        ]
        
        for lang, text in test_cases:
            segmenter = pysbd.Segmenter(language=lang, clean=True)
            sentences = segmenter.segment(text)
            logging.info(f"✓ Segmentation test [{lang}]: {len(sentences)} sentences")
        
    except Exception as e:
        logging.error(f"Sentence segmentation test failed: {e}")

def setup_cache_directories():
    """Ensure cache directories exist and are writable"""
    logging.info("Setting up cache directories...")
    
    cache_dirs = [
        os.getenv("HF_HOME", "~/.cache/huggingface"),
        os.getenv("TORCH_HOME", "~/.cache/torch"),
        os.getenv("NLTK_DATA", "~/.cache/nltk_data"),
    ]
    
    for cache_dir in cache_dirs:
        expanded = os.path.expanduser(cache_dir)
        os.makedirs(expanded, exist_ok=True)
        logging.info(f"✓ Created/verified: {expanded}")

def verify_offline_readiness():
    """Verify that all components can work offline"""
    logging.info("=" * 60)
    logging.info("Verifying offline readiness...")
    logging.info("=" * 60)
    
    # Check cache directories exist and have content
    cache_checks = [
        ("Hugging Face", os.getenv("HF_HOME", "~/.cache/huggingface")),
        ("Torch", os.getenv("TORCH_HOME", "~/.cache/torch")),
        ("NLTK", os.getenv("NLTK_DATA", "~/.cache/nltk_data")),
    ]
    
    total_size_mb = 0
    all_ok = True
    
    for name, cache_dir in cache_checks:
        expanded = os.path.expanduser(cache_dir)
        if os.path.exists(expanded):
            try:
                size_bytes = sum(f.stat().st_size for f in Path(expanded).rglob('*') if f.is_file())
                size_mb = size_bytes / (1024**2)
                total_size_mb += size_mb
                
                if size_mb > 0.1:  # At least 100KB
                    logging.info(f"✓ {name} cache: {size_mb:.1f} MB")
                else:
                    logging.warning(f"⚠ {name} cache exists but appears empty: {size_mb:.3f} MB")
                    all_ok = False
            except Exception as e:
                logging.warning(f"⚠ Could not check {name} cache: {e}")
                all_ok = False
        else:
            logging.warning(f"✗ {name} cache not found: {expanded}")
            all_ok = False
    
    logging.info(f"Total cached data: {total_size_mb:.1f} MB")
    
    if not all_ok:
        logging.warning("Some caches may not be properly initialized")
    
    return all_ok

def main():
    """Main initialization routine"""
    logging.info("=" * 60)
    logging.info("Docker Initialization - CPU Mode")
    logging.info("CUDA is NOT available during build (this is expected)")
    logging.info("Models will be cached for GPU use at runtime")
    logging.info("=" * 60)
    
    try:
        # 0. Setup cache directories
        setup_cache_directories()
        
        # 1. Download Chatterbox models (main download)
        download_chatterbox_models()
        
        # 2. Download pkuseg data (Chinese tokenizer used by Chatterbox)
        download_pkuseg_data()
        
        # 3. Download NLTK data (used by pysbd)
        download_nltk_data()
        
        # 4. Test sentence segmentation
        test_sentence_segmentation()
        
        # 5. Verify everything is cached
        all_ok = verify_offline_readiness()
        
        logging.info("=" * 60)
        if all_ok:
            logging.info("✓ Docker initialization complete - image ready for offline use")
            logging.info("✓ Models will load on GPU at runtime if CUDA is available")
        else:
            logging.warning("⚠ Initialization complete with warnings - check logs above")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"Docker initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
