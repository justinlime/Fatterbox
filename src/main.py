import argparse
import asyncio
import logging
import os
from threading import Thread

from audio_processor import AudioProcessor
from config import Config
from openapi_server import FlaskServer
from model_manager import ModelManager
from wyoming_server import WyomingServer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for Chatterbox with Wyoming protocol support and multi-voice support."
    )
    
    # Server arguments
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--openai-port", type=int, default=int(os.getenv("OPENAI_PORT", os.getenv("PORT", "5002"))))
    parser.add_argument("--wyoming-port", type=int, default=int(os.getenv("WYOMING_PORT", "10200")))
    parser.add_argument("--debug", action="store_true", default=os.getenv("DEBUG", "").lower() == "true")
    parser.add_argument("--device", type=str, default=os.getenv("DEVICE", "cpu"))
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction,
                       default=os.getenv("STREAM", "").lower() == "true")
    parser.add_argument("--language-id", type=str, default=os.getenv("LANGUAGE_ID", "en"))
    
    # Voice configuration
    parser.add_argument("--voices-dir", type=str, default=os.getenv("VOICES_DIR"),
                       help="Directory containing voice audio files (wav, mp3, flac, etc.)")
    parser.add_argument("--default-voice", type=str, default=os.getenv("DEFAULT_VOICE"),
                       help="Default voice name for OpenAI-compatible voice names (alloy, echo, etc.)")
    
    # Generation arguments
    parser.add_argument("--exaggeration", type=float, default=float(os.getenv("EXAGGERATION", "0.5")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.8")))
    parser.add_argument("--cfg-weight", type=float, default=float(os.getenv("CFG_WEIGHT", "1.0")),
                       help="Classifier-free guidance weight (default: 1.0)")
    parser.add_argument("--min-p", type=float, default=float(os.getenv("MIN_P", "0.05")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "1.0")))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.getenv("REPETITION_PENALTY", "1.2")))
    parser.add_argument("--dtype", type=str, default=os.getenv("DTYPE", os.getenv("T3_DTYPE", "bfloat16")),
                       choices=["float32", "float16", "bfloat16"])
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.debug else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize components
    config = Config(args)
    model_manager = ModelManager(config)
    audio_processor = AudioProcessor()
    
    # Start voice directory polling
    config.voice_loader.start_polling()
    
    # Start Flask server in background thread
    openapi_server = FlaskServer(model_manager, audio_processor, config)
    flask_thread = Thread(target=openapi_server.run, daemon=True)
    flask_thread.start()
    
    logging.info(f"Flask server started on {config.host}:{config.openai_port}")
    logging.info(f"Starting Wyoming server on port {config.wyoming_port}...")
    
    # Run Wyoming server
    try:
        wyoming_server = WyomingServer(model_manager, audio_processor, config)
        asyncio.run(wyoming_server.run())
    except KeyboardInterrupt:
        logging.info("Shutting down servers...")
        config.voice_loader.stop_polling()


if __name__ == "__main__":
    main()
