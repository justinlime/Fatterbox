import logging
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional


class VoiceLoader:
    """Loads and manages multiple voice prompts from directory with runtime discovery"""
    
    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus'}
    OPENAI_VOICE_NAMES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    def __init__(self, voices_dir: str, default_voice: Optional[str] = None):
        self.voices_dir = Path(voices_dir)
        self.voices: Dict[str, str] = {}
        self.default_voice_name = default_voice
        self.last_scan_time = 0
        self.scan_lock = Lock()
        self.polling_thread = None
        self.should_poll = False
        
        # Initial scan
        if not self.voices_dir.exists():
            raise ValueError(f"Voices directory does not exist: {voices_dir}")
        if not self.voices_dir.is_dir():
            raise ValueError(f"Voices path is not a directory: {voices_dir}")
        
        self._scan_voices_directory()
        
        if not self.voices:
            raise ValueError(f"No audio files found in voices directory: {voices_dir}")
        
        # Set default voice
        if self.default_voice_name:
            if self.default_voice_name not in self.voices:
                logging.warning(f"Default voice '{self.default_voice_name}' not found, using first available")
                self.default_voice_name = list(self.voices.keys())[0]
        else:
            self.default_voice_name = list(self.voices.keys())[0]
        
        logging.info(f"Default voice set to: {self.default_voice_name}")
        logging.info(f"OpenAI-compatible voice names mapped to: {self.default_voice_name}")
    
    def _scan_voices_directory(self):
        """Scan directory and load all audio files as voices"""
        with self.scan_lock:
            voices_before = set(self.voices.keys())
            
            # Scan for audio files
            audio_files = []
            for ext in self.SUPPORTED_AUDIO_FORMATS:
                audio_files.extend(self.voices_dir.glob(f"*{ext}"))
                audio_files.extend(self.voices_dir.glob(f"*{ext.upper()}"))
            
            # Update voices dictionary
            new_voices = {}
            for audio_file in sorted(audio_files):
                voice_name = audio_file.stem
                new_voices[voice_name] = str(audio_file.absolute())
            
            # Detect changes
            voices_after = set(new_voices.keys())
            added = voices_after - voices_before
            removed = voices_before - voices_after
            
            if added:
                logging.info(f"Added voices: {sorted(added)}")
            if removed:
                logging.info(f"Removed voices: {sorted(removed)}")
            
            self.voices = new_voices
            self.last_scan_time = time.time()
            
            if not voices_before and self.voices:
                # Initial scan
                logging.info(f"Loaded {len(self.voices)} voices: {sorted(self.voices.keys())}")
    
    def _polling_loop(self):
        """Background thread that polls the directory for changes every 1 second"""
        logging.info("Voice polling started (interval: 1.0s)")
        while self.should_poll:
            time.sleep(1.0)
            if self.should_poll:  # Check again after sleep
                try:
                    self._scan_voices_directory()
                except Exception as e:
                    logging.error(f"Error during voice directory polling: {e}")
    
    def start_polling(self):
        """Start the background polling thread"""
        if self.polling_thread is None or not self.polling_thread.is_alive():
            self.should_poll = True
            self.polling_thread = Thread(target=self._polling_loop, daemon=True)
            self.polling_thread.start()
            logging.info("Voice directory polling enabled")
    
    def stop_polling(self):
        """Stop the background polling thread"""
        self.should_poll = False
        if self.polling_thread is not None:
            self.polling_thread.join(timeout=2.0)
            logging.info("Voice directory polling stopped")
    
    def get_voice_path(self, voice_name: str) -> Optional[str]:
        """Get audio prompt path for a voice name"""
        # Check if it's an OpenAI voice name
        if voice_name in self.OPENAI_VOICE_NAMES:
            voice_name = self.default_voice_name
        
        return self.voices.get(voice_name)
    
    def get_default_voice_path(self) -> str:
        """Get default voice audio prompt path"""
        return self.voices[self.default_voice_name]
    
    def list_voices(self) -> List[str]:
        """Get list of all available voice names"""
        return sorted(self.voices.keys())
    
    def get_all_voices_with_openai(self) -> List[str]:
        """Get all voices including OpenAI-compatible names"""
        actual_voices = self.list_voices()
        return sorted(set(actual_voices + self.OPENAI_VOICE_NAMES))
