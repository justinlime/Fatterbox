"""Utility functions and constants."""
import os
import re


# ANSI color codes for logging
class Colors:
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'


def get_env_str(key: str, default: str) -> str:
    """Get string from environment variable with fallback to default."""
    return os.getenv(key, default)


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


def get_env_bool(key: str, default: bool) -> bool:
    """Get bool from environment variable with fallback to default."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def split_text(text: str, max_sentences: int = 1, max_chunk_length: int = 300) -> list[str]:
    """
    Split text into chunks for pseudo-streaming optimized for low latency.
    Uses single-sentence chunks by default for fastest time-to-first-audio.
    Also ensures chunks don't exceed max_chunk_length to prevent VRAM spikes.
    """
    # Split by sentence-ending punctuation (.!?) keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If we have no sentences, split by max_chunk_length if text is too long
    if not sentences:
        if len(text) > max_chunk_length:
            # Split long text by word boundaries
            words = text.split()
            chunks = []
            current = ""
            for word in words:
                if len(current) + len(word) + 1 > max_chunk_length:
                    if current:
                        chunks.append(current.strip())
                    current = word
                else:
                    current += (" " if current else "") + word
            if current:
                chunks.append(current.strip())
            return chunks
        return [text]
    
    # Group sentences into chunks, respecting both max_sentences and max_chunk_length
    chunks = []
    current_chunk = ""
    sentence_count = 0
    
    for sentence in sentences:
        # Check if adding this sentence would exceed limits
        would_exceed_length = len(current_chunk) + len(sentence) + 1 > max_chunk_length
        would_exceed_count = sentence_count >= max_sentences
        
        if current_chunk and (would_exceed_length or would_exceed_count):
            # Flush current chunk and start new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            sentence_count = 1
        else:
            # Add sentence to current chunk
            current_chunk += (" " if current_chunk else "") + sentence
            sentence_count += 1
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
