import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

import torch
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (Synthesize, SynthesizeChunk, SynthesizeStart,
                          SynthesizeStop, SynthesizeStopped)

from audio_processor import AudioProcessor
from config import Config
from model_manager import ModelManager


class WyomingEventHandler(AsyncEventHandler):
    """Handles Wyoming protocol events for TTS with parallel prefetching"""
    
    def __init__(self, wyoming_info: Info, model_manager: ModelManager, 
                 audio_processor: AudioProcessor, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.config = config
        
        # Audio format
        self.sample_rate = 24000  # Chatterbox outputs at 24kHz
        self.sample_width = 2
        self.channels = 1
        self.chunk_size = 2048  # Larger chunks for better throughput
        
        # Thread pool for parallel generation (max 3 concurrent as per wyoming_openai)
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Streaming state
        self.streaming_text_buffer = []
        self.streaming_voice = None
        self.is_streaming = False
    
    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming protocol events"""
        if event.type == "describe":
            logging.info("Received describe event")
            # Refresh Wyoming info to include any newly added voices
            updated_info = self._get_updated_wyoming_info()
            await self.write_event(updated_info.event())
            return True
        
        # Streaming protocol
        if SynthesizeStart.is_type(event.type):
            synth_start = SynthesizeStart.from_event(event)
            self.streaming_voice = synth_start.voice.name if synth_start.voice else self.config.voice_loader.default_voice_name
            self.streaming_text_buffer = []
            self.is_streaming = True
            logging.info(f"Wyoming TTS streaming started with voice '{self.streaming_voice}'")
            return True
        
        if SynthesizeChunk.is_type(event.type):
            synth_chunk = SynthesizeChunk.from_event(event)
            self.streaming_text_buffer.append(synth_chunk.text)
            logging.debug(f"Received text chunk: '{synth_chunk.text}'")
            return True
        
        if SynthesizeStop.is_type(event.type):
            logging.info("Wyoming TTS streaming stopped")
            full_text = "".join(self.streaming_text_buffer)
            await self._generate_and_stream_audio(full_text, self.streaming_voice or self.config.voice_loader.default_voice_name)
            await self.write_event(SynthesizeStopped().event())
            
            self.streaming_text_buffer = []
            self.streaming_voice = None
            self.is_streaming = False
            return True
        
        # Legacy protocol
        if Synthesize.is_type(event.type):
            if self.is_streaming:
                logging.warning("Ignoring legacy Synthesize during streaming session")
                return True
            
            synthesize = Synthesize.from_event(event)
            voice_name = synthesize.voice.name if synthesize.voice else self.config.voice_loader.default_voice_name
            logging.info(f"Wyoming TTS request (legacy): '{synthesize.text}' with voice '{voice_name}'")
            await self._generate_and_stream_audio(synthesize.text, voice_name)
            return True
        
        logging.warning(f"Unexpected event type: {event.type}")
        return True
    
    def _get_updated_wyoming_info(self) -> Info:
        """Get updated Wyoming info with current voices"""
        return Info(
            tts=[TtsProgram(
                name="chatterbox",
                version="1.0.0",
                description="Chatterbox TTS with voice cloning",
                attribution=Attribution(
                    name="Chatterbox",
                    url="https://github.com/chatterbox-tts/chatterbox"
                ),
                installed=True,
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=voice_name,
                        version="1.0.0",
                        description=f"Chatterbox voice: {voice_name}",
                        attribution=Attribution(name="Chatterbox", url=""),
                        installed=True,
                        languages=[self.config.language_id]
                    )
                    for voice_name in self.config.voice_loader.list_voices()
                ]
            )]
        )
    
    async def _generate_and_stream_audio(self, text: str, voice_name: str):
        """Generate audio with parallel prefetching and stream via Wyoming protocol"""
        loop = asyncio.get_event_loop()
        
        # Get voice audio prompt (polling handles discovery automatically)
        audio_prompt_path = self.config.voice_loader.get_voice_path(voice_name)
        if not audio_prompt_path:
            logging.error(f"Voice '{voice_name}' not found, using default")
            audio_prompt_path = self.config.voice_loader.get_default_voice_path()
        
        # Prepare voice conditionals in executor to not block
        await loop.run_in_executor(
            self.executor, self.model_manager.prepare_voice_conditionals, audio_prompt_path
        )
        
        # Generation kwargs - uses default cfg_weight from config
        kwargs = {
            "language_id": self.config.language_id,
            "exaggeration": self.config.exaggeration,
            "temperature": self.config.temperature,
            "cfg_weight": self.config.cfg_weight,
            "min_p": self.config.min_p,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "t3_params": self.config.t3_params,
        }
        
        logging.info(f"Wyoming request: text='{text[:50]}...', voice='{voice_name}', cfg_weight={self.config.cfg_weight}")
        
        # Send audio start
        await self.write_event(AudioStart(
            rate=self.sample_rate,
            width=self.sample_width,
            channels=self.channels
        ).event())
        
        # Split into sentences
        sentences = [s for s in self.audio_processor.split_sentences(text) if s.strip()]
        
        if not sentences:
            await self.write_event(AudioStop().event())
            return
        
        # Use parallel prefetching with async queue for low latency
        # This mimics wyoming_openai's approach of prefetching up to 3 requests
        await self._parallel_generate_and_stream(sentences, kwargs, loop)
        
        # Send audio stop
        await self.write_event(AudioStop().event())
        await loop.run_in_executor(self.executor, self.model_manager.cleanup_after_generation)
        logging.info("Finished Wyoming TTS generation")
    
    async def _parallel_generate_and_stream(self, sentences: List[str], kwargs: dict, loop):
        """Generate sentences in parallel (up to 3) and stream sequentially"""
        # Queue to hold generated audio futures
        futures = []
        
        # Submit up to 3 generation tasks at a time
        for i, sentence in enumerate(sentences):
            # Wait if we have 3 tasks in flight
            if len(futures) >= 3:
                # Pop and stream the first completed task
                wav_tensor = await futures.pop(0)
                await self._stream_audio_chunk(wav_tensor)
            
            # Submit next generation task
            logging.debug(f"Submitting generation for sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
            
            # Create a wrapper function that properly unpacks kwargs
            def generate_with_kwargs(text=sentence, params=kwargs):
                return self.model_manager.generate(text, **params)
            
            future = loop.run_in_executor(self.executor, generate_with_kwargs)
            futures.append(future)
        
        # Stream remaining futures in order
        for future in futures:
            wav_tensor = await future
            await self._stream_audio_chunk(wav_tensor)
    
    async def _stream_audio_chunk(self, wav_tensor: torch.Tensor):
        """Stream a single audio chunk"""
        waveform = self.audio_processor.process_waveform(wav_tensor)
        audio_bytes = self.audio_processor.to_pcm16(waveform)
        
        # Stream in larger chunks for better throughput, minimal delay
        for i in range(0, len(audio_bytes), self.chunk_size):
            chunk = audio_bytes[i:i + self.chunk_size]
            await self.write_event(AudioChunk(
                rate=self.sample_rate,
                width=self.sample_width,
                channels=self.channels,
                audio=chunk
            ).event())
            
            # Small delay to prevent overwhelming the client
            if i + self.chunk_size < len(audio_bytes):
                await asyncio.sleep(0.0001)


class WyomingServer:
    """Wyoming protocol server"""
    
    def __init__(self, model_manager: ModelManager, audio_processor: AudioProcessor, config: Config):
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.config = config
    
    def _create_wyoming_info(self) -> Info:
        """Create Wyoming info with available voices"""
        return Info(
            tts=[TtsProgram(
                name="chatterbox",
                version="1.0.0",
                description="Chatterbox TTS with voice cloning",
                attribution=Attribution(
                    name="Chatterbox",
                    url="https://github.com/chatterbox-tts/chatterbox"
                ),
                installed=True,
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=voice_name,
                        version="1.0.0",
                        description=f"Chatterbox voice: {voice_name}",
                        attribution=Attribution(name="Chatterbox", url=""),
                        installed=True,
                        languages=[self.config.language_id]
                    )
                    for voice_name in self.config.voice_loader.list_voices()
                ]
            )]
        )
    
    async def run(self):
        """Run Wyoming server"""
        wyoming_info = self._create_wyoming_info()
        logging.info(f"Wyoming server starting on port {self.config.wyoming_port}")
        
        await AsyncServer.from_uri(f"tcp://0.0.0.0:{self.config.wyoming_port}").run(
            partial(WyomingEventHandler, wyoming_info, self.model_manager, 
                   self.audio_processor, self.config)
        )
