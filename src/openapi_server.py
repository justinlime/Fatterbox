import logging
from typing import List

import torch
from flask import Flask, Response, jsonify, request, send_file

from audio_processor import AudioProcessor
from config import Config
from model_manager import ModelManager
from voice_loader import VoiceLoader


class FlaskServer:
    """OpenAI-compatible API server"""
    
    def __init__(self, model_manager: ModelManager, audio_processor: AudioProcessor, config: Config):
        self.app = Flask(__name__)
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.config = config
        
        self._register_routes()
    
    def _register_routes(self):
        """Register Flask routes"""
        self.app.route("/v1/audio/speech", methods=["POST"])(self.openai_tts)
        self.app.route("/v1/models", methods=["GET"])(self.list_models)
        self.app.route("/v1/voices", methods=["GET"])(self.list_voices)
        # OpenWebUI compat
        self.app.route("/v1/audio/models", methods=["GET"])(self.list_models)
        self.app.route("/v1/audio/voices", methods=["GET"])(self.list_voices)
    
    def openai_tts(self):
        """OpenAI-compatible TTS endpoint"""
        payload = request.get_json(force=True)
        
        # Extract parameters
        text = payload.get("input", "")
        voice = payload.get("voice", self.config.voice_loader.default_voice_name)
        # OpenAI uses "speed" parameter, but we map it to cfg_weight
        # If "speed" is provided, use it; otherwise use default cfg_weight
        cfg_weight = payload.get("speed", self.config.cfg_weight)
        stream = payload.get("stream", self.config.stream)
        fmt = payload.get("response_format", "pcm").lower()
        
        if not text:
            return jsonify({"error": "Missing input in request"}), 400
        
        # Get audio prompt (polling handles discovery automatically)
        audio_prompt_path = self.config.voice_loader.get_voice_path(voice)
        if not audio_prompt_path:
            available_voices = self.config.voice_loader.get_all_voices_with_openai()
            return jsonify({
                "error": f"Voice '{voice}' not found",
                "available_voices": available_voices
            }), 400
        
        # Prepare model
        self.model_manager.prepare_voice_conditionals(audio_prompt_path)
        
        # Generation kwargs
        kwargs = {
            "language_id": self.config.language_id,
            "exaggeration": self.config.exaggeration,
            "temperature": self.config.temperature,
            "cfg_weight": cfg_weight,
            "min_p": self.config.min_p,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "t3_params": self.config.t3_params,
        }
        
        logging.info(f"OpenAI {'streaming' if stream else 'non-streaming'} request: text='{text[:50]}...', voice='{voice}', cfg_weight={cfg_weight}")
        
        if stream:
            return self._stream_response(text, fmt, kwargs)
        else:
            return self._full_response(text, fmt, kwargs)
    
    def _stream_response(self, text: str, fmt: str, kwargs: dict):
        """Generate streaming response"""
        if fmt not in ['pcm', 'mp3']:
            return jsonify({"error": f"Streaming only supports 'pcm' and 'mp3' formats"}), 400
        
        def generate_stream():
            try:
                sentences = self.audio_processor.split_sentences(text)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    wav_tensor = self.model_manager.generate(sentence, **kwargs)
                    waveform = self.audio_processor.process_waveform(wav_tensor)
                    
                    if fmt == 'pcm':
                        yield self.audio_processor.to_pcm16(waveform)
                    elif fmt == 'mp3':
                        buffer = self.audio_processor.save_audio(waveform, 'mp3')
                        yield buffer.getvalue()
            finally:
                self.model_manager.cleanup_after_generation()
        
        mimetype = self.audio_processor.MIMETYPES[fmt]
        return Response(generate_stream(), mimetype=mimetype)
    
    def _full_response(self, text: str, fmt: str, kwargs: dict):
        """Generate full response"""
        if fmt not in self.audio_processor.MIMETYPES:
            return jsonify({"error": f"Unsupported format: {fmt}"}), 400
        
        try:
            sentences = self.audio_processor.split_sentences(text)
            audio_chunks = self.model_manager.generate_batch(sentences, **kwargs)
            
            final_audio = torch.cat(audio_chunks, dim=-1) if len(audio_chunks) > 1 else audio_chunks[0]
            waveform = self.audio_processor.process_waveform(final_audio)
            
            buffer = self.audio_processor.save_audio(waveform, fmt)
            mimetype = self.audio_processor.MIMETYPES[fmt]
            
            return send_file(buffer, mimetype=mimetype)
        finally:
            self.model_manager.cleanup_after_generation()
    
    def list_models(self):
        """List available models"""
        return jsonify({
            "object": "list",
            "data": [
                {"id": "tts-1", "object": "model", "created": 1677610600, "owned_by": "openai"},
                {"id": "tts-1-hd", "object": "model", "created": 1677610600, "owned_by": "openai"}
            ]
        })
    
    def list_voices(self):
        """List available voices"""
        actual_voices = self.config.voice_loader.list_voices()
        all_voices = self.config.voice_loader.get_all_voices_with_openai()
        default_voice = self.config.voice_loader.default_voice_name
        
        return jsonify({
            "voices": all_voices,
            "actual_voices": actual_voices,
            "openai_voices": VoiceLoader.OPENAI_VOICE_NAMES,
            "default_voice": default_voice,
            "openai_mapping": f"OpenAI voices (alloy, echo, etc.) map to '{default_voice}'",
            "count": len(actual_voices)
        })
    
    def run(self):
        """Run Flask server"""
        self.app.run(
            host=self.config.host,
            port=self.config.openai_port,
            debug=self.config.debug,
            use_reloader=False
        )
