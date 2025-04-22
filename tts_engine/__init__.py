"""
TTS Engine package for Orpheus text-to-speech system.

This package contains the core components for audio generation:
- inference.py: Token generation and API handling
- speechpipe.py: Audio conversion pipeline
- coreml_wrapper.py: Apple Silicon Neural Engine acceleration
"""

# Make key components available at package level
from .inference import (
    generate_speech_from_api,
    stream_audio,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    VOICE_TO_LANGUAGE,
    AVAILABLE_LANGUAGES,
    MAX_BATCH_CHARS,
    CROSSFADE_MS,
    list_available_voices,
    API_URL,
    HEADERS
)

# Expose hardware detection flags
from .speechpipe import (
    APPLE_SILICON,
    CUDA_AVAILABLE,
    DEVICE
)

__all__ = [
    # Core Functions
    "generate_speech_from_api",
    "stream_audio",
    "list_available_voices",
    
    # Speech Processing
    "convert_to_audio",
    "turn_token_into_id",
    "reset_state",
    "DEVICE",
    
    # Constants & Settings
    "AVAILABLE_VOICES",
    "DEFAULT_VOICE",
    "VOICE_TO_LANGUAGE",
    "AVAILABLE_LANGUAGES",
    "MAX_BATCH_CHARS",
    "CROSSFADE_MS",
    
    # Configuration (Example)
    "API_URL",
    "HEADERS",
]
