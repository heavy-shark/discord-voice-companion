"""Transcription package.

Handles audio preprocessing and speech-to-text via Gemini (Live API streaming,
or one request per utterance).

Utterance segmentation is driven by Discord's own speaking state plus an idle
gap (see :mod:`.coordinator`), so there is no local VAD.
"""

from .coordinator import PerUserTranscriptionCoordinator
from .gemini_stt import GeminiSTTClient, GeminiSTTError
from .preprocessing import convert_to_linear16, get_audio_duration, to_mono

__all__ = [
    "GeminiSTTClient",
    "GeminiSTTError",
    "convert_to_linear16",
    "to_mono",
    "get_audio_duration",
    "PerUserTranscriptionCoordinator",
]
