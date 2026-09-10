"""Discord integration package.

Handles all Discord API interactions including gateway connection,
voice channel management, audio receive, and playback.
"""

from .client import DiscordClient
from .pcm_stream import PCMStreamAudioSource, mono_to_stereo
from .playback import VoicePlaybackManager
from .voice_sink import DiscordAudioSink

__all__ = [
    "DiscordClient",
    "DiscordAudioSink",
    "PCMStreamAudioSource",
    "VoicePlaybackManager",
    "mono_to_stereo",
]
