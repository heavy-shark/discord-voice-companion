"""Discord voice playback using VoiceClient."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

import discord

from ..runtime_ffmpeg import configure_pydub_once, resolve_ffmpeg_executable
from .pcm_stream import PCMStreamAudioSource

logger = logging.getLogger(__name__)

# Buffer this much speech before unmuting, so the first frames do not play as
# silence while the TTS response is still opening.
_PREBUFFER_SECONDS = 0.20
_PREBUFFER_BYTES = int(48000 * _PREBUFFER_SECONDS) * 2  # mono PCM16 at 48 kHz


class VoicePlaybackManager:
    """
    Manages audio playback into Discord voice channels.

    Uses VoiceClient.play() with FFmpeg-based AudioSource.
    Handles playback completion events and "speaking mode" state.
    """

    def __init__(self, voice_client: discord.VoiceClient) -> None:
        """
        Initialize playback manager.

        Args:
            voice_client: Discord VoiceClient connected to channel
        """
        self.voice_client = voice_client
        self.is_speaking = False
        self._playback_done_event: Optional[asyncio.Event] = None
        self._playback_error: Optional[Exception] = None
        self._active_stream: Optional[PCMStreamAudioSource] = None
        logger.info("VoicePlaybackManager initialized")

    def _finish_playback(self, error: Optional[Exception]) -> None:
        """Finalize playback state on the asyncio loop."""
        self.is_speaking = False
        self._playback_error = error
        if self._playback_done_event and not self._playback_done_event.is_set():
            self._playback_done_event.set()

    async def play_audio(self, audio_source: discord.AudioSource) -> None:
        """
        Play audio into voice channel and wait for completion.

        Args:
            audio_source: AudioSource to play (e.g., FFmpegAudio for MP3)
        """
        if self.is_playing():
            raise RuntimeError("Playback already in progress.")

        loop = asyncio.get_running_loop()
        self.is_speaking = True
        self._playback_error = None
        self._playback_done_event = asyncio.Event()

        def on_complete(error: Optional[Exception]) -> None:
            """Called when playback finishes."""
            loop.call_soon_threadsafe(self._finish_playback, error)

        try:
            logger.debug("Starting audio playback")
            self.voice_client.play(audio_source, after=on_complete)
            await self._playback_done_event.wait()
        except Exception as exc:
            self._finish_playback(None)
            logger.error("Failed to play audio: %s", exc, exc_info=True)
            raise RuntimeError("Failed to start audio playback.") from exc
        finally:
            self._playback_done_event = None

        if self._playback_error is not None:
            error = self._playback_error
            self._playback_error = None
            raise RuntimeError("Playback failed.") from error

    async def play_file(self, audio_path: Path) -> None:
        """Create an FFmpegOpusAudio source from a file path and play it."""
        logger.info("Playing audio file %s", audio_path)
        configure_pydub_once()
        ffmpeg = resolve_ffmpeg_executable()
        audio_source = discord.FFmpegOpusAudio(str(audio_path), executable=ffmpeg)
        await self.play_audio(audio_source)

    async def play_mono_pcm_stream(self, chunks: AsyncIterator[bytes]) -> float:
        """Play 48 kHz mono PCM16 as it arrives; return the seconds actually played.

        Playback starts once a short lead-in has buffered and continues while the
        rest of the stream is still downloading, so time-to-first-word is bounded
        by the TTS provider rather than by full synthesis.
        """
        if self.is_playing():
            raise RuntimeError("Playback already in progress.")

        source = PCMStreamAudioSource()
        self._active_stream = source
        play_task: Optional[asyncio.Task[None]] = None

        def start_playback() -> None:
            nonlocal play_task
            if play_task is None:
                play_task = asyncio.create_task(
                    self.play_audio(source), name="tts-playback"
                )

        try:
            pending = bytearray()
            async for chunk in chunks:
                if source.is_stopped:
                    logger.debug("Playback was interrupted; dropping the rest of the stream.")
                    break
                if not chunk:
                    continue

                if play_task is not None:
                    source.feed_mono(chunk)
                    continue

                pending.extend(chunk)
                if len(pending) >= _PREBUFFER_BYTES:
                    source.feed_mono(bytes(pending))
                    pending.clear()
                    start_playback()

            # Reply shorter than the lead-in buffer: play whatever arrived.
            if play_task is None and pending and not source.is_stopped:
                source.feed_mono(bytes(pending))
                start_playback()
        finally:
            # Closing the input first lets an already-running player drain and
            # finish on its own instead of deadlocking on more bytes.
            source.close_input()
            if play_task is not None:
                await play_task
            if self._active_stream is source:
                self._active_stream = None

        return source.seconds_played

    async def stop_playback(self) -> None:
        """Stop current playback."""
        stream, self._active_stream = self._active_stream, None
        if stream is not None:
            stream.stop()

        if not self.is_playing():
            return

        if self.voice_client.is_playing():
            self.voice_client.stop()

        self._finish_playback(None)
        logger.debug("Playback stopped")

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.voice_client.is_playing() or self.is_speaking
