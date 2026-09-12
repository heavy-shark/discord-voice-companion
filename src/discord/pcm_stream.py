"""A Discord audio source fed by a live PCM stream.

``discord.py`` pulls audio from an :class:`discord.AudioSource` on a dedicated
player thread, once every 20 ms, and expects exactly one frame of 48 kHz stereo
PCM16 back. :class:`PCMStreamAudioSource` bridges that pull-based thread to the
asyncio task streaming TTS bytes from ElevenLabs, so a reply starts playing while
the rest of the sentence is still being synthesized.
"""

from __future__ import annotations

import logging
import threading
from array import array

import discord

logger = logging.getLogger(__name__)

SAMPLE_RATE_HZ = 48000
CHANNELS = 2
SAMPLE_WIDTH_BYTES = 2
FRAME_DURATION_MS = 20

# 3840 bytes: 20 ms * 48 kHz * 2 channels * 2 bytes.
FRAME_SIZE_BYTES = (
    SAMPLE_RATE_HZ * FRAME_DURATION_MS // 1000 * CHANNELS * SAMPLE_WIDTH_BYTES
)
SILENCE_FRAME = b"\x00" * FRAME_SIZE_BYTES

# How long the player thread waits for more bytes before emitting a silence frame.
# Longer than one frame so a brief network hiccup does not punch a hole in speech.
_STARVE_TIMEOUT_SECONDS = 0.25

# Give up and end the source after this much continuous starvation, so a dead
# upstream cannot leave the bot "speaking" silence forever.
_MAX_STARVED_FRAMES = 250  # 5 seconds


def mono_to_stereo(mono_pcm: bytes) -> bytes:
    """Duplicate each mono PCM16 sample into a stereo frame."""
    if not mono_pcm:
        return b""
    usable = len(mono_pcm) - (len(mono_pcm) % SAMPLE_WIDTH_BYTES)
    if usable <= 0:
        return b""
    mono = array("h")
    mono.frombytes(mono_pcm[:usable])
    stereo = array("h", bytes(len(mono) * 2 * SAMPLE_WIDTH_BYTES))
    stereo[0::2] = mono
    stereo[1::2] = mono
    return stereo.tobytes()


class PCMStreamAudioSource(discord.AudioSource):
    """Play 48 kHz stereo PCM16 pushed in from another thread or task."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._condition = threading.Condition()
        self._input_closed = False
        self._stopped = False
        self._frames_read = 0
        self._starved_frames = 0
        # Producer side only (one asyncio task): the trailing byte of a chunk
        # that cut through the middle of a PCM16 sample.
        self._mono_remainder = b""

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def feed_mono(self, mono_pcm: bytes) -> None:
        """Append 48 kHz *mono* PCM16 (what ElevenLabs ``pcm_48000`` returns).

        HTTP chunks cut at arbitrary byte offsets, so roughly half of them end
        mid-sample. The odd byte is carried into the next chunk: dropping it
        would shift every later sample by one byte, which plays back as very
        loud noise rather than speech.
        """
        if not mono_pcm:
            return
        if self._mono_remainder:
            mono_pcm = self._mono_remainder + mono_pcm
            self._mono_remainder = b""
        if len(mono_pcm) % SAMPLE_WIDTH_BYTES:
            mono_pcm, self._mono_remainder = mono_pcm[:-1], mono_pcm[-1:]
        self.feed(mono_to_stereo(mono_pcm))

    def feed(self, stereo_pcm: bytes) -> None:
        """Append 48 kHz stereo PCM16."""
        if not stereo_pcm:
            return
        with self._condition:
            if self._stopped:
                return
            self._buffer.extend(stereo_pcm)
            self._condition.notify_all()

    def close_input(self) -> None:
        """Signal that no more audio is coming; playback ends once drained."""
        with self._condition:
            self._input_closed = True
            self._condition.notify_all()

    def stop(self) -> None:
        """Abandon buffered audio and end playback now (barge-in)."""
        self._mono_remainder = b""
        with self._condition:
            self._stopped = True
            self._input_closed = True
            self._buffer.clear()
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Consumer side (discord.py player thread)
    # ------------------------------------------------------------------

    @property
    def seconds_played(self) -> float:
        """Audio actually handed to Discord, for usage accounting."""
        return self._frames_read * FRAME_DURATION_MS / 1000.0

    @property
    def is_stopped(self) -> bool:
        """True once :meth:`stop` was called (barge-in or cleanup)."""
        with self._condition:
            return self._stopped

    def is_opus(self) -> bool:
        """Always False: this source yields PCM for discord.py to encode."""
        return False

    def read(self) -> bytes:
        """Return exactly one 20 ms frame, or ``b""`` to end playback."""
        with self._condition:
            while True:
                if self._stopped:
                    return b""

                if len(self._buffer) >= FRAME_SIZE_BYTES:
                    frame = bytes(self._buffer[:FRAME_SIZE_BYTES])
                    del self._buffer[:FRAME_SIZE_BYTES]
                    self._frames_read += 1
                    self._starved_frames = 0
                    return frame

                if self._input_closed:
                    if not self._buffer:
                        return b""
                    # Final partial frame: pad with silence rather than truncate,
                    # which would clip the last syllable.
                    frame = bytes(self._buffer).ljust(FRAME_SIZE_BYTES, b"\x00")
                    self._buffer.clear()
                    self._frames_read += 1
                    return frame

                if not self._condition.wait(timeout=_STARVE_TIMEOUT_SECONDS):
                    self._starved_frames += 1
                    if self._starved_frames >= _MAX_STARVED_FRAMES:
                        logger.warning(
                            "PCM stream starved for %.1fs; ending playback.",
                            _MAX_STARVED_FRAMES * FRAME_DURATION_MS / 1000.0,
                        )
                        return b""
                    # Hold the channel open with silence while we wait for bytes.
                    self._frames_read += 1
                    return SILENCE_FRAME

    def cleanup(self) -> None:
        """Called by discord.py once the player is done with this source."""
        self.stop()
