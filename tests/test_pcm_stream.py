"""Tests for the streaming PCM audio source handed to Discord."""

from __future__ import annotations

import math
import threading
from array import array

from src.discord.pcm_stream import (
    FRAME_SIZE_BYTES,
    SILENCE_FRAME,
    PCMStreamAudioSource,
    mono_to_stereo,
)


def mono(*samples: int) -> bytes:
    return array("h", samples).tobytes()


def test_frame_size_matches_discord_20ms_stereo_framing() -> None:
    # 20 ms * 48 kHz * 2 channels * 2 bytes
    assert FRAME_SIZE_BYTES == 3840
    assert len(SILENCE_FRAME) == FRAME_SIZE_BYTES


def test_mono_to_stereo_duplicates_each_sample() -> None:
    assert mono_to_stereo(mono(1, -2, 3)) == mono(1, 1, -2, -2, 3, 3)


def test_mono_to_stereo_handles_empty_and_truncated_input() -> None:
    assert mono_to_stereo(b"") == b""
    # A stray odd byte cannot form a sample and is dropped.
    assert mono_to_stereo(b"\x01") == b""


def test_read_returns_full_frames_in_order() -> None:
    source = PCMStreamAudioSource()
    # Two frames' worth of stereo audio = one frame of mono at half the size.
    source.feed(b"\x01" * FRAME_SIZE_BYTES + b"\x02" * FRAME_SIZE_BYTES)
    source.close_input()

    assert source.read() == b"\x01" * FRAME_SIZE_BYTES
    assert source.read() == b"\x02" * FRAME_SIZE_BYTES
    assert source.read() == b""


def test_final_partial_frame_is_padded_not_truncated() -> None:
    """Truncating the tail would clip the last syllable of a reply."""
    source = PCMStreamAudioSource()
    source.feed(b"\x07" * 100)
    source.close_input()

    frame = source.read()
    assert len(frame) == FRAME_SIZE_BYTES
    assert frame[:100] == b"\x07" * 100
    assert frame[100:] == b"\x00" * (FRAME_SIZE_BYTES - 100)
    assert source.read() == b""


def test_feed_mono_is_upmixed_to_stereo() -> None:
    source = PCMStreamAudioSource()
    source.feed_mono(mono(5) * (FRAME_SIZE_BYTES // 4))
    source.close_input()

    frame = source.read()
    samples = array("h")
    samples.frombytes(frame)
    # Left and right carry the same value.
    assert samples[0] == samples[1] == 5


def drain(source: PCMStreamAudioSource) -> bytes:
    """Read the source dry the way discord.py's player thread would."""
    out = bytearray()
    while True:
        frame = source.read()
        if not frame:
            return bytes(out)
        out.extend(frame)


def rms(pcm: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(float(value) * value for value in samples) / len(samples))


def test_feed_mono_carries_a_split_sample_across_chunks() -> None:
    """HTTP chunks cut anywhere, including through the middle of a sample."""
    source = PCMStreamAudioSource()
    payload = mono(1000, -2000, 3000, -4000)
    # Split so the second chunk starts mid-sample.
    source.feed_mono(payload[:3])
    source.feed_mono(payload[3:])
    source.close_input()

    assert drain(source).startswith(mono(1000, 1000, -2000, -2000, 3000, 3000, -4000, -4000))


def test_odd_chunk_boundaries_do_not_distort_the_audio() -> None:
    """The regression: a dropped byte shifted every later sample by one byte,
    which plays back as very loud noise instead of speech."""
    tone = array(
        "h", [int(12000 * math.sin(2 * math.pi * 220 * i / 48000)) for i in range(9600)]
    ).tobytes()

    source = PCMStreamAudioSource()
    position = 0
    sizes = (1023, 4097, 511, 2049, 1024)  # as ElevenLabs streams it: mostly odd
    for index in range(len(sizes) * 10):
        if position >= len(tone):
            break
        size = sizes[index % len(sizes)]
        source.feed_mono(tone[position : position + size])
        position += size
    source.close_input()

    played = drain(source)
    expected = mono_to_stereo(tone)

    assert played[: len(expected)] == expected
    # Byte-misaligned PCM16 is near full-scale noise; the level alone catches it.
    assert rms(played) == 0.0 or abs(rms(played) - rms(expected)) < 1.0


def test_stopping_clears_a_half_received_sample() -> None:
    """A barge-in must not leave half a sample to prepend to the next reply."""
    source = PCMStreamAudioSource()
    source.feed_mono(b"\x01")
    source.stop()

    restarted = PCMStreamAudioSource()
    restarted.feed_mono(mono(7))
    restarted.close_input()

    assert drain(restarted).startswith(mono(7, 7))


def test_read_ends_immediately_when_stopped_mid_playback() -> None:
    """Barge-in: buffered audio is dropped rather than played out."""
    source = PCMStreamAudioSource()
    source.feed(b"\x01" * FRAME_SIZE_BYTES * 10)
    source.stop()

    assert source.is_stopped
    assert source.read() == b""


def test_feeding_after_stop_is_ignored() -> None:
    source = PCMStreamAudioSource()
    source.stop()
    source.feed(b"\x01" * FRAME_SIZE_BYTES)

    assert source.read() == b""


def test_empty_input_ends_playback_without_emitting_a_frame() -> None:
    source = PCMStreamAudioSource()
    source.close_input()

    assert source.read() == b""


def test_read_emits_silence_while_waiting_for_more_audio() -> None:
    """A slow TTS stream must not stall the 20 ms player loop."""
    source = PCMStreamAudioSource()

    assert source.read() == SILENCE_FRAME

    source.feed(b"\x09" * FRAME_SIZE_BYTES)
    source.close_input()
    assert source.read() == b"\x09" * FRAME_SIZE_BYTES


def test_read_unblocks_as_soon_as_a_producer_thread_feeds_audio() -> None:
    source = PCMStreamAudioSource()
    payload = b"\x04" * FRAME_SIZE_BYTES

    def producer() -> None:
        source.feed(payload)

    thread = threading.Timer(0.02, producer)
    thread.start()
    try:
        frame = source.read()
        # Either the silence keep-alive or the real frame, then the real frame.
        if frame == SILENCE_FRAME:
            frame = source.read()
        assert frame == payload
    finally:
        thread.join()


def test_seconds_played_tracks_frames_handed_to_discord() -> None:
    source = PCMStreamAudioSource()
    source.feed(b"\x00" * FRAME_SIZE_BYTES * 50)
    source.close_input()

    for _ in range(50):
        source.read()

    assert source.seconds_played == 1.0  # 50 frames * 20 ms


def test_cleanup_stops_the_source() -> None:
    source = PCMStreamAudioSource()
    source.cleanup()
    assert source.is_stopped


def test_is_opus_is_false_so_discord_encodes_the_pcm() -> None:
    assert PCMStreamAudioSource().is_opus() is False
