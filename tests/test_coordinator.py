"""Failure handling in the per-user transcription coordinator.

The bug these cover: one failed utterance used to tear down the whole voice
session, so a single Gemini 429 made the bot leave the channel.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from src.models import AudioFrame, TranscriptSegment
from src.transcription.coordinator import (
    PerUserTranscriptionCoordinator,
    TranscriptionUnavailableError,
)


def frame(user_id: int, username: str, timestamp: float, *, samples: int = 320) -> AudioFrame:
    """One mono 16 kHz PCM16 frame with an explicit monotonic timestamp."""
    return AudioFrame(
        user_id=user_id,
        username=username,
        pcm_bytes=b"\x00\x01" * samples,
        sample_rate_hz=16000,
        channels=1,
        timestamp_monotonic=timestamp,
    )


class ScriptedSTT:
    """Stub STT: each call either raises or yields one final segment."""

    def __init__(self, *, failures: int = 0, error: Exception | None = None) -> None:
        self.remaining_failures = failures
        self.error = error or RuntimeError("429 RESOURCE_EXHAUSTED (simulated)")
        self.calls = 0
        self.fail_for_users: set[int] = set()

    def stream_recognize(
        self,
        *,
        audio_stream: AsyncIterator[bytes],
        user_id: int,
        username: str,
        sample_rate_hz: int,
        utterance_started_at: float,
    ) -> AsyncIterator[TranscriptSegment]:
        self.calls += 1
        should_fail = self.remaining_failures > 0 or user_id in self.fail_for_users
        if self.remaining_failures > 0:
            self.remaining_failures -= 1

        async def generate() -> AsyncIterator[TranscriptSegment]:
            async for _chunk in audio_stream:
                pass
            if should_fail:
                raise self.error
            yield TranscriptSegment(
                user_id=user_id,
                username=username,
                text=f"utterance {self.calls}",
                start_ts=utterance_started_at,
                end_ts=utterance_started_at + 1.0,
                is_final=True,
            )

        return generate()


def build_coordinator(
    stt: ScriptedSTT,
    segments: list[TranscriptSegment],
    *,
    max_consecutive_failures: int = 10,
    errors: list[tuple[BaseException, int]] | None = None,
) -> PerUserTranscriptionCoordinator:
    def on_error(exc: BaseException, consecutive: int) -> None:
        if errors is not None:
            errors.append((exc, consecutive))

    return PerUserTranscriptionCoordinator(
        stt_client=stt,  # type: ignore[arg-type]
        segment_handler=segments.append,
        idle_timeout_seconds=0.5,
        poll_interval_seconds=0.01,
        max_consecutive_failures=max_consecutive_failures,
        on_transcription_error=on_error if errors is not None else None,
    )


async def utter(
    coordinator: PerUserTranscriptionCoordinator,
    user_id: int,
    username: str,
    at: float,
) -> None:
    """Feed one frame and close the utterance, then let the worker finish."""
    await coordinator.process_frame(frame(user_id, username, at))
    await coordinator._close_session(user_id)
    await asyncio.sleep(0)
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# A failed utterance is dropped, not fatal
# ---------------------------------------------------------------------------


async def test_one_failed_utterance_keeps_the_coordinator_running() -> None:
    """The regression: a single 429 used to raise and end the voice session."""
    stt = ScriptedSTT(failures=1)
    segments: list[TranscriptSegment] = []
    coordinator = build_coordinator(stt, segments)

    await utter(coordinator, 1, "Daniel", 10.0)

    assert coordinator.failure_count == 1
    assert coordinator.consecutive_failure_count == 1
    # Still usable: the next utterance transcribes normally.
    await utter(coordinator, 1, "Daniel", 20.0)
    await coordinator.shutdown()

    assert [segment.text for segment in segments] == ["utterance 2"]


async def test_the_run_loop_survives_a_failed_utterance() -> None:
    """End to end through run(): the consumer task must not complete with an error."""
    stt = ScriptedSTT(failures=1)
    segments: list[TranscriptSegment] = []
    coordinator = build_coordinator(stt, segments)

    queue: asyncio.Queue[AudioFrame] = asyncio.Queue()
    stop_event = asyncio.Event()
    task = asyncio.create_task(coordinator.run(audio_queue=queue, stop_event=stop_event))

    queue.put_nowait(frame(1, "Daniel", 10.0))
    await asyncio.sleep(0.05)
    # A long gap closes the first utterance (which fails) and opens a second.
    queue.put_nowait(frame(1, "Daniel", 30.0))
    await asyncio.sleep(0.05)

    assert not task.done(), "one failed utterance must not stop the consumer"

    stop_event.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert coordinator.failure_count == 1
    assert [segment.text for segment in segments] == ["utterance 2"]


async def test_a_success_resets_the_consecutive_failure_counter() -> None:
    stt = ScriptedSTT(failures=2)
    segments: list[TranscriptSegment] = []
    coordinator = build_coordinator(stt, segments, max_consecutive_failures=4)

    await utter(coordinator, 1, "Daniel", 10.0)
    await utter(coordinator, 1, "Daniel", 20.0)
    assert coordinator.consecutive_failure_count == 2

    await utter(coordinator, 1, "Daniel", 30.0)
    assert coordinator.consecutive_failure_count == 0

    # Two more failures must not trip a limit of four now that the streak broke.
    stt.remaining_failures = 2
    await utter(coordinator, 1, "Daniel", 40.0)
    await utter(coordinator, 1, "Daniel", 50.0)
    await coordinator.shutdown()

    assert coordinator.failure_count == 4


async def test_one_speaker_failing_does_not_block_another() -> None:
    stt = ScriptedSTT()
    stt.fail_for_users = {1}
    segments: list[TranscriptSegment] = []
    coordinator = build_coordinator(stt, segments)

    await coordinator.process_frame(frame(1, "Daniel", 10.0))
    await coordinator.process_frame(frame(2, "Olena", 10.0))
    await coordinator._close_session(1)
    await coordinator._close_session(2)
    await coordinator.shutdown()

    assert coordinator.failure_count == 1
    assert [segment.username for segment in segments] == ["Olena"]


async def test_the_error_callback_reports_every_failure() -> None:
    stt = ScriptedSTT(failures=2)
    errors: list[tuple[BaseException, int]] = []
    coordinator = build_coordinator(stt, [], errors=errors)

    await utter(coordinator, 1, "Daniel", 10.0)
    await utter(coordinator, 1, "Daniel", 20.0)
    await coordinator.shutdown()

    assert [count for _exc, count in errors] == [1, 2]
    assert "RESOURCE_EXHAUSTED" in str(errors[0][0])


async def test_a_failing_error_callback_is_swallowed() -> None:
    """A broken observer must not resurrect the bug it was added to report."""
    stt = ScriptedSTT(failures=1)

    def explode(_exc: BaseException, _count: int) -> None:
        raise ValueError("handler is broken")

    coordinator = PerUserTranscriptionCoordinator(
        stt_client=stt,  # type: ignore[arg-type]
        segment_handler=lambda _segment: None,
        idle_timeout_seconds=0.5,
        on_transcription_error=explode,
    )

    await utter(coordinator, 1, "Daniel", 10.0)
    await coordinator.shutdown()

    assert coordinator.failure_count == 1


# ---------------------------------------------------------------------------
# ... but a backend that is really gone still stops the session
# ---------------------------------------------------------------------------


async def test_enough_consecutive_failures_stop_the_coordinator() -> None:
    stt = ScriptedSTT(failures=99)
    coordinator = build_coordinator(stt, [], max_consecutive_failures=3)

    queue: asyncio.Queue[AudioFrame] = asyncio.Queue()
    stop_event = asyncio.Event()
    task = asyncio.create_task(coordinator.run(audio_queue=queue, stop_event=stop_event))

    for index in range(3):
        queue.put_nowait(frame(1, "Daniel", 10.0 + index * 20.0))
        await asyncio.sleep(0.05)

    with pytest.raises(TranscriptionUnavailableError, match="3 times in a row"):
        await asyncio.wait_for(task, timeout=2.0)

    assert coordinator.failure_count == 3


async def test_the_give_up_error_keeps_the_original_cause() -> None:
    cause = RuntimeError("401 Unauthorized")
    stt = ScriptedSTT(failures=99, error=cause)
    coordinator = build_coordinator(stt, [], max_consecutive_failures=2)

    await utter(coordinator, 1, "Daniel", 10.0)
    await utter(coordinator, 1, "Daniel", 20.0)

    with pytest.raises(TranscriptionUnavailableError) as excinfo:
        coordinator._raise_if_unavailable()

    assert excinfo.value.__cause__ is cause
    await coordinator.shutdown()


async def test_the_failure_limit_is_never_below_one() -> None:
    """A zero would otherwise mean 'give up before anything has failed'."""
    coordinator = PerUserTranscriptionCoordinator(
        stt_client=ScriptedSTT(),  # type: ignore[arg-type]
        segment_handler=lambda _segment: None,
        max_consecutive_failures=0,
    )

    assert coordinator.max_consecutive_failures == 1
    coordinator._raise_if_unavailable()
