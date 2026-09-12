"""Per-user utterance coordination for streaming speech-to-text."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from ..models import AudioFrame, TranscriptSegment
from .gemini_stt import GeminiSTTClient
from .preprocessing import convert_to_linear16

logger = logging.getLogger(__name__)

SegmentHandler = Callable[[TranscriptSegment], object]
SttAudioSecondsHandler = Callable[[float], None]
SpeechStartHandler = Callable[[int, str], None]
# (error, consecutive failure count) - fired once per dropped utterance.
TranscriptionErrorHandler = Callable[[BaseException, int], None]

# One transient 429 must never end a voice session, but a backend that is really
# gone should not be retried forever either.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10


class TranscriptionUnavailableError(RuntimeError):
    """Raised when enough utterances fail in a row that the STT backend looks down."""


@dataclass(slots=True)
class _UtteranceSession:
    """Track one active STT streaming session for a single user."""

    user_id: int
    username: str
    sample_rate_hz: int
    utterance_started_at: float
    last_frame_monotonic: float
    audio_queue: asyncio.Queue[bytes | None]
    task: asyncio.Task[None]


class PerUserTranscriptionCoordinator:
    """Split incoming PCM frames into per-user streaming utterances."""

    def __init__(
        self,
        stt_client: GeminiSTTClient,
        segment_handler: SegmentHandler,
        *,
        idle_timeout_seconds: float = 0.5,
        poll_interval_seconds: float = 0.1,
        target_sample_rate_hz: int = 16000,
        on_stt_audio_seconds: SttAudioSecondsHandler | None = None,
        on_speech_started: SpeechStartHandler | None = None,
        on_transcription_error: TranscriptionErrorHandler | None = None,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ) -> None:
        self.stt_client = stt_client
        self.segment_handler = segment_handler
        self.idle_timeout_seconds = idle_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.target_sample_rate_hz = target_sample_rate_hz
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self._on_stt_audio_seconds = on_stt_audio_seconds
        self._on_speech_started = on_speech_started
        self._on_transcription_error = on_transcription_error
        self._active_sessions: dict[int, _UtteranceSession] = {}
        self._all_tasks: set[asyncio.Task[None]] = set()
        self._consecutive_failures = 0
        self._failure_count = 0
        self._unavailable_cause: BaseException | None = None

    @property
    def active_user_count(self) -> int:
        """Return the number of users with active utterances."""
        return len(self._active_sessions)

    @property
    def consecutive_failure_count(self) -> int:
        """Utterances that have failed in a row; reset by the next success."""
        return self._consecutive_failures

    @property
    def failure_count(self) -> int:
        """Total utterances dropped because transcription raised."""
        return self._failure_count

    async def run(
        self,
        audio_queue: asyncio.Queue[AudioFrame],
        stop_event: asyncio.Event,
    ) -> None:
        """Consume audio frames until the caller signals shutdown."""
        try:
            while True:
                self._raise_if_unavailable()
                if stop_event.is_set() and audio_queue.empty():
                    break

                try:
                    frame = await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=self.poll_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    await self._close_idle_sessions()
                    continue

                await self.process_frame(frame)
                await self._close_idle_sessions(reference_monotonic=frame.timestamp_monotonic)
        finally:
            await self.shutdown()
            self._raise_if_unavailable()

    async def process_frame(self, frame: AudioFrame) -> None:
        """Route one PCM frame into the correct per-user utterance stream."""
        session = self._active_sessions.get(frame.user_id)
        if session is not None:
            idle_gap = frame.timestamp_monotonic - session.last_frame_monotonic
            if idle_gap >= self.idle_timeout_seconds:
                await self._close_session(frame.user_id)
                session = None

        if session is None:
            session = self._start_session(frame)

        session.last_frame_monotonic = frame.timestamp_monotonic
        mono_bytes = convert_to_linear16(
            frame.pcm_bytes,
            from_sample_rate=frame.sample_rate_hz,
            to_sample_rate=self.target_sample_rate_hz,
            channels=frame.channels,
        )
        if mono_bytes:
            session.audio_queue.put_nowait(mono_bytes)
            if self._on_stt_audio_seconds is not None:
                delta = len(mono_bytes) / (2.0 * float(self.target_sample_rate_hz))
                if delta > 0:
                    self._on_stt_audio_seconds(delta)

    async def shutdown(self) -> None:
        """Close active STT sessions and wait for all background tasks."""
        for user_id in list(self._active_sessions):
            await self._close_session(user_id)

        if not self._all_tasks:
            return

        await asyncio.gather(*list(self._all_tasks))

    async def _close_idle_sessions(self, reference_monotonic: float | None = None) -> None:
        now = time.monotonic() if reference_monotonic is None else reference_monotonic
        stale_users = [
            user_id
            for user_id, session in self._active_sessions.items()
            if now - session.last_frame_monotonic >= self.idle_timeout_seconds
        ]
        for user_id in stale_users:
            await self._close_session(user_id)

    def _start_session(self, frame: AudioFrame) -> _UtteranceSession:
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        utterance_started_at = time.time()
        task = asyncio.create_task(
            self._consume_utterance(
                user_id=frame.user_id,
                username=frame.username,
                sample_rate_hz=self.target_sample_rate_hz,
                utterance_started_at=utterance_started_at,
                audio_queue=audio_queue,
            ),
            name=f"stt-utterance-{frame.user_id}",
        )
        task.add_done_callback(self._all_tasks.discard)
        self._all_tasks.add(task)

        session = _UtteranceSession(
            user_id=frame.user_id,
            username=frame.username,
            sample_rate_hz=self.target_sample_rate_hz,
            utterance_started_at=utterance_started_at,
            last_frame_monotonic=frame.timestamp_monotonic,
            audio_queue=audio_queue,
            task=task,
        )
        self._active_sessions[frame.user_id] = session
        logger.debug("Started STT utterance for %s (%s)", frame.username, frame.user_id)
        if self._on_speech_started is not None:
            # Fires as soon as someone starts talking, well before their words are
            # transcribed - early enough to cut the bot off mid-sentence.
            try:
                self._on_speech_started(frame.user_id, frame.username)
            except Exception:
                logger.warning("on_speech_started handler failed", exc_info=True)
        return session

    async def _close_session(self, user_id: int) -> None:
        session = self._active_sessions.pop(user_id, None)
        if session is None:
            return

        session.audio_queue.put_nowait(None)
        logger.debug("Closed STT utterance for %s (%s)", session.username, session.user_id)

    async def _consume_utterance(
        self,
        *,
        user_id: int,
        username: str,
        sample_rate_hz: int,
        utterance_started_at: float,
        audio_queue: asyncio.Queue[bytes | None],
    ) -> None:
        async def audio_stream() -> AsyncIterator[bytes]:
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                yield chunk

        try:
            async for segment in self.stt_client.stream_recognize(
                audio_stream=audio_stream(),
                user_id=user_id,
                username=username,
                sample_rate_hz=sample_rate_hz,
                utterance_started_at=utterance_started_at,
            ):
                handler_result = self.segment_handler(segment)
                if inspect.isawaitable(handler_result):
                    await handler_result
        except Exception as exc:
            # One bad utterance (a Gemini 429, a dropped socket) is expected noise:
            # drop it and keep listening. Only a long unbroken streak of failures
            # means the backend is really gone, and only then do we give up.
            self._record_utterance_failure(exc, user_id=user_id, username=username)
        else:
            self._consecutive_failures = 0

    def _record_utterance_failure(
        self,
        exc: BaseException,
        *,
        user_id: int,
        username: str,
    ) -> None:
        self._failure_count += 1
        self._consecutive_failures += 1
        logger.error(
            "STT utterance failed for %s (%s); dropping it. %d failure(s) in a row, "
            "giving up at %d.",
            username,
            user_id,
            self._consecutive_failures,
            self.max_consecutive_failures,
            exc_info=exc,
        )

        if self._on_transcription_error is not None:
            try:
                self._on_transcription_error(exc, self._consecutive_failures)
            except Exception:
                logger.warning("on_transcription_error handler failed", exc_info=True)

        if self._consecutive_failures >= self.max_consecutive_failures and self._unavailable_cause is None:
            self._unavailable_cause = exc

    def _raise_if_unavailable(self) -> None:
        cause = self._unavailable_cause
        if cause is None:
            return
        raise TranscriptionUnavailableError(
            f"Transcription failed {self._consecutive_failures} times in a row "
            f"(last error: {cause}); the speech-to-text backend looks unavailable."
        ) from cause
