"""Speech-to-text backed by Gemini.

Two interchangeable backends, both exposing the ``stream_recognize`` shape that
:class:`~src.transcription.coordinator.PerUserTranscriptionCoordinator` drives:

``live``
    Streams the utterance into ``gemini-3.5-transcribe-live`` over the Live API
    WebSocket and yields interim hypotheses plus a final transcript. Lowest
    latency, and the only backend that produces partial text.

``batch``
    Buffers the whole utterance, wraps it in a WAV container and sends one
    ``generate_content`` call. Slower (no text until the speaker stops) but it
    works with any audio-capable Gemini model, so it is the fallback when the
    live transcription model is not available on an API key.

Both backends receive 16 kHz mono little-endian PCM16, which is what the
coordinator already produces via ``convert_to_linear16``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import suppress
from typing import Any

from google import genai
from google.genai import types

from ..models import TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_LIVE_MODEL = "gemini-3.5-transcribe-live"
DEFAULT_BATCH_MODEL = "gemini-2.5-flash"
DEFAULT_LANGUAGE_HINTS: tuple[str, ...] = ("uk-UA", "ru-RU", "en-US")

# The Live API expects 16 kHz mono PCM16; the coordinator resamples to match.
LIVE_SAMPLE_RATE_HZ = 16000

# Utterances shorter than this are almost always a door slam or a cough. Sending
# them wastes a request and tends to come back as hallucinated filler text.
_MIN_UTTERANCE_SECONDS = 0.35

_BATCH_PROMPT = (
    "Transcribe the speech in this audio verbatim. "
    "Return only the transcript text, with no commentary, no speaker labels, "
    "and no quotation marks. If there is no intelligible speech, return an "
    "empty string."
)

# Things a transcription model says when handed silence. Treated as "nothing was
# said" rather than as user speech.
_EMPTY_TRANSCRIPT_MARKERS = frozenset(
    {
        "",
        "-",
        "...",
        "(silence)",
        "[silence]",
        "(no speech)",
        "[no speech]",
        "(inaudible)",
        "[inaudible]",
        "no speech detected",
    }
)


class GeminiSTTError(RuntimeError):
    """Raised when Gemini transcription cannot be used with the current config."""


def _clean_transcript(text: str) -> str:
    """Strip surrounding quotes/whitespace and drop 'nothing was said' markers."""
    cleaned = (text or "").strip().strip('"').strip("'").strip()
    if cleaned.lower() in _EMPTY_TRANSCRIPT_MARKERS:
        return ""
    return cleaned


def _pcm_to_wav(pcm_bytes: bytes, sample_rate_hz: int) -> bytes:
    """Wrap raw mono PCM16 in a WAV container so Gemini can sniff the format."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm_bytes)
    return buffer.getvalue()


def _duration_seconds(byte_count: int, sample_rate_hz: int) -> float:
    """Seconds of mono PCM16 represented by *byte_count* bytes."""
    if sample_rate_hz <= 0:
        return 0.0
    return byte_count / (2.0 * float(sample_rate_hz))


class GeminiSTTClient:
    """Transcribe per-user Discord utterances with Gemini."""

    def __init__(
        self,
        api_key: str,
        *,
        backend: str = "live",
        live_model: str = DEFAULT_LIVE_MODEL,
        batch_model: str = DEFAULT_BATCH_MODEL,
        language_hints: Sequence[str] | None = None,
        emit_interim: bool = True,
        client: Any | None = None,
    ) -> None:
        key = (api_key or "").strip().strip('"').strip("'")
        if not key and client is None:
            raise GeminiSTTError(
                "No Gemini API key. Set GOOGLE_GEMINI_API_KEY in .env or the "
                "Gemini key in the settings editor."
            )

        normalized_backend = (backend or "live").strip().lower()
        if normalized_backend not in ("live", "batch"):
            raise GeminiSTTError(
                f"Unknown STT backend {backend!r}; expected 'live' or 'batch'."
            )

        self.backend = normalized_backend
        self.live_model = (live_model or DEFAULT_LIVE_MODEL).strip()
        self.batch_model = (batch_model or DEFAULT_BATCH_MODEL).strip()
        self.language_hints: tuple[str, ...] = tuple(
            DEFAULT_LANGUAGE_HINTS if language_hints is None else language_hints
        )
        self.emit_interim = emit_interim

        # ``client`` is an injection seam for tests; production always builds one.
        self._client = client if client is not None else genai.Client(api_key=key)
        # Set once the server rejects the optional biasing fields, so we stop
        # re-sending them for every subsequent utterance.
        self._drop_language_hints = False

        logger.info(
            "GeminiSTTClient initialized (backend=%s, model=%s, hints=%s)",
            self.backend,
            self.model_name,
            ",".join(self.language_hints) or "auto",
        )

    @property
    def model_name(self) -> str:
        """The model this client will actually call."""
        return self.live_model if self.backend == "live" else self.batch_model

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    async def validate_connectivity(self) -> None:
        """Fail fast with an actionable message if the model is not usable."""
        try:
            available = [
                (model.name or "").removeprefix("models/")
                async for model in await self._client.aio.models.list()
            ]
        except Exception as exc:  # network/auth problems surface here
            raise GeminiSTTError(
                f"Could not reach the Gemini API: {exc}. Check GOOGLE_GEMINI_API_KEY "
                "and network access."
            ) from exc

        if self.model_name in available:
            logger.info("Gemini STT model %s is available.", self.model_name)
            return

        suggestions = [name for name in available if "transcribe" in name or "flash" in name]
        raise GeminiSTTError(
            f"Gemini model {self.model_name!r} is not available for this API key. "
            f"Candidates on this key: {', '.join(sorted(suggestions)[:12]) or 'none'}. "
            "Set stt.live_model / stt.batch_model in settings.json (or GEMINI_STT_MODEL "
            "in .env) to one of them, or switch stt.backend to 'batch'."
        )

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    async def stream_recognize(
        self,
        *,
        audio_stream: AsyncIterator[bytes],
        user_id: int,
        username: str,
        sample_rate_hz: int,
        utterance_started_at: float,
    ) -> AsyncIterator[TranscriptSegment]:
        """Yield transcript segments for one user's utterance."""
        if self.backend == "live":
            iterator = self._stream_recognize_live(
                audio_stream=audio_stream,
                user_id=user_id,
                username=username,
                sample_rate_hz=sample_rate_hz,
                utterance_started_at=utterance_started_at,
            )
        else:
            iterator = self._stream_recognize_batch(
                audio_stream=audio_stream,
                user_id=user_id,
                username=username,
                sample_rate_hz=sample_rate_hz,
                utterance_started_at=utterance_started_at,
            )
        async for segment in iterator:
            yield segment

    def _transcription_config(self) -> types.AudioTranscriptionConfig:
        """Build the transcription config, minus any field the server rejected."""
        if self._drop_language_hints or not self.language_hints:
            # An empty language_codes list is what enables automatic detection.
            return types.AudioTranscriptionConfig(language_codes=[])
        return types.AudioTranscriptionConfig(
            language_codes=[],
            language_hints=types.LanguageHints(language_codes=list(self.language_hints)),
        )

    async def _stream_recognize_live(
        self,
        *,
        audio_stream: AsyncIterator[bytes],
        user_id: int,
        username: str,
        sample_rate_hz: int,
        utterance_started_at: float,
    ) -> AsyncIterator[TranscriptSegment]:
        if sample_rate_hz != LIVE_SAMPLE_RATE_HZ:
            raise GeminiSTTError(
                f"Live transcription needs {LIVE_SAMPLE_RATE_HZ} Hz mono PCM, "
                f"got {sample_rate_hz} Hz."
            )

        config = types.LiveConnectConfig(
            # Transcription only: we never want the live model generating replies,
            # that is the chat session's job.
            response_modalities=[types.Modality.TEXT],
            input_audio_transcription=self._transcription_config(),
        )
        mime_type = f"audio/pcm;rate={sample_rate_hz}"

        try:
            session_ctx = self._client.aio.live.connect(model=self.live_model, config=config)
        except Exception as exc:
            raise GeminiSTTError(f"Could not open a Gemini Live session: {exc}") from exc

        streamed_bytes = 0
        final_emitted = False

        async with session_ctx as session:

            async def pump_audio() -> None:
                """Forward the utterance, then tell the server the turn is over."""
                nonlocal streamed_bytes
                try:
                    async for chunk in audio_stream:
                        if not chunk:
                            continue
                        streamed_bytes += len(chunk)
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type=mime_type)
                        )
                finally:
                    with suppress(Exception):
                        await session.send_realtime_input(audio_stream_end=True)

            pump_task = asyncio.create_task(
                pump_audio(), name=f"gemini-stt-send-{user_id}"
            )

            try:
                async for message in session.receive():
                    content = getattr(message, "server_content", None)
                    if content is None:
                        continue

                    interim = getattr(content, "interim_input_transcription", None)
                    if self.emit_interim and interim is not None:
                        text = _clean_transcript(getattr(interim, "text", "") or "")
                        if text:
                            yield TranscriptSegment(
                                user_id=user_id,
                                username=username,
                                text=text,
                                start_ts=utterance_started_at,
                                end_ts=utterance_started_at
                                + _duration_seconds(streamed_bytes, sample_rate_hz),
                                is_final=False,
                                language_code=getattr(interim, "language_code", None),
                            )

                    final = getattr(content, "input_transcription", None)
                    if final is not None:
                        text = _clean_transcript(getattr(final, "text", "") or "")
                        if text:
                            final_emitted = True
                            yield TranscriptSegment(
                                user_id=user_id,
                                username=username,
                                text=text,
                                start_ts=utterance_started_at,
                                end_ts=utterance_started_at
                                + _duration_seconds(streamed_bytes, sample_rate_hz),
                                is_final=True,
                                language_code=getattr(final, "language_code", None),
                            )
                        if getattr(final, "finished", False):
                            break

                    if getattr(content, "turn_complete", False):
                        break
            except Exception as exc:
                if self._maybe_disable_language_hints(exc):
                    logger.warning(
                        "Gemini rejected the language hints; retrying without them."
                    )
                raise
            finally:
                pump_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pump_task

        if not final_emitted:
            logger.debug(
                "Live transcription produced no final text for %s (%s bytes streamed).",
                username,
                streamed_bytes,
            )

    async def _stream_recognize_batch(
        self,
        *,
        audio_stream: AsyncIterator[bytes],
        user_id: int,
        username: str,
        sample_rate_hz: int,
        utterance_started_at: float,
    ) -> AsyncIterator[TranscriptSegment]:
        chunks: list[bytes] = []
        async for chunk in audio_stream:
            if chunk:
                chunks.append(chunk)

        pcm_bytes = b"".join(chunks)
        duration = _duration_seconds(len(pcm_bytes), sample_rate_hz)
        if duration < _MIN_UTTERANCE_SECONDS:
            logger.debug(
                "Skipping %.2fs utterance from %s (below %.2fs floor).",
                duration,
                username,
                _MIN_UTTERANCE_SECONDS,
            )
            return

        wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate_hz)
        config = types.GenerateContentConfig(
            audio_transcription_config=self._transcription_config(),
            temperature=0.0,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self.batch_model,
                contents=[
                    _BATCH_PROMPT,
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                ],
                config=config,
            )
        except Exception as exc:
            if self._maybe_disable_language_hints(exc):
                logger.warning(
                    "Gemini rejected the language hints; retrying this utterance without them."
                )
                response = await self._client.aio.models.generate_content(
                    model=self.batch_model,
                    contents=[
                        _BATCH_PROMPT,
                        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    ],
                    config=types.GenerateContentConfig(
                        audio_transcription_config=self._transcription_config(),
                        temperature=0.0,
                    ),
                )
            else:
                raise GeminiSTTError(
                    f"Gemini batch transcription failed on model {self.batch_model!r}: {exc}"
                ) from exc

        text = _clean_transcript(getattr(response, "text", "") or "")
        if not text:
            logger.debug("Batch transcription returned no speech for %s.", username)
            return

        yield TranscriptSegment(
            user_id=user_id,
            username=username,
            text=text,
            start_ts=utterance_started_at,
            end_ts=utterance_started_at + duration,
            is_final=True,
            language_code=_first_response_language(response),
        )

    def _maybe_disable_language_hints(self, exc: BaseException) -> bool:
        """Drop the optional biasing fields once, if the server complains about them."""
        if self._drop_language_hints or not self.language_hints:
            return False
        message = str(exc).lower()
        if "language_hints" in message or "language hints" in message:
            self._drop_language_hints = True
            return True
        return False


def _first_response_language(response: Any) -> str | None:
    """Best-effort language code from a ``generate_content`` transcription response."""
    candidates: Iterable[Any] = getattr(response, "candidates", None) or ()
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or ():
            language = getattr(part, "language_code", None)
            if language:
                return str(language)
    return None
