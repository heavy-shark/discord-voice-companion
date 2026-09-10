"""Tests for Gemini-backed speech-to-text."""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from src.models import TranscriptSegment
from src.transcription.gemini_stt import (
    DEFAULT_LANGUAGE_HINTS,
    LIVE_SAMPLE_RATE_HZ,
    GeminiSTTClient,
    GeminiSTTError,
    _clean_transcript,
    _duration_seconds,
    _pcm_to_wav,
)


def pcm(seconds: float, sample_rate_hz: int = LIVE_SAMPLE_RATE_HZ) -> bytes:
    """Silent mono PCM16 of the requested length."""
    return b"\x00\x00" * int(seconds * sample_rate_hz)


async def stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_clean_transcript_strips_quotes_and_whitespace() -> None:
    assert _clean_transcript('  "привіт"  ') == "привіт"
    assert _clean_transcript("'hello'") == "hello"


@pytest.mark.parametrize(
    "raw", ["", "   ", "...", "(silence)", "[NO SPEECH]", "no speech detected"]
)
def test_clean_transcript_treats_silence_markers_as_empty(raw: str) -> None:
    assert _clean_transcript(raw) == ""


def test_duration_seconds_uses_16_bit_mono_framing() -> None:
    assert _duration_seconds(2 * 16000, 16000) == pytest.approx(1.0)
    assert _duration_seconds(0, 16000) == 0.0
    assert _duration_seconds(1000, 0) == 0.0


def test_pcm_to_wav_produces_a_readable_mono_container() -> None:
    payload = pcm(0.5)
    wav_bytes = _pcm_to_wav(payload, LIVE_SAMPLE_RATE_HZ)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == LIVE_SAMPLE_RATE_HZ
        assert wav.readframes(wav.getnframes()) == payload


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_api_key_is_rejected_with_a_pointer_to_the_setting() -> None:
    with pytest.raises(GeminiSTTError, match="GOOGLE_GEMINI_API_KEY"):
        GeminiSTTClient("   ")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(GeminiSTTError, match="expected 'live' or 'batch'"):
        GeminiSTTClient("key", backend="whisper", client=object())


def test_model_name_follows_the_selected_backend() -> None:
    live = GeminiSTTClient("key", backend="live", live_model="live-x", client=object())
    batch = GeminiSTTClient("key", backend="batch", batch_model="batch-y", client=object())
    assert live.model_name == "live-x"
    assert batch.model_name == "batch-y"


def test_language_hints_default_to_ukrainian_russian_english() -> None:
    client = GeminiSTTClient("key", client=object())
    assert client.language_hints == DEFAULT_LANGUAGE_HINTS
    assert set(client.language_hints) == {"uk-UA", "ru-RU", "en-US"}


def test_transcription_config_always_enables_auto_detection() -> None:
    client = GeminiSTTClient("key", client=object())
    config = client._transcription_config()
    # An empty language_codes list is what turns on automatic detection; the
    # hints only bias it.
    assert config.language_codes == []
    assert config.language_hints.language_codes == list(DEFAULT_LANGUAGE_HINTS)


def test_transcription_config_drops_hints_once_the_server_rejects_them() -> None:
    client = GeminiSTTClient("key", client=object())
    assert client._maybe_disable_language_hints(ValueError("Unknown field language_hints"))
    assert client._transcription_config().language_hints is None
    # Only the first rejection flips the flag.
    assert not client._maybe_disable_language_hints(ValueError("language_hints"))


def test_unrelated_errors_do_not_disable_language_hints() -> None:
    client = GeminiSTTClient("key", client=object())
    assert not client._maybe_disable_language_hints(ValueError("quota exceeded"))
    hints = client._transcription_config().language_hints
    assert hints.language_codes == list(DEFAULT_LANGUAGE_HINTS)


# ---------------------------------------------------------------------------
# Batch backend
# ---------------------------------------------------------------------------


class FakeBatchModels:
    def __init__(self, text: str = "привіт", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None and len(self.calls) == 1:
            raise self.error
        return SimpleNamespace(text=self.text, candidates=[])


def fake_batch_client(models: FakeBatchModels) -> SimpleNamespace:
    return SimpleNamespace(aio=SimpleNamespace(models=models))


async def collect(client: GeminiSTTClient, audio: bytes) -> list[TranscriptSegment]:
    return [
        segment
        async for segment in client.stream_recognize(
            audio_stream=stream(audio),
            user_id=7,
            username="Oleksii",
            sample_rate_hz=LIVE_SAMPLE_RATE_HZ,
            utterance_started_at=100.0,
        )
    ]


async def test_batch_backend_returns_one_final_segment() -> None:
    models = FakeBatchModels(text="як справи")
    client = GeminiSTTClient("key", backend="batch", client=fake_batch_client(models))

    segments = await collect(client, pcm(1.0))

    assert len(segments) == 1
    segment = segments[0]
    assert segment.text == "як справи"
    assert segment.is_final
    assert segment.user_id == 7
    assert segment.username == "Oleksii"
    assert segment.start_ts == 100.0
    assert segment.end_ts == pytest.approx(101.0)


async def test_batch_backend_sends_wav_audio_and_the_configured_model() -> None:
    models = FakeBatchModels()
    client = GeminiSTTClient(
        "key", backend="batch", batch_model="gemini-test", client=fake_batch_client(models)
    )

    await collect(client, pcm(1.0))

    call = models.calls[0]
    assert call["model"] == "gemini-test"
    audio_part = call["contents"][1]
    assert audio_part.inline_data.mime_type == "audio/wav"
    assert audio_part.inline_data.data.startswith(b"RIFF")


async def test_batch_backend_skips_utterances_below_the_length_floor() -> None:
    models = FakeBatchModels()
    client = GeminiSTTClient("key", backend="batch", client=fake_batch_client(models))

    segments = await collect(client, pcm(0.1))

    assert segments == []
    assert models.calls == []


async def test_batch_backend_yields_nothing_when_no_speech_was_heard() -> None:
    models = FakeBatchModels(text="(silence)")
    client = GeminiSTTClient("key", backend="batch", client=fake_batch_client(models))

    assert await collect(client, pcm(1.0)) == []


async def test_batch_backend_retries_without_hints_when_they_are_rejected() -> None:
    models = FakeBatchModels(text="ok", error=ValueError("unknown field language_hints"))
    client = GeminiSTTClient("key", backend="batch", client=fake_batch_client(models))

    segments = await collect(client, pcm(1.0))

    assert [segment.text for segment in segments] == ["ok"]
    assert len(models.calls) == 2
    first, second = models.calls
    assert first["config"].audio_transcription_config.language_hints is not None
    assert second["config"].audio_transcription_config.language_hints is None


async def test_batch_backend_surfaces_other_failures() -> None:
    models = FakeBatchModels(error=RuntimeError("quota exceeded"))
    client = GeminiSTTClient("key", backend="batch", client=fake_batch_client(models))

    with pytest.raises(GeminiSTTError, match="quota exceeded"):
        await collect(client, pcm(1.0))


# ---------------------------------------------------------------------------
# Live backend
# ---------------------------------------------------------------------------


def transcription(text: str, *, finished: bool = False, language: str | None = None):
    return SimpleNamespace(text=text, finished=finished, language_code=language)


def server_message(*, interim=None, final=None, turn_complete: bool = False):
    return SimpleNamespace(
        server_content=SimpleNamespace(
            interim_input_transcription=interim,
            input_transcription=final,
            turn_complete=turn_complete,
        )
    )


class FakeLiveSession:
    def __init__(self, messages: list) -> None:
        self._messages = messages
        self.sent_audio: list[bytes] = []
        self.stream_ended = False

    async def send_realtime_input(self, *, audio=None, audio_stream_end=None) -> None:
        if audio is not None:
            self.sent_audio.append(audio.data)
        if audio_stream_end:
            self.stream_ended = True

    async def receive(self):
        for message in self._messages:
            await asyncio.sleep(0)
            yield message


class FakeLiveConnect:
    def __init__(self, session: FakeLiveSession) -> None:
        self.session = session
        self.kwargs: dict = {}

    def connect(self, **kwargs):
        self.kwargs = kwargs
        session = self.session

        class _Ctx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *exc_info):
                return False

        return _Ctx()


def fake_live_client(session: FakeLiveSession) -> tuple[SimpleNamespace, FakeLiveConnect]:
    live = FakeLiveConnect(session)
    return SimpleNamespace(aio=SimpleNamespace(live=live)), live


async def test_live_backend_emits_interim_then_final() -> None:
    session = FakeLiveSession(
        [
            server_message(interim=transcription("прив")),
            server_message(final=transcription("привіт", finished=True, language="uk-UA")),
        ]
    )
    client_obj, _ = fake_live_client(session)
    client = GeminiSTTClient("key", backend="live", client=client_obj)

    segments = await collect(client, pcm(1.0))

    assert [(s.text, s.is_final) for s in segments] == [("прив", False), ("привіт", True)]
    assert segments[-1].language_code == "uk-UA"


async def test_live_backend_can_suppress_interim_results() -> None:
    session = FakeLiveSession(
        [
            server_message(interim=transcription("par")),
            server_message(final=transcription("partial", finished=True)),
        ]
    )
    client_obj, _ = fake_live_client(session)
    client = GeminiSTTClient("key", backend="live", emit_interim=False, client=client_obj)

    segments = await collect(client, pcm(1.0))

    assert [s.text for s in segments] == ["partial"]
    assert all(s.is_final for s in segments)


async def test_live_backend_streams_audio_and_closes_the_turn() -> None:
    session = FakeLiveSession([server_message(final=transcription("hi", finished=True))])
    client_obj, connect = fake_live_client(session)
    client = GeminiSTTClient("key", backend="live", live_model="live-model", client=client_obj)

    await collect(client, pcm(0.5))
    # The pump task closes the turn after the generator is exhausted.
    await asyncio.sleep(0)

    assert connect.kwargs["model"] == "live-model"
    assert connect.kwargs["config"].input_audio_transcription is not None
    assert b"".join(session.sent_audio) == pcm(0.5)


async def test_live_backend_rejects_a_mismatched_sample_rate() -> None:
    session = FakeLiveSession([])
    client_obj, _ = fake_live_client(session)
    client = GeminiSTTClient("key", backend="live", client=client_obj)

    with pytest.raises(GeminiSTTError, match="16000 Hz"):
        async for _ in client.stream_recognize(
            audio_stream=stream(pcm(0.5, 8000)),
            user_id=1,
            username="user",
            sample_rate_hz=8000,
            utterance_started_at=0.0,
        ):
            pass


async def test_live_backend_stops_on_turn_complete_without_a_transcript() -> None:
    session = FakeLiveSession([server_message(turn_complete=True)])
    client_obj, _ = fake_live_client(session)
    client = GeminiSTTClient("key", backend="live", client=client_obj)

    assert await collect(client, pcm(1.0)) == []
