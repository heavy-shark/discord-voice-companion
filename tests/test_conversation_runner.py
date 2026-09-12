"""Failure handling in the voice conversation runner.

The bug these cover: a single Gemini 429 or ElevenLabs 401 on the reply path used
to kill the supervisor task, which unwound the whole session and made the bot
leave the channel.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from src.conversation import TriggerReason
from src.models import Persona
from src.session import conversation_runner as runner_module
from src.session.conversation_runner import (
    ConversationRunnerConfig,
    ReplyPipelineError,
    run_voice_conversation,
)
from src.session.metrics import SessionMetrics

SUPERVISOR_TICK = 0.25


def persona() -> Persona:
    return Persona(
        persona_id="default",
        display_name="Companion",
        system_instruction="Be brief.",
        genai_model="gemini-2.5-flash",
        elevenlabs_voice_id="voice1",
    )


def runner_config(*, greet_on_join: bool = False) -> ConversationRunnerConfig:
    return ConversationRunnerConfig(
        stt_idle_timeout_seconds=1.2,
        mention_window_seconds=30.0,
        greet_on_join=greet_on_join,
        reply_locale="auto",
        talkativeness=65,
        spoken_languages=("uk-UA",),
    )


class FakeSettingsStore:
    def get_personalities(self) -> list[Persona]:
        return [persona()]

    def get_ui_preferences(self) -> dict[str, object]:
        return {}

    def resolve_api_secret(self, _key: str, *_env_names: str) -> str:
        return "fake-gemini-key"

    def resolve_stt_backend(self) -> str:
        return "batch"

    def resolve_stt_live_model(self) -> str:
        return "live-model"

    def resolve_stt_batch_model(self) -> str:
        return "batch-model"

    def resolve_stt_language_hints(self) -> list[str]:
        return ["uk-UA"]

    def resolve_stt_emit_interim(self) -> bool:
        return False


class FakeDiscordClient:
    def __init__(self) -> None:
        self.listening = False
        self.sinks: list[object] = []

    async def start_listening(self, sink: object) -> None:
        self.listening = True
        self.sinks.append(sink)

    async def stop_listening(self) -> None:
        self.listening = False

    def is_listening(self) -> bool:
        return self.listening

    def set_voice_join_handler(self, _channel_id, _handler) -> None:
        return None


class FakeChatManager:
    """Chat manager whose replies can be scripted to fail."""

    def __init__(self, *, failures: int = 0, error: Exception | None = None, **_kwargs) -> None:
        self.remaining_failures = failures
        self.error = error or RuntimeError("429 RESOURCE_EXHAUSTED")
        self.replies: list[str] = []
        self.sessions_created = 0

    async def create_chat_session(self, _persona: Persona) -> None:
        self.sessions_created += 1

    async def send_message(self, prompt: str) -> str:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise self.error
        self.replies.append(prompt)
        return f"reply {len(self.replies)}"

    async def should_volunteer(self, _transcript: str) -> bool:
        return True

    def format_conversation_prompt(self, transcript: str) -> str:
        return transcript

    def get_token_usage_breakdown(self) -> tuple[int, int]:
        return (0, 0)


class FakeTTSClient:
    def __init__(self, _api_key: str, *, failures: int = 0, error: Exception | None = None) -> None:
        self.remaining_failures = failures
        self.error = error or RuntimeError("401 Unauthorized: quota exhausted")
        self.spoken: list[str] = []

    def stream_pcm(self, text: str, _persona: Persona) -> AsyncIterator[bytes]:
        async def generate() -> AsyncIterator[bytes]:
            if self.remaining_failures > 0:
                self.remaining_failures -= 1
                raise self.error
            self.spoken.append(text)
            yield b"\x00\x00"

        return generate()

    def record_tts_duration(self, _seconds: float) -> None:
        return None

    def record_tts_characters(self, _count: int) -> None:
        return None


class FakePlaybackManager:
    def __init__(self, _voice_client: object) -> None:
        self.played = 0

    async def play_mono_pcm_stream(self, stream: AsyncIterator[bytes]) -> float:
        async for _chunk in stream:
            self.played += 1
        return 0.1

    def is_playing(self) -> bool:
        return False

    async def stop_playback(self) -> None:
        return None


class AlwaysReplyPolicy:
    """Policy stub: every tick asks for a reply, so tests do not wait on timers."""

    def __init__(self, **_kwargs) -> None:
        self.declined = 0
        self.bot_replies = 0
        self.reason = TriggerReason.ADDRESSED

    def set_mention_triggers(self, _names: list[str]) -> None:
        return None

    def text_contains_mention_trigger(self, _text: str) -> bool:
        return False

    def record_human_speech(self) -> None:
        return None

    def record_mention(self) -> None:
        return None

    def record_bot_reply(self) -> None:
        self.bot_replies += 1

    def record_declined_reply(self) -> None:
        self.declined += 1

    def evaluate(self, *, has_pending_transcript: bool) -> object:
        return SimpleNamespace(should_reply=True, reason=self.reason)

    def snapshot_for_ui(self) -> tuple[bool, float, float]:
        return (False, 0.0, 0.0)


class FakeTranscriptWriter:
    def __init__(self, **_kwargs) -> None:
        self.path = "memory"
        self.bot_replies: list[tuple[str, str]] = []

    def add_segment(self, _segment: object) -> None:
        return None

    def add_bot_reply(self, text: str, *, label: str = "") -> None:
        self.bot_replies.append((text, label))

    def add_stt_seconds(self, _seconds: float) -> None:
        return None

    def add_tts_seconds(self, _seconds: float) -> None:
        return None

    def add_tts_characters(self, _count: int) -> None:
        return None

    def set_genai_token_counts(self, _inp: int, _out: int) -> None:
        return None


class FakeSTTClient:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def validate_connectivity(self) -> None:
        return None

    def stream_recognize(self, **_kwargs: object) -> AsyncIterator[object]:
        async def generate() -> AsyncIterator[object]:
            return
            yield  # pragma: no cover - never reached, keeps this an async generator

        return generate()


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chat_manager: FakeChatManager,
    tts_client: FakeTTSClient,
    policy: AlwaysReplyPolicy,
    writer: FakeTranscriptWriter,
) -> None:
    monkeypatch.setattr(runner_module, "GenAIChatManager", lambda **kwargs: chat_manager)
    monkeypatch.setattr(runner_module, "ElevenLabsTTSClient", lambda _key: tts_client)
    monkeypatch.setattr(runner_module, "ReplyTriggerPolicy", lambda **kwargs: policy)
    monkeypatch.setattr(runner_module, "TranscriptSessionWriter", lambda **kwargs: writer)
    monkeypatch.setattr(runner_module, "GeminiSTTClient", FakeSTTClient)
    monkeypatch.setattr(runner_module, "VoicePlaybackManager", FakePlaybackManager)
    monkeypatch.setattr(runner_module, "DiscordAudioSink", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(runner_module, "attach_active_transcript_writer", lambda _writer: None)
    monkeypatch.setattr(runner_module, "detach_active_transcript_writer", lambda _writer: None)


def fake_voice_client() -> SimpleNamespace:
    guild = SimpleNamespace(id=7, name="Guild", me=None)
    channel = SimpleNamespace(id=42, name="Lobby", guild=guild, members=[])
    return SimpleNamespace(channel=channel)


async def start_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chat_manager: FakeChatManager | None = None,
    tts_client: FakeTTSClient | None = None,
    policy: AlwaysReplyPolicy | None = None,
    writer: FakeTranscriptWriter | None = None,
    greet_on_join: bool = False,
    metrics: SessionMetrics | None = None,
) -> tuple[asyncio.Task[None], asyncio.Event, FakeDiscordClient]:
    chat_manager = chat_manager or FakeChatManager()
    tts_client = tts_client or FakeTTSClient("key")
    policy = policy or AlwaysReplyPolicy()
    writer = writer or FakeTranscriptWriter()
    install_fakes(
        monkeypatch,
        chat_manager=chat_manager,
        tts_client=tts_client,
        policy=policy,
        writer=writer,
    )

    discord_client = FakeDiscordClient()
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        run_voice_conversation(
            discord_client,  # type: ignore[arg-type]
            fake_voice_client(),
            None,
            elevenlabs_api_key="key",
            settings_store=FakeSettingsStore(),  # type: ignore[arg-type]
            runner_config=runner_config(greet_on_join=greet_on_join),
            external_stop_event=stop_event,
            metrics=metrics,
        ),
        name="test-session",
    )
    return task, stop_event, discord_client


async def stop_session(task: asyncio.Task[None], stop_event: asyncio.Event) -> None:
    stop_event.set()
    await asyncio.wait_for(task, timeout=5.0)


async def wait_for(predicate, *, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# The reply path survives backend errors
# ---------------------------------------------------------------------------


async def test_a_failed_reply_does_not_end_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Gemini 429 used to kill the supervisor and take the session with it."""
    chat_manager = FakeChatManager(failures=1)
    policy = AlwaysReplyPolicy()
    metrics = SessionMetrics()

    task, stop_event, discord_client = await start_session(
        monkeypatch, chat_manager=chat_manager, policy=policy, metrics=metrics
    )

    assert await wait_for(lambda: policy.bot_replies >= 1)
    assert not task.done(), "the session must outlive a failed reply"
    assert discord_client.is_listening()
    assert policy.declined == 1
    assert "429" in metrics.last_error

    await stop_session(task, stop_event)


async def test_an_elevenlabs_quota_error_does_not_end_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The free tier 401s at 10 000 characters; the bot must stay and keep listening."""
    tts_client = FakeTTSClient("key", failures=1)
    policy = AlwaysReplyPolicy()
    metrics = SessionMetrics()

    task, stop_event, discord_client = await start_session(
        monkeypatch, tts_client=tts_client, policy=policy, metrics=metrics
    )

    assert await wait_for(lambda: len(tts_client.spoken) >= 1)
    assert not task.done()
    assert discord_client.is_listening()
    assert "401" in metrics.last_error

    await stop_session(task, stop_event)


async def test_enough_consecutive_reply_failures_end_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "_MAX_CONSECUTIVE_REPLY_FAILURES", 3)
    chat_manager = FakeChatManager(failures=99)
    policy = AlwaysReplyPolicy()

    task, _stop_event, _discord_client = await start_session(
        monkeypatch, chat_manager=chat_manager, policy=policy
    )

    with pytest.raises(ReplyPipelineError, match="3 times in a row"):
        await asyncio.wait_for(task, timeout=5.0)

    assert policy.declined == 3


async def test_a_failing_volunteer_check_is_survivable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``should_volunteer`` hits Gemini too, so it can 429 like anything else."""
    chat_manager = FakeChatManager()
    calls = {"n": 0}

    async def flaky_should_volunteer(_transcript: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return True

    chat_manager.should_volunteer = flaky_should_volunteer  # type: ignore[method-assign]
    policy = AlwaysReplyPolicy()
    policy.reason = TriggerReason.VOLUNTEERED

    task, stop_event, _discord_client = await start_session(
        monkeypatch, chat_manager=chat_manager, policy=policy
    )

    assert await wait_for(lambda: policy.bot_replies >= 1)
    assert not task.done()

    await stop_session(task, stop_event)


# ---------------------------------------------------------------------------
# The greeting is not load-bearing
# ---------------------------------------------------------------------------


async def test_a_failed_greeting_still_starts_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not being able to say hello is no reason to refuse to listen."""
    chat_manager = FakeChatManager(failures=1)
    metrics = SessionMetrics()

    task, stop_event, discord_client = await start_session(
        monkeypatch, chat_manager=chat_manager, greet_on_join=True, metrics=metrics
    )

    assert await wait_for(lambda: discord_client.is_listening())
    assert not task.done()
    assert "join greeting failed" in metrics.last_error
    assert chat_manager.sessions_created == 1, "the chat session must not be recreated"

    await stop_session(task, stop_event)


# ---------------------------------------------------------------------------
# Say why the session ended
# ---------------------------------------------------------------------------


async def test_the_session_logs_which_task_ended_it(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A panel operator used to see the bot leave with no reason at all."""
    monkeypatch.setattr(runner_module, "_MAX_CONSECUTIVE_REPLY_FAILURES", 1)
    chat_manager = FakeChatManager(failures=99)

    with caplog.at_level(logging.ERROR, logger="src.session.conversation_runner"):
        task, _stop_event, _discord_client = await start_session(
            monkeypatch, chat_manager=chat_manager
        )
        with pytest.raises(ReplyPipelineError):
            await asyncio.wait_for(task, timeout=5.0)

    assert "genai-supervisor" in caplog.text
    assert "Voice session ending" in caplog.text


async def test_a_clean_stop_names_the_task_that_finished(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="src.session.conversation_runner"):
        task, stop_event, _discord_client = await start_session(monkeypatch)
        await asyncio.sleep(SUPERVISOR_TICK)
        await stop_session(task, stop_event)

    assert "task 'conversation-runtime' finished normally" in caplog.text
