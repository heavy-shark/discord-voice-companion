"""Tests for the long-lived companion service lifecycle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.models import Persona
from src.session import ConversationRunnerConfig
from src.session import service as service_module
from src.session.service import CompanionService


def runner_config() -> ConversationRunnerConfig:
    return ConversationRunnerConfig(
        stt_idle_timeout_seconds=1.2,
        mention_window_seconds=30.0,
        greet_on_join=False,
        reply_locale="auto",
        talkativeness=65,
        spoken_languages=("uk-UA", "ru-RU", "en-US"),
    )


class FakeDiscordClient:
    def __init__(self) -> None:
        self.joined: list[int] = []
        self.left = 0

    async def join_voice_channel(self, channel_id: int) -> object:
        self.joined.append(channel_id)
        return SimpleNamespace(channel=SimpleNamespace(id=channel_id, name="Lobby"))

    async def leave_voice_channel(self) -> None:
        self.left += 1


class FakeSettingsStore:
    def __init__(self, personas: list[Persona] | None = None) -> None:
        self._personas = personas if personas is not None else [_persona()]

    def get_personalities(self) -> list[Persona]:
        return self._personas


def _persona(persona_id: str = "default", name: str = "Companion") -> Persona:
    return Persona(
        persona_id=persona_id,
        display_name=name,
        system_instruction="Be brief.",
        genai_model="gemini-2.5-flash",
        elevenlabs_voice_id="voice1",
    )


def build_service(
    monkeypatch: pytest.MonkeyPatch,
    runner,
    *,
    settings: FakeSettingsStore | None = None,
) -> tuple[CompanionService, FakeDiscordClient]:
    monkeypatch.setattr(service_module, "run_voice_conversation", runner)
    discord_client = FakeDiscordClient()
    service = CompanionService(
        discord_client,  # type: ignore[arg-type]
        settings or FakeSettingsStore(),  # type: ignore[arg-type]
        runner_config(),
        elevenlabs_api_key="key",
    )
    return service, discord_client


async def test_join_starts_a_session_and_leave_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()

    async def runner(**kwargs) -> None:
        started.set()
        await kwargs["external_stop_event"].wait()

    service, discord_client = build_service(monkeypatch, runner)

    await service.join(123, "Lobby")
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert service.is_running
    assert discord_client.joined == [123]
    assert service.status().channel_name == "Lobby"

    await service.leave()

    assert not service.is_running
    assert discord_client.left >= 1
    assert service.status().channel_id is None


async def test_join_while_running_switches_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    async def runner(**kwargs) -> None:
        await kwargs["external_stop_event"].wait()

    service, discord_client = build_service(monkeypatch, runner)

    await service.join(1, "First")
    await asyncio.sleep(0)
    await service.join(2, "Second")
    await asyncio.sleep(0)

    assert discord_client.joined == [1, 2]
    assert service.status().channel_name == "Second"

    await service.leave()


async def test_a_failing_session_returns_to_idle_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead session must not take the bot offline; /join has to still work."""

    async def runner(**_kwargs) -> None:
        raise RuntimeError("gemini exploded")

    service, discord_client = build_service(monkeypatch, runner)

    await service.join(7, "Lobby")
    # Let the session task run and fail.
    for _ in range(10):
        await asyncio.sleep(0)
        if not service.is_running:
            break

    assert not service.is_running
    status = service.status()
    assert status.is_active is False
    assert "gemini exploded" in status.last_error
    # It left the channel it could not work in.
    assert discord_client.left >= 1


async def test_leave_is_a_no_op_when_nothing_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def runner(**_kwargs) -> None:
        return None

    service, _ = build_service(monkeypatch, runner)

    await service.leave()
    assert not service.is_running


async def test_a_session_that_ignores_the_stop_event_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "_STOP_TIMEOUT_SECONDS", 0.05)
    cancelled = asyncio.Event()

    async def runner(**_kwargs) -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    service, _ = build_service(monkeypatch, runner)

    await service.join(5, "Lobby")
    await asyncio.sleep(0)
    await service.leave()

    assert cancelled.is_set()
    assert not service.is_running


async def test_status_reports_the_selected_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    async def runner(**kwargs) -> None:
        await kwargs["external_stop_event"].wait()

    settings = FakeSettingsStore([_persona("default", "Компаньйон")])
    service, _ = build_service(monkeypatch, runner, settings=settings)

    await service.join(9, "Lobby")
    await asyncio.sleep(0)

    assert service.status().persona_name == "Компаньйон"

    await service.leave()


async def test_user_audio_predicate_is_passed_to_the_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The predicate is set after connect, so it must reach a later session."""
    seen: dict[str, object] = {}

    async def runner(**kwargs) -> None:
        seen["predicate"] = kwargs["user_audio_allowed"]

    service, _ = build_service(monkeypatch, runner)

    def allow(_user_id: int) -> bool:
        return True

    service.set_user_audio_allowed(allow)
    await service.join(3, "Lobby")
    for _ in range(10):
        await asyncio.sleep(0)
        if "predicate" in seen:
            break

    assert seen["predicate"] is allow
