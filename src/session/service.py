"""Long-running companion service for headless (server) operation.

The CLI drives one session and exits; the TUI drives one session per button
press. Neither fits a process supervisor that just runs ``python main.py`` and
expects it to stay up: there is no terminal to pick a channel from, and a crashed
session must not take the whole process down.

:class:`CompanionService` owns that lifecycle. It stays connected to the gateway,
idles until something asks it to join a voice channel (a slash command, or an
auto-join at startup), and returns to idle if a session ends or fails.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass

from ..discord import DiscordClient
from ..storage import SettingsStore
from .conversation_runner import ConversationRunnerConfig, run_voice_conversation
from .metrics import SessionMetrics

logger = logging.getLogger(__name__)

# How long to wait for a session to wind down before giving up on a clean stop.
_STOP_TIMEOUT_SECONDS = 20.0

# A voice session that dies on its own (websocket drop, a backend outage that
# outlasted the in-session retries) should get back into the channel by itself
# rather than leaving a silent bot for an operator to notice.
_REJOIN_ATTEMPTS = 3
_REJOIN_BASE_DELAY_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class SessionStatus:
    """A snapshot of what the service is doing, for ``/status``."""

    connected: bool
    channel_id: int | None
    channel_name: str
    persona_name: str
    status_line: str
    last_error: str

    @property
    def is_active(self) -> bool:
        return self.connected and self.channel_id is not None


class CompanionService:
    """Start, stop and supervise voice sessions for one bot process."""

    def __init__(
        self,
        discord_client: DiscordClient,
        settings_store: SettingsStore,
        runner_config: ConversationRunnerConfig,
        *,
        elevenlabs_api_key: str,
        persona_id: str | None = None,
        user_audio_allowed: object | None = None,
    ) -> None:
        self._discord = discord_client
        self._settings = settings_store
        self._runner_config = runner_config
        self._elevenlabs_api_key = elevenlabs_api_key
        self._persona_id = persona_id
        self._user_audio_allowed = user_audio_allowed

        self._session_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._metrics = SessionMetrics()
        self._channel_id: int | None = None
        self._channel_name = ""
        self._last_error = ""
        # Serializes join/leave so two slash commands cannot race the lifecycle.
        self._lock = asyncio.Lock()

    def set_user_audio_allowed(self, predicate: object | None) -> None:
        """Set the sink predicate deciding whose audio is transcribed.

        Exposed as a setter because the predicate needs the bot's own user id,
        which does not exist until the gateway is connected — after the service
        has already been constructed and handed to the command registrar.
        """
        self._user_audio_allowed = predicate

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while a voice session is live."""
        return self._session_task is not None and not self._session_task.done()

    def status(self) -> SessionStatus:
        """Snapshot for ``/status`` and for logging."""
        persona_name = ""
        personas = self._settings.get_personalities()
        for persona in personas:
            if self._persona_id is None or persona.persona_id == self._persona_id:
                persona_name = persona.display_name
                break

        return SessionStatus(
            connected=self.is_running,
            channel_id=self._channel_id if self.is_running else None,
            channel_name=self._channel_name if self.is_running else "",
            persona_name=persona_name,
            status_line=self._metrics.status_line if self.is_running else "idle",
            last_error=self._last_error,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def join(self, channel_id: int, channel_name: str = "") -> None:
        """Join *channel_id* and start conversing. Replaces any running session."""
        async with self._lock:
            if self.is_running:
                logger.info("Switching voice channel; stopping the current session first.")
                await self._stop_locked()

            self._last_error = ""
            self._metrics = SessionMetrics()
            self._channel_id = channel_id
            self._channel_name = channel_name

            voice_client = await self._discord.join_voice_channel(channel_id)
            self._stop_event = asyncio.Event()
            self._session_task = asyncio.create_task(
                self._run_session(voice_client),
                name=f"companion-session-{channel_id}",
            )
            logger.info("Companion session started in channel %s (%s)", channel_name, channel_id)

    async def leave(self) -> None:
        """Stop the session and leave the voice channel."""
        async with self._lock:
            await self._stop_locked()

    async def close(self) -> None:
        """Shut down for process exit."""
        await self.leave()

    async def _stop_locked(self) -> None:
        """Stop the running session. Caller must hold ``self._lock``."""
        task, self._session_task = self._session_task, None
        stop_event, self._stop_event = self._stop_event, None
        self._channel_id = None
        self._channel_name = ""

        if task is not None and not task.done():
            if stop_event is not None:
                stop_event.set()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=_STOP_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("Session did not stop within %.0fs; cancelling.", _STOP_TIMEOUT_SECONDS)
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            except Exception:
                logger.exception("Session raised while stopping")

        with suppress(Exception):
            await self._discord.leave_voice_channel()

    async def _run_session(self, voice_client: object) -> None:
        """Run one conversation session, and never let its failure kill the process."""
        stop_event = self._stop_event
        assert stop_event is not None
        # ``_stop_locked`` clears these while we may still be retrying.
        channel_id = self._channel_id
        channel_name = self._channel_name

        attempt = 0
        while True:
            try:
                await run_voice_conversation(
                    discord_client=self._discord,
                    voice_client=voice_client,
                    listen_window_seconds=None,
                    elevenlabs_api_key=self._elevenlabs_api_key,
                    settings_store=self._settings,
                    runner_config=self._runner_config,
                    persona_id=self._persona_id,
                    external_stop_event=stop_event,
                    metrics=self._metrics,
                    user_audio_allowed=self._user_audio_allowed,  # type: ignore[arg-type]
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # The bot stays online and can be asked to /join again, rather than
                # taking the whole process down and waiting on a supervisor restart.
                self._last_error = str(exc)
                logger.exception(
                    "Voice session in channel %s (%s) failed.", channel_name or "?", channel_id
                )
                with suppress(Exception):
                    await self._discord.leave_voice_channel()

            if stop_event.is_set() or channel_id is None:
                logger.info("Not rejoining: the session was asked to stop. Returning to idle.")
                return

            attempt += 1
            if attempt > _REJOIN_ATTEMPTS:
                logger.error(
                    "Voice session failed %d time(s) in a row; returning to idle. "
                    "Use /join to try again.",
                    _REJOIN_ATTEMPTS + 1,
                )
                return

            delay = _REJOIN_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Rejoining voice channel %s in %.1fs (attempt %d/%d).",
                channel_id,
                delay,
                attempt,
                _REJOIN_ATTEMPTS,
            )
            self._metrics.status_line = "reconnecting"
            await asyncio.sleep(delay)
            if stop_event.is_set():
                logger.info("Stop requested while reconnecting; returning to idle.")
                return

            try:
                voice_client = await self._discord.join_voice_channel(channel_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("Rejoining voice channel %s failed; returning to idle.", channel_id)
                return
