"""Entry point for running the companion as a long-lived server process.

This is the file a process manager runs (``python main.py``). It takes no
arguments: everything comes from ``.env``. Start it, then type ``/join`` in
Discord while sitting in a voice channel.

For local development with a terminal, ``python -m src.main --tui`` gives the
dashboard instead.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import suppress

import discord
from src.discord import DiscordClient
from src.discord.companion_commands import register_companion_commands
from src.discord.preflight import PreflightError, validate_voice_dependencies
from src.main import (
    configure_logging,
    resolve_auto_join_channel_id,
    resolve_barge_in,
    resolve_greet_on_join,
    resolve_mention_window_seconds,
    resolve_reply_cooldown_seconds,
    resolve_reply_silence_seconds,
    resolve_require_consent,
    resolve_spoken_languages,
    resolve_stt_idle_timeout_seconds,
    resolve_talkativeness,
    voice_user_audio_allowed,
)
from src.runtime_dirs import load_application_dotenv
from src.session import ConversationRunnerConfig
from src.session.service import CompanionService
from src.storage import SettingsStore
from src.storage.reply_locale import resolve_bot_reply_language_code
from src.tts import resolve_elevenlabs_api_key

logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """A required setting is missing or unusable."""


def _require_credentials(settings_store: SettingsStore) -> tuple[str, str, str]:
    """Return (discord token, gemini key, elevenlabs key) or explain what is missing."""
    discord_token = settings_store.resolve_discord_secret("token", "DISCORD_TOKEN") or ""
    gemini_key = (
        settings_store.resolve_api_secret("google_gemini_api_key", "GOOGLE_GEMINI_API_KEY") or ""
    )
    elevenlabs_key = resolve_elevenlabs_api_key(settings_store) or ""

    missing = [
        name
        for name, value in (
            ("DISCORD_TOKEN", discord_token),
            ("GOOGLE_GEMINI_API_KEY", gemini_key),
            ("ELEVENLABS_API_KEY", elevenlabs_key),
        )
        if not value.strip()
    ]
    if missing:
        raise ConfigError(
            "Missing required setting(s): "
            + ", ".join(missing)
            + ".\nOpen the .env file next to main.py and fill them in — it lists where "
            "each key comes from. Then commit and restart."
        )
    return discord_token, gemini_key, elevenlabs_key


def _build_runner_config(settings_store: SettingsStore) -> ConversationRunnerConfig:
    return ConversationRunnerConfig(
        stt_idle_timeout_seconds=resolve_stt_idle_timeout_seconds(),
        mention_window_seconds=resolve_mention_window_seconds(settings_store),
        greet_on_join=resolve_greet_on_join(settings_store),
        reply_locale=resolve_bot_reply_language_code(settings_store),
        talkativeness=resolve_talkativeness(settings_store),
        spoken_languages=resolve_spoken_languages(settings_store),
        reply_silence_seconds=resolve_reply_silence_seconds(settings_store),
        reply_cooldown_seconds=resolve_reply_cooldown_seconds(settings_store),
        barge_in=resolve_barge_in(settings_store),
    )


def _install_shutdown_handlers(stop_event: asyncio.Event) -> None:
    """Ask the loop to stop on SIGINT/SIGTERM so the panel gets a clean exit."""
    loop = asyncio.get_running_loop()

    def request_stop(signal_name: str) -> None:
        logger.info("Received %s; shutting down.", signal_name)
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, request_stop, sig_name)
        except NotImplementedError:
            # Windows' ProactorEventLoop has no add_signal_handler; fall back to
            # the plain handler, which is enough to break the idle wait.
            signal.signal(sig, lambda *_: request_stop(sig_name))


async def run() -> int:
    load_application_dotenv()
    configure_logging()

    settings_store = SettingsStore()

    try:
        validate_voice_dependencies()
    except PreflightError as exc:
        logger.error("Voice dependencies are not usable: %s", exc)
        logger.error("Run: pip install -r requirements.txt")
        return 1

    try:
        discord_token, _gemini_key, elevenlabs_key = _require_credentials(settings_store)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    if not settings_store.get_personalities():
        logger.error(
            "No characters configured. settings.json (or settings-example.json, which is "
            "used as the fallback) must define at least one persona."
        )
        return 1

    runner_config = _build_runner_config(settings_store)
    require_consent = resolve_require_consent(settings_store)

    try:
        auto_join_channel_id = resolve_auto_join_channel_id(settings_store)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    discord_client = DiscordClient(discord_token)
    service = CompanionService(
        discord_client,
        settings_store,
        runner_config,
        elevenlabs_api_key=elevenlabs_key,
        user_audio_allowed=None,  # replaced below, once the bot's own id is known
    )
    discord_client.add_command_registrar(
        lambda tree: register_companion_commands(tree, service)
    )

    stop_event = asyncio.Event()
    _install_shutdown_handlers(stop_event)

    logger.info(
        "Starting companion (talkativeness=%s, reply language=%s, consent required=%s)",
        runner_config.talkativeness,
        runner_config.reply_locale,
        require_consent,
    )

    try:
        await discord_client.connect()
    except Exception as exc:
        logger.error("Could not connect to Discord: %s", exc)
        if isinstance(exc.__cause__, discord.PrivilegedIntentsRequired):
            # The token is fine; sending people to reset it wastes their time.
            logger.error(
                "This is the privileged intent above, not the token. Enable SERVER MEMBERS "
                "INTENT in the Developer Portal and start again."
            )
        else:
            logger.error(
                "Check DISCORD_TOKEN in .env — it must be the Bot token from the Developer "
                "Portal (Bot tab -> Reset Token), not the client secret or application id."
            )
        logger.debug("Gateway failure detail", exc_info=True)
        # Close the aiohttp session the failed login left behind.
        with suppress(Exception):
            await discord_client.disconnect()
        return 1

    # The predicate needs the bot's own user id, which only exists after connect.
    service.set_user_audio_allowed(
        voice_user_audio_allowed(discord_client, require_consent=require_consent)
    )

    if discord_client.last_app_command_sync_summary:
        logger.info("%s", discord_client.last_app_command_sync_summary)

    try:
        if auto_join_channel_id is not None:
            try:
                await service.join(auto_join_channel_id)
            except Exception:
                logger.exception(
                    "Auto-join of channel %s failed; staying online. Use /join instead.",
                    auto_join_channel_id,
                )
        else:
            logger.info("Ready. Join a voice channel in Discord and run /join.")

        await stop_event.wait()
    finally:
        logger.info("Stopping...")
        with suppress(Exception):
            await service.close()
        with suppress(Exception):
            await discord_client.disconnect()

    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
