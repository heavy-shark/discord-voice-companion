"""Resolve the language the companion replies in."""

from __future__ import annotations

import logging
import os

from .settings import AUTO_REPLY_LOCALE, SettingsStore

logger = logging.getLogger(__name__)


def resolve_bot_reply_language_code(settings_store: SettingsStore) -> str:
    """Language tag for spoken replies, or ``"auto"`` to mirror the speaker.

    Precedence: ``reply_language`` in the runtime config (set from the settings
    editor), then ``BOT_REPLY_LANGUAGE`` / ``BOT_LANGUAGE`` from the environment,
    then ``"auto"``.

    ``"auto"`` is the default because the companion is expected to sit in a
    channel where Ukrainian, Russian and English are all in use; pinning one
    BCP-47 tag would force it to answer Ukrainian questions in English (or vice
    versa). Set an explicit tag to lock replies to one language.
    """
    runtime = settings_store.get_runtime_config()
    settings_override = str(runtime.get("reply_language", "")).strip()
    if settings_override:
        logger.info("Reply locale uses settings override %s", settings_override)
        return settings_override

    env_override = (
        os.getenv("BOT_REPLY_LANGUAGE", "").strip() or os.getenv("BOT_LANGUAGE", "").strip()
    )
    if env_override:
        logger.info("Reply locale uses environment override %s", env_override)
        return env_override

    return AUTO_REPLY_LOCALE


def is_auto_reply_locale(locale: str | None) -> bool:
    """True when replies should mirror the language each speaker used."""
    return (locale or "").strip().lower() in ("", AUTO_REPLY_LOCALE)
