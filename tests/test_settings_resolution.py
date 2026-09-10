"""Tests for settings-first runtime and secret resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from src.main import get_required_token, resolve_greet_on_join, resolve_reply_cooldown_seconds
from src.storage import (
    DEFAULT_STT_BACKEND,
    DEFAULT_STT_BATCH_MODEL,
    DEFAULT_STT_LANGUAGE_HINTS,
    DEFAULT_STT_LIVE_MODEL,
    SettingsStore,
)
from src.tts import resolve_elevenlabs_api_key


def _store(tmp_path: Path) -> SettingsStore:
    return SettingsStore(settings_path=tmp_path / "settings.json")


def _write_json(path: Path, payload: dict[str, object]) -> str:
    text = json.dumps(payload, indent=2)
    path.write_text(text, encoding="utf-8")
    return text


def _fallback_payload(persona_id: str = "fallback") -> dict[str, object]:
    return {
        "personas": [
            {
                "id": persona_id,
                "name": f"{persona_id.title()} Bot",
                "system_instruction": "Be concise.",
                "genai_model": "gemini-2.5-flash",
                "elevenlabs_voice_id": "voice-id",
            }
        ],
        "ui": {
            "last_guild_id": None,
            "last_channel_id": None,
            "last_persona_id": persona_id,
        },
        "stt": {"language_code": "en-US"},
    }


def test_loads_settings_example_when_settings_json_missing(tmp_path: Path) -> None:
    _write_json(tmp_path / "settings-example.json", _fallback_payload())

    store = _store(tmp_path)

    assert [persona.persona_id for persona in store.personas] == ["fallback"]
    assert store.get_ui_preferences()["last_persona_id"] == "fallback"
    assert not (tmp_path / "settings.json").exists()


def test_save_after_settings_example_fallback_creates_settings_json(tmp_path: Path) -> None:
    template_path = tmp_path / "settings-example.json"
    original_template = _write_json(template_path, _fallback_payload())
    store = _store(tmp_path)

    store.update_ui_preference("last_persona_id", "saved")

    assert store.save() is True

    saved_path = tmp_path / "settings.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["ui"]["last_persona_id"] == "saved"
    assert template_path.read_text(encoding="utf-8") == original_template


def test_existing_settings_json_takes_precedence_over_template(tmp_path: Path) -> None:
    _write_json(tmp_path / "settings-example.json", _fallback_payload("template"))
    _write_json(tmp_path / "settings.json", _fallback_payload("persisted"))

    store = _store(tmp_path)

    assert [persona.persona_id for persona in store.personas] == ["persisted"]
    assert store.get_ui_preferences()["last_persona_id"] == "persisted"


def test_missing_settings_and_template_leaves_store_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.settings == {}
    assert store.personas == []


def test_reply_cooldown_prefers_settings_over_env(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set_runtime_config(
        reply_silence_seconds=5.0,
        reply_cooldown_seconds=45.0,
        mention_window_seconds=30.0,
        reply_language="",
        greet_on_join=True,
    )

    with patch.dict(os.environ, {"BOT_REPLY_COOLDOWN_SECONDS": "999"}, clear=False):
        assert resolve_reply_cooldown_seconds(store) == 45.0


def test_greet_on_join_prefers_settings_over_env(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set_runtime_config(
        reply_silence_seconds=5.0,
        reply_cooldown_seconds=180.0,
        mention_window_seconds=30.0,
        reply_language="",
        greet_on_join=False,
    )
    with patch.dict(os.environ, {"BOT_GREET_ON_JOIN": "true"}, clear=False):
        assert resolve_greet_on_join(store) is False


def test_greet_on_join_follows_env_without_settings_override(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with patch.dict(os.environ, {"BOT_GREET_ON_JOIN": "false"}, clear=False):
        assert resolve_greet_on_join(store) is False


def test_discord_token_prefers_settings_over_env(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set_discord_config(
        token="settings-token",
        server_id="",
    )
    store.set_api_config(
        google_gemini_api_key="",
        elevenlabs_api_key="",
    )

    with patch.dict(os.environ, {"DISCORD_TOKEN": "env-token"}, clear=False):
        assert get_required_token(store) == "settings-token"


def test_elevenlabs_key_prefers_settings_over_env(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set_discord_config(
        token="",
        server_id="",
    )
    store.set_api_config(
        google_gemini_api_key="",
        elevenlabs_api_key="settings-eleven",
    )

    with patch.dict(
        os.environ,
        {"ELEVENLABS_API_KEY": "env-primary", "ELEVEN_API_KEY": "env-fallback"},
        clear=False,
    ):
        assert resolve_elevenlabs_api_key(store) == "settings-eleven"


def test_resolve_stt_backend_prefers_settings_over_env(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.update_stt_config(backend="batch")
    with patch.dict(os.environ, {"GEMINI_STT_BACKEND": "live"}, clear=False):
        assert store.resolve_stt_backend() == "batch"


def test_resolve_stt_backend_falls_back_to_env_then_default(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with patch.dict(os.environ, {"GEMINI_STT_BACKEND": "batch"}, clear=False):
        assert store.resolve_stt_backend() == "batch"
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GEMINI_STT_BACKEND", None)
        assert store.resolve_stt_backend() == DEFAULT_STT_BACKEND


def test_resolve_stt_backend_rejects_unknown_values(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.update_stt_config(backend="whisper")
    assert store.resolve_stt_backend() == DEFAULT_STT_BACKEND


def test_resolve_stt_models_prefer_settings_over_env(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.update_stt_config(live_model="settings-live", batch_model="settings-batch")
    with patch.dict(
        os.environ,
        {"GEMINI_STT_LIVE_MODEL": "env-live", "GEMINI_STT_BATCH_MODEL": "env-batch"},
        clear=False,
    ):
        assert store.resolve_stt_live_model() == "settings-live"
        assert store.resolve_stt_batch_model() == "settings-batch"


def test_resolve_stt_models_default_when_unset(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for key in ("GEMINI_STT_LIVE_MODEL", "GEMINI_STT_BATCH_MODEL"):
        os.environ.pop(key, None)
    assert store.resolve_stt_live_model() == DEFAULT_STT_LIVE_MODEL
    assert store.resolve_stt_batch_model() == DEFAULT_STT_BATCH_MODEL


def test_resolve_stt_language_hints_defaults_to_uk_ru_en(tmp_path: Path) -> None:
    store = _store(tmp_path)
    os.environ.pop("GEMINI_STT_LANGUAGE_HINTS", None)
    assert store.resolve_stt_language_hints() == list(DEFAULT_STT_LANGUAGE_HINTS)


def test_resolve_stt_language_hints_accepts_a_comma_separated_string(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.update_stt_config(language_hints="uk-UA, pl-PL ,, en-GB")
    assert store.resolve_stt_language_hints() == ["uk-UA", "pl-PL", "en-GB"]


def test_resolve_stt_language_hints_can_be_emptied_for_pure_auto_detect(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.update_stt_config(language_hints=[])
    assert store.resolve_stt_language_hints() == []


def test_discord_server_id_prefers_settings_over_env(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set_discord_config(
        token="",
        server_id="123456789",
    )

    with patch.dict(os.environ, {"DISCORD_SERVER_ID": "987654321"}, clear=False):
        assert store.resolve_discord_int("server_id", "DISCORD_SERVER_ID") == 123456789
