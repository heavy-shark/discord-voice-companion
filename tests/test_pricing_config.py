"""Pricing block resolution for session cost estimates."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.settings import SettingsStore


def test_resolve_pricing_config_defaults(tmp_path: Path) -> None:
    """Published list prices, Sept 2026: Gemini 2.5 Flash and ElevenLabs Creator."""
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    store = SettingsStore(settings_path=path)
    p = store.resolve_pricing_config()
    assert p["elevenlabs_usd_per_1k_characters"] == pytest.approx(0.18)
    assert p["genai_usd_per_1m_input_tokens"] == pytest.approx(0.30)
    assert p["genai_usd_per_1m_output_tokens"] == pytest.approx(2.50)
    # Audio input is its own, much higher rate; transcription is priced with it.
    assert p["genai_usd_per_1m_audio_input_tokens"] == pytest.approx(1.00)
    assert p["google_stt_usd_per_minute"] == pytest.approx(0.001)


def test_the_audio_input_rate_can_be_overridden_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GENAI_USD_PER_1M_AUDIO_INPUT_TOKENS", "2.5")
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    store = SettingsStore(settings_path=path)
    rates = store.resolve_pricing_config()
    assert rates["genai_usd_per_1m_audio_input_tokens"] == pytest.approx(2.5)


def test_resolve_pricing_config_settings_override_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GENAI_USD_PER_1M_OUTPUT_TOKENS", "9.99")
    path = tmp_path / "settings.json"
    path.write_text(
        '{"pricing": {"genai_usd_per_1m_output_tokens": 2.0}}',
        encoding="utf-8",
    )
    store = SettingsStore(settings_path=path)
    p = store.resolve_pricing_config()
    assert p["genai_usd_per_1m_output_tokens"] == pytest.approx(2.0)


def test_get_pricing_config_accepts_legacy_genai_key_casing(tmp_path: Path) -> None:
    """Regression: defaults once used ``1M`` keys; UI expects ``1m`` keys."""
    path = tmp_path / "settings.json"
    path.write_text(
        '{"pricing": {"genai_usd_per_1M_input_tokens": 0.4, "genai_usd_per_1M_output_tokens": 2.0}}',
        encoding="utf-8",
    )
    store = SettingsStore(settings_path=path)
    p = store.get_pricing_config()
    assert p["genai_usd_per_1m_input_tokens"] == pytest.approx(0.4)
    assert p["genai_usd_per_1m_output_tokens"] == pytest.approx(2.0)


def test_resolve_pricing_config_env_when_missing_in_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_STT_USD_PER_MINUTE", "0.05")
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    store = SettingsStore(settings_path=path)
    assert store.resolve_pricing_config()["google_stt_usd_per_minute"] == pytest.approx(0.05)
