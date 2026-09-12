"""Session cost estimates: transcription, replies, voice-over."""

from __future__ import annotations

import pytest

from src.session.costs import (
    UsageSnapshot,
    estimate_costs,
    format_cost_report,
    zeroed,
)

RATES = {
    "elevenlabs_usd_per_1k_characters": 0.18,
    "genai_usd_per_1m_input_tokens": 0.30,
    "genai_usd_per_1m_output_tokens": 2.50,
    "genai_usd_per_1m_audio_input_tokens": 1.00,
    "google_stt_usd_per_minute": 0.001,
}


def test_transcription_audio_is_priced_at_the_audio_rate() -> None:
    """Audio input costs 3.3x text input; merging them understates the bill."""
    usage = UsageSnapshot(stt_audio_input_tokens=1_000_000)
    assert estimate_costs(usage, RATES).transcription_usd == pytest.approx(1.00)

    as_text = UsageSnapshot(stt_text_input_tokens=1_000_000)
    assert estimate_costs(as_text, RATES).transcription_usd == pytest.approx(0.30)


def test_each_service_lands_in_its_own_bucket() -> None:
    usage = UsageSnapshot(
        stt_audio_input_tokens=100_000,
        stt_output_tokens=1_000,
        chat_input_tokens=200_000,
        chat_output_tokens=10_000,
        tts_characters=5_000,
    )
    costs = estimate_costs(usage, RATES)

    assert costs.transcription_usd == pytest.approx(0.1 + 0.0025)
    assert costs.replies_usd == pytest.approx(0.06 + 0.025)
    assert costs.voice_usd == pytest.approx(0.9)
    assert costs.total_usd == pytest.approx(1.0875)


def test_missing_rates_are_treated_as_free_not_as_a_crash() -> None:
    usage = UsageSnapshot(stt_audio_input_tokens=1_000, tts_characters=1_000)
    assert estimate_costs(usage, {}).total_usd == pytest.approx(0.0)


def test_subtracting_snapshots_gives_the_usage_of_one_window() -> None:
    earlier = UsageSnapshot(chat_input_tokens=100, tts_characters=50, stt_audio_seconds=4.0)
    later = UsageSnapshot(chat_input_tokens=360, tts_characters=130, stt_audio_seconds=9.5)

    batch = later - earlier

    assert batch.chat_input_tokens == 260
    assert batch.tts_characters == 80
    assert batch.stt_audio_seconds == pytest.approx(5.5)


def test_an_idle_window_is_empty_so_the_log_can_stay_quiet() -> None:
    assert (UsageSnapshot() - UsageSnapshot()).is_empty
    assert not UsageSnapshot(tts_characters=1).is_empty
    # Seconds alone are not billable; tokens and characters are.
    assert UsageSnapshot(stt_audio_seconds=3.0).is_empty


def test_zeroed_keeps_the_shape_and_drops_the_usage() -> None:
    assert zeroed(UsageSnapshot(chat_input_tokens=5, tts_characters=9)).is_empty


def test_the_report_names_every_service_and_the_total() -> None:
    usage = UsageSnapshot(
        stt_audio_input_tokens=320,
        stt_output_tokens=12,
        chat_input_tokens=800,
        chat_output_tokens=40,
        tts_characters=120,
        stt_audio_seconds=10.0,
    )
    line = format_cost_report("last 30s", usage, RATES)

    assert line.startswith("Cost last 30s |")
    assert "transcription 10.0s audio" in line
    assert "replies 800+40 tok" in line
    assert "voice 120 chars" in line
    assert line.endswith(f"= ${estimate_costs(usage, RATES).total_usd:.4f}")


def test_sub_cent_amounts_keep_enough_digits_to_be_readable() -> None:
    """At these volumes a window costs fractions of a cent; $0.0000 says nothing."""
    line = format_cost_report("last 30s", UsageSnapshot(stt_audio_input_tokens=10), RATES)
    assert "$0.000010" in line
