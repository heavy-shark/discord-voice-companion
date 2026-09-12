"""What a voice session costs, from transcription to voice-over.

Three services are billed, in three different units:

``transcription``
    Gemini ``generate_content`` on the buffered utterance. Audio input is its
    own (much pricier) rate than text input, so the two are never merged.

``replies``
    The same Gemini account: the chat turns plus every volunteer-gate call, all
    text in and text out.

``voice``
    ElevenLabs, billed per character of the text handed to it.

Everything here is a pure function of counters and a rate card, so the numbers
can be checked without touching a network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

_PER_MILLION = 1_000_000.0
_PER_THOUSAND = 1_000.0


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """Cumulative billable units observed so far in a session."""

    stt_audio_input_tokens: int = 0
    stt_text_input_tokens: int = 0
    stt_output_tokens: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    tts_characters: int = 0
    stt_audio_seconds: float = 0.0

    def __sub__(self, earlier: "UsageSnapshot") -> "UsageSnapshot":
        """Usage added since *earlier*, for "what did the last 30 seconds cost"."""
        return UsageSnapshot(
            stt_audio_input_tokens=self.stt_audio_input_tokens - earlier.stt_audio_input_tokens,
            stt_text_input_tokens=self.stt_text_input_tokens - earlier.stt_text_input_tokens,
            stt_output_tokens=self.stt_output_tokens - earlier.stt_output_tokens,
            chat_input_tokens=self.chat_input_tokens - earlier.chat_input_tokens,
            chat_output_tokens=self.chat_output_tokens - earlier.chat_output_tokens,
            tts_characters=self.tts_characters - earlier.tts_characters,
            stt_audio_seconds=self.stt_audio_seconds - earlier.stt_audio_seconds,
        )

    @property
    def is_empty(self) -> bool:
        """True when nothing billable happened."""
        return not any(
            (
                self.stt_audio_input_tokens,
                self.stt_text_input_tokens,
                self.stt_output_tokens,
                self.chat_input_tokens,
                self.chat_output_tokens,
                self.tts_characters,
            )
        )


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """USD estimated from a rate card. Not a bill — the providers' is."""

    transcription_usd: float = 0.0
    replies_usd: float = 0.0
    voice_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.transcription_usd + self.replies_usd + self.voice_usd


def estimate_costs(usage: UsageSnapshot, rates: dict[str, float]) -> CostBreakdown:
    """Price *usage* with a rate card from ``SettingsStore.resolve_pricing_config``."""
    audio_in = rates.get("genai_usd_per_1m_audio_input_tokens", 0.0)
    text_in = rates.get("genai_usd_per_1m_input_tokens", 0.0)
    text_out = rates.get("genai_usd_per_1m_output_tokens", 0.0)
    per_1k_chars = rates.get("elevenlabs_usd_per_1k_characters", 0.0)

    transcription = (
        usage.stt_audio_input_tokens * audio_in
        + usage.stt_text_input_tokens * text_in
        + usage.stt_output_tokens * text_out
    ) / _PER_MILLION
    replies = (
        usage.chat_input_tokens * text_in + usage.chat_output_tokens * text_out
    ) / _PER_MILLION
    voice = usage.tts_characters * per_1k_chars / _PER_THOUSAND

    return CostBreakdown(
        transcription_usd=transcription,
        replies_usd=replies,
        voice_usd=voice,
    )


def _usd(amount: float) -> str:
    """Fractions of a cent are the norm here, so four decimals is the floor."""
    if amount and abs(amount) < 0.0001:
        return f"${amount:.6f}"
    return f"${amount:.4f}"


def format_cost_report(
    label: str,
    usage: UsageSnapshot,
    rates: dict[str, float],
) -> str:
    """One log line: what was consumed and what it is estimated to cost."""
    costs = estimate_costs(usage, rates)
    return (
        f"Cost {label} | "
        f"transcription {usage.stt_audio_seconds:.1f}s audio, "
        f"{usage.stt_audio_input_tokens + usage.stt_text_input_tokens}+"
        f"{usage.stt_output_tokens} tok {_usd(costs.transcription_usd)} | "
        f"replies {usage.chat_input_tokens}+{usage.chat_output_tokens} tok "
        f"{_usd(costs.replies_usd)} | "
        f"voice {usage.tts_characters} chars {_usd(costs.voice_usd)} | "
        f"= {_usd(costs.total_usd)}"
    )


def zeroed(usage: UsageSnapshot) -> UsageSnapshot:
    """A snapshot with the same shape and no usage (for the first delta)."""
    return replace(
        usage,
        stt_audio_input_tokens=0,
        stt_text_input_tokens=0,
        stt_output_tokens=0,
        chat_input_tokens=0,
        chat_output_tokens=0,
        tts_characters=0,
        stt_audio_seconds=0.0,
    )
