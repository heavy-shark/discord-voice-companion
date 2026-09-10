"""Reply trigger policy - silence, mentions, cooldown and talkativeness."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

from ..storage.settings import DEFAULT_TALKATIVENESS

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TALKATIVENESS",
    "ReplyTriggerPolicy",
    "TalkativenessProfile",
    "TriggerDecision",
    "TriggerReason",
    "TriggerState",
]

# Talkativeness maps one 0-100 dial onto the timings below. The endpoints are
# chosen so that 0 is "only speaks when spoken to" and 100 is "behaves like
# another person in the call"; the default sits between the two.

_SILENCE_AT_QUIET = 6.0
_SILENCE_AT_CHATTY = 1.6
_COOLDOWN_AT_QUIET = 180.0
_COOLDOWN_AT_CHATTY = 8.0


class TriggerState(Enum):
    """Bot state machine states."""

    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    MENTION_WAITING = "mention_waiting"


class TriggerReason(Enum):
    """Why a reply is being generated - drives how the prompt is framed."""

    ADDRESSED = "addressed"
    """Someone said the bot's name; it must answer."""

    VOLUNTEERED = "volunteered"
    """Nobody asked; the bot is choosing to join in and may stay quiet."""


def _interpolate(quiet_value: float, chatty_value: float, talkativeness: int) -> float:
    """Interpolate a duration geometrically between the two endpoints.

    Linear interpolation would make the middle of the dial feel far quieter than
    it reads: halfway between a 180 s and an 8 s cooldown is 94 s, which is not
    "moderately chatty" by any listener's judgement. Geometric interpolation
    treats the dial as a ratio, so the midpoint lands near 38 s instead.
    """
    ratio = max(0, min(100, int(talkativeness))) / 100.0
    return quiet_value * ((chatty_value / quiet_value) ** ratio)


@dataclass(frozen=True, slots=True)
class TalkativenessProfile:
    """Timings derived from the 0-100 talkativeness dial."""

    talkativeness: int
    silence_timeout_seconds: float
    cooldown_seconds: float
    volunteers: bool
    """Whether the bot ever speaks without being addressed."""

    @classmethod
    def from_dial(cls, talkativeness: int) -> "TalkativenessProfile":
        level = max(0, min(100, int(talkativeness)))
        return cls(
            talkativeness=level,
            silence_timeout_seconds=_interpolate(_SILENCE_AT_QUIET, _SILENCE_AT_CHATTY, level),
            cooldown_seconds=_interpolate(_COOLDOWN_AT_QUIET, _COOLDOWN_AT_CHATTY, level),
            # At 0 the bot is purely reactive: it answers when addressed and
            # otherwise never interrupts.
            volunteers=level > 0,
        )


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Outcome of one policy evaluation."""

    should_reply: bool
    reason: TriggerReason | None = None

    def __bool__(self) -> bool:
        return self.should_reply


_NO_REPLY = TriggerDecision(False)


class ReplyTriggerPolicy:
    """
    Decides when the companion speaks.

    - Addressed by name: answer once the speaker pauses, or when the mention
      window elapses (so a long question still gets an answer).
    - Otherwise: consider volunteering after a silence gap, rate-limited by a
      cooldown. Both timings come from the talkativeness dial. The caller is
      expected to run a cheap "is this worth saying?" check on ``VOLUNTEERED``
      triggers, which is what keeps a chatty setting from being obnoxious.
    """

    def __init__(
        self,
        silence_timeout_seconds: float | None = None,
        cooldown_seconds: float | None = None,
        mention_window_seconds: float = 30.0,
        *,
        talkativeness: int = DEFAULT_TALKATIVENESS,
    ) -> None:
        """
        Args:
            silence_timeout_seconds: Overrides the talkativeness-derived silence gap.
            cooldown_seconds: Overrides the talkativeness-derived cooldown.
            mention_window_seconds: Max wait after a mention before answering anyway.
            talkativeness: 0-100 dial; 0 only answers when addressed.
        """
        self.profile = TalkativenessProfile.from_dial(talkativeness)
        self.silence_timeout_seconds = (
            self.profile.silence_timeout_seconds
            if silence_timeout_seconds is None
            else silence_timeout_seconds
        )
        self.cooldown_seconds = (
            self.profile.cooldown_seconds if cooldown_seconds is None else cooldown_seconds
        )
        self.mention_window_seconds = mention_window_seconds

        self.state = TriggerState.LISTENING
        self.last_human_speech_timestamp: float | None = None
        self.last_bot_reply_timestamp: float | None = None
        self.mention_detected_timestamp: float | None = None
        self.mention_triggers: list[str] = []

        logger.info(
            "ReplyTriggerPolicy initialized: talkativeness=%s, silence=%.1fs, "
            "cooldown=%.1fs, mention_window=%.1fs, volunteers=%s",
            self.profile.talkativeness,
            self.silence_timeout_seconds,
            self.cooldown_seconds,
            self.mention_window_seconds,
            self.profile.volunteers,
        )

    @property
    def talkativeness(self) -> int:
        """The configured 0-100 dial."""
        return self.profile.talkativeness

    def set_mention_triggers(self, names: list[str]) -> None:
        """Set lowercase substrings that open the mention window when found in transcript text."""
        seen: set[str] = set()
        out: list[str] = []
        for raw in names:
            s = str(raw).strip().lower()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        self.mention_triggers = out
        logger.debug("Mention triggers set: %s", self.mention_triggers)

    def text_contains_mention_trigger(self, text: str) -> bool:
        """True if any configured trigger appears as a substring of *text* (case-insensitive)."""
        if not self.mention_triggers:
            return False
        low = text.lower()
        return any(t in low for t in self.mention_triggers)

    def record_human_speech(self) -> None:
        """Record that human speech was detected."""
        self.last_human_speech_timestamp = time.time()
        logger.debug("Human speech detected")

    def record_mention(self) -> None:
        """Record that bot's nickname was mentioned."""
        if self.state != TriggerState.MENTION_WAITING:
            self.mention_detected_timestamp = time.time()
            self.state = TriggerState.MENTION_WAITING
            logger.info("Mention detected, entering mention window")

    def record_bot_reply(self) -> None:
        """Record that bot sent a reply."""
        self.last_bot_reply_timestamp = time.time()
        self.state = TriggerState.LISTENING
        self.mention_detected_timestamp = None
        logger.debug("Bot reply recorded, cooldown reset")

    def record_declined_reply(self) -> None:
        """Record that the bot considered volunteering and chose to stay quiet.

        The cooldown still starts, so a declined turn does not cause the policy to
        re-evaluate the same silence on the very next tick.
        """
        self.last_bot_reply_timestamp = time.time()
        self.state = TriggerState.LISTENING
        logger.debug("Volunteered reply declined; cooldown started")

    def evaluate(
        self,
        current_time: float | None = None,
        *,
        has_pending_transcript: bool = True,
    ) -> TriggerDecision:
        """Decide whether to reply now, and whether the bot was addressed.

        Args:
            current_time: Current timestamp (uses ``time.time()`` if omitted).
            has_pending_transcript: Require at least one user transcript since the
                last bot turn, so silence alone never triggers a reply.
        """
        if current_time is None:
            current_time = time.time()

        if not has_pending_transcript:
            return _NO_REPLY

        if self.state == TriggerState.MENTION_WAITING and self.mention_detected_timestamp:
            mention_elapsed = current_time - self.mention_detected_timestamp

            if self.last_human_speech_timestamp:
                silence_elapsed = current_time - self.last_human_speech_timestamp
                if silence_elapsed >= self.silence_timeout_seconds:
                    logger.info("Mention window: speaker paused, answering")
                    self.state = TriggerState.LISTENING
                    self.mention_detected_timestamp = None
                    return TriggerDecision(True, TriggerReason.ADDRESSED)

            if mention_elapsed >= self.mention_window_seconds:
                logger.info("Mention window: timeout elapsed, answering anyway")
                self.state = TriggerState.LISTENING
                self.mention_detected_timestamp = None
                return TriggerDecision(True, TriggerReason.ADDRESSED)

            return _NO_REPLY

        if not self.profile.volunteers:
            return _NO_REPLY

        if not self.last_human_speech_timestamp:
            return _NO_REPLY

        silence_elapsed = current_time - self.last_human_speech_timestamp
        if silence_elapsed < self.silence_timeout_seconds:
            return _NO_REPLY

        if self.last_bot_reply_timestamp:
            cooldown_elapsed = current_time - self.last_bot_reply_timestamp
            if cooldown_elapsed < self.cooldown_seconds:
                logger.debug(
                    "Cooldown active: %.1fs / %.1fs", cooldown_elapsed, self.cooldown_seconds
                )
                return _NO_REPLY

        logger.info("Considering a volunteered reply (silence %.1fs)", silence_elapsed)
        return TriggerDecision(True, TriggerReason.VOLUNTEERED)

    def should_trigger_reply(
        self,
        current_time: float | None = None,
        *,
        has_pending_transcript: bool = True,
    ) -> bool:
        """Boolean form of :meth:`evaluate`."""
        return bool(
            self.evaluate(current_time, has_pending_transcript=has_pending_transcript)
        )

    def get_state(self) -> TriggerState:
        """Get current state machine state."""
        return self.state

    def set_state(self, state: TriggerState) -> None:
        """Set state machine state."""
        self.state = state
        logger.debug("State changed to: %s", state.value)

    def snapshot_for_ui(self, now: float | None = None) -> tuple[bool, float, float]:
        """
        Values for dashboards: mention-window active, seconds left in that window,
        and seconds remaining on the post-reply cooldown timer.

        Mention window is active only in MENTION_WAITING with a recorded timestamp.
        Cooldown uses time since the last bot reply; 0.0 if no reply yet or cooldown elapsed.
        """
        if now is None:
            now = time.time()

        mention_waiting = (
            self.state == TriggerState.MENTION_WAITING
            and self.mention_detected_timestamp is not None
        )
        mention_left = 0.0
        if mention_waiting and self.mention_detected_timestamp is not None:
            mention_left = max(
                0.0, self.mention_window_seconds - (now - self.mention_detected_timestamp)
            )

        cooldown_left = 0.0
        if self.last_bot_reply_timestamp is not None:
            cooldown_left = max(
                0.0, self.cooldown_seconds - (now - self.last_bot_reply_timestamp)
            )

        return mention_waiting, mention_left, cooldown_left
