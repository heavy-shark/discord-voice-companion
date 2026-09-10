"""Tests for conversation module."""

import json
import time
from pathlib import Path

import pytest

from src.conversation import (
    DEFAULT_TALKATIVENESS,
    PersonaManager,
    ReplyTriggerPolicy,
    TalkativenessProfile,
    TriggerReason,
    TriggerState,
)
from src.conversation.log import ConversationLog
from src.models import Persona, TranscriptSegment
from src.storage.transcripts import TranscriptSessionWriter


class TestReplyTriggerPolicy:
    """Test reply trigger policy."""

    def test_policy_initialization(self):
        """Timings default to whatever the talkativeness dial implies."""
        policy = ReplyTriggerPolicy()
        expected = TalkativenessProfile.from_dial(DEFAULT_TALKATIVENESS)
        assert policy.state == TriggerState.LISTENING
        assert policy.talkativeness == DEFAULT_TALKATIVENESS
        assert policy.silence_timeout_seconds == expected.silence_timeout_seconds
        assert policy.cooldown_seconds == expected.cooldown_seconds

    def test_explicit_timings_override_the_dial(self):
        policy = ReplyTriggerPolicy(silence_timeout_seconds=4.0, cooldown_seconds=90.0)
        assert policy.silence_timeout_seconds == 4.0
        assert policy.cooldown_seconds == 90.0

    def test_higher_talkativeness_shortens_both_timers(self):
        quiet = ReplyTriggerPolicy(talkativeness=10)
        chatty = ReplyTriggerPolicy(talkativeness=90)
        assert chatty.silence_timeout_seconds < quiet.silence_timeout_seconds
        assert chatty.cooldown_seconds < quiet.cooldown_seconds

    def test_talkativeness_is_clamped_to_the_valid_range(self):
        assert ReplyTriggerPolicy(talkativeness=-50).talkativeness == 0
        assert ReplyTriggerPolicy(talkativeness=500).talkativeness == 100

    def test_the_dial_endpoints_are_the_configured_extremes(self):
        quiet = TalkativenessProfile.from_dial(0)
        chatty = TalkativenessProfile.from_dial(100)
        assert quiet.silence_timeout_seconds == pytest.approx(6.0)
        assert quiet.cooldown_seconds == pytest.approx(180.0)
        assert chatty.silence_timeout_seconds == pytest.approx(1.6)
        assert chatty.cooldown_seconds == pytest.approx(8.0)

    def test_the_middle_of_the_dial_is_not_dragged_toward_the_quiet_end(self):
        """Linear interpolation would put the midpoint cooldown at 94s, which
        does not read as 'moderately chatty' to anyone listening."""
        middle = TalkativenessProfile.from_dial(50)
        assert 30.0 < middle.cooldown_seconds < 45.0
        assert 2.5 < middle.silence_timeout_seconds < 3.5

    def test_the_default_dial_behaves_like_an_engaged_participant(self):
        profile = TalkativenessProfile.from_dial(DEFAULT_TALKATIVENESS)
        assert profile.volunteers
        assert profile.silence_timeout_seconds < 3.0
        assert profile.cooldown_seconds < 30.0

    def test_zero_talkativeness_never_volunteers(self):
        """At 0 the bot answers when addressed and otherwise stays silent."""
        policy = ReplyTriggerPolicy(talkativeness=0, silence_timeout_seconds=0.0)
        policy.record_human_speech()
        assert policy.evaluate(has_pending_transcript=True).should_reply is False

        policy.record_mention()
        decision = policy.evaluate(has_pending_transcript=True)
        assert decision.should_reply is True
        assert decision.reason is TriggerReason.ADDRESSED

    def test_unprompted_trigger_is_reported_as_volunteered(self):
        policy = ReplyTriggerPolicy(
            talkativeness=90, silence_timeout_seconds=0.0, cooldown_seconds=0.0
        )
        policy.record_human_speech()
        decision = policy.evaluate(has_pending_transcript=True)
        assert decision.should_reply is True
        assert decision.reason is TriggerReason.VOLUNTEERED

    def test_declining_to_volunteer_starts_the_cooldown(self):
        policy = ReplyTriggerPolicy(
            talkativeness=90, silence_timeout_seconds=0.0, cooldown_seconds=60.0
        )
        policy.record_human_speech()
        assert policy.evaluate(has_pending_transcript=True).should_reply is True

        policy.record_declined_reply()
        # Same pending transcript, but the bot just passed on it.
        assert policy.evaluate(has_pending_transcript=True).should_reply is False

    def test_no_trigger_without_speech(self):
        """Test no trigger when no speech recorded."""
        policy = ReplyTriggerPolicy()
        assert policy.should_trigger_reply() is False

    def test_trigger_after_silence(self):
        """Test trigger after silence period."""
        policy = ReplyTriggerPolicy(silence_timeout_seconds=0.1)
        policy.record_human_speech()

        time.sleep(0.2)
        assert policy.should_trigger_reply(has_pending_transcript=True) is True

    def test_no_trigger_without_pending_transcript(self):
        """Silence alone must not fire without user transcript since last bot turn."""
        policy = ReplyTriggerPolicy(silence_timeout_seconds=0.0, cooldown_seconds=0.0)
        policy.last_human_speech_timestamp = time.time() - 10.0
        assert policy.should_trigger_reply(has_pending_transcript=False) is False

    def test_cooldown_blocks_default_path(self):
        """Default path respects cooldown after a bot reply."""
        policy = ReplyTriggerPolicy(
            silence_timeout_seconds=1.0,
            cooldown_seconds=3600.0,
            mention_window_seconds=30.0,
        )
        now = time.time()
        policy.last_human_speech_timestamp = now - 10.0
        policy.last_bot_reply_timestamp = now - 5.0
        assert policy.should_trigger_reply(current_time=now, has_pending_transcript=True) is False

    def test_first_reply_allowed_without_prior_bot_timestamp(self):
        """Cooldown check is skipped when the bot has not replied yet."""
        policy = ReplyTriggerPolicy(
            silence_timeout_seconds=1.0,
            cooldown_seconds=3600.0,
        )
        now = time.time()
        policy.last_human_speech_timestamp = now - 10.0
        policy.last_bot_reply_timestamp = None
        assert policy.should_trigger_reply(current_time=now, has_pending_transcript=True) is True

    def test_mention_silence_resets_to_listening(self):
        """After mention-window silence trigger, state returns to LISTENING."""
        policy = ReplyTriggerPolicy(silence_timeout_seconds=1.0, mention_window_seconds=30.0)
        now = time.time()
        policy.state = TriggerState.MENTION_WAITING
        policy.mention_detected_timestamp = now
        policy.last_human_speech_timestamp = now - 5.0
        assert policy.should_trigger_reply(current_time=now, has_pending_transcript=True) is True
        assert policy.get_state() == TriggerState.LISTENING
        assert policy.mention_detected_timestamp is None

    def test_mention_timeout_clears_timestamp(self):
        policy = ReplyTriggerPolicy(
            silence_timeout_seconds=100.0,
            mention_window_seconds=10.0,
        )
        now = time.time()
        policy.state = TriggerState.MENTION_WAITING
        policy.mention_detected_timestamp = now
        policy.last_human_speech_timestamp = now
        assert policy.should_trigger_reply(current_time=now + 11.0, has_pending_transcript=True) is True
        assert policy.get_state() == TriggerState.LISTENING
        assert policy.mention_detected_timestamp is None

    def test_record_bot_reply_resets_mention_state(self):
        policy = ReplyTriggerPolicy()
        policy.state = TriggerState.MENTION_WAITING
        policy.mention_detected_timestamp = time.time()
        policy.record_bot_reply()
        assert policy.get_state() == TriggerState.LISTENING
        assert policy.mention_detected_timestamp is None

    def test_multi_mention_trigger_substring(self):
        policy = ReplyTriggerPolicy()
        policy.set_mention_triggers(["L.O.C", "loc", "Liam"])
        assert policy.text_contains_mention_trigger("hey loc?")
        assert policy.text_contains_mention_trigger("Hvad så, LOC?")
        assert policy.text_contains_mention_trigger("talk to liam now")
        assert not policy.text_contains_mention_trigger("no match here")

    def test_set_mention_triggers_dedupes_case_insensitive(self):
        policy = ReplyTriggerPolicy()
        policy.set_mention_triggers(["Loc", "LOC", "loc"])
        assert policy.mention_triggers == ["loc"]


class TestPersonaManager:
    """Test persona management."""

    def test_manager_initialization(self):
        """Test PersonaManager initialization."""
        personas = [
            Persona(
                persona_id="test",
                display_name="Test Persona",
                system_instruction="Test",
                genai_model="gemini-2.5-flash",
                elevenlabs_voice_id="voice123",
            )
        ]
        manager = PersonaManager(personas)
        assert len(manager.list_personas()) == 1

    def test_get_persona(self):
        """Test getting persona by ID."""
        persona = Persona(
            persona_id="test",
            display_name="Test",
            system_instruction="Test",
            genai_model="gemini-2.5-flash",
            elevenlabs_voice_id="voice123",
        )
        manager = PersonaManager([persona])
        found = manager.get_persona("test")
        assert found is not None
        assert found.display_name == "Test"

    def test_switch_persona(self):
        """Test switching personas."""
        personas = [
            Persona(
                persona_id="p1",
                display_name="First",
                system_instruction="First",
                genai_model="gemini-2.5-flash",
                elevenlabs_voice_id="v1",
            ),
            Persona(
                persona_id="p2",
                display_name="Second",
                system_instruction="Second",
                genai_model="gemini-2.5-flash",
                elevenlabs_voice_id="v2",
            ),
        ]
        manager = PersonaManager(personas)
        manager.set_current_persona("p2")
        assert manager.get_current_persona().persona_id == "p2"


class TestConversationLog:
    """Test transcript merging behavior."""

    def test_merge_segments_orders_by_end_time(self):
        """Merged transcript text should be ordered deterministically."""
        log = ConversationLog()
        log.add_segment(
            TranscriptSegment(
                user_id=2,
                username="Bob",
                text="second",
                start_ts=2.0,
                end_ts=3.0,
                is_final=True,
            )
        )
        log.add_segment(
            TranscriptSegment(
                user_id=1,
                username="Alice",
                text="first",
                start_ts=1.0,
                end_ts=2.0,
                is_final=True,
            )
        )

        assert log.merge_segments() == "Alice: first\nBob: second"

    def test_merge_segments_only_returns_since_last_bot_turn(self):
        """Bot turn boundaries should keep older transcript lines out of new prompts."""
        log = ConversationLog()
        log.add_segment(
            TranscriptSegment(
                user_id=1,
                username="Alice",
                text="before",
                start_ts=1.0,
                end_ts=2.0,
                is_final=True,
            )
        )
        log.add_bot_turn("reply")
        log.add_segment(
            TranscriptSegment(
                user_id=2,
                username="Bob",
                text="after",
                start_ts=3.0,
                end_ts=4.0,
                is_final=True,
            )
        )

        assert log.merge_segments() == "Bob: after"

    def test_has_pending_since_bot(self):
        log = ConversationLog()
        assert log.has_pending_since_bot() is False
        log.add_segment(
            TranscriptSegment(
                user_id=1,
                username="Alice",
                text="hi",
                start_ts=1.0,
                end_ts=2.0,
                is_final=True,
            )
        )
        assert log.has_pending_since_bot() is True
        log.add_bot_turn("bot said")
        assert log.has_pending_since_bot() is False


class TestTranscriptSessionWriterExtras:
    """Transcript JSON helpers used in conversation mode."""

    def test_bot_reply_and_total_tokens(self, tmp_path: Path) -> None:
        writer = TranscriptSessionWriter(
            guild_id=1,
            guild_name="Guild",
            channel_id=2,
            channel_name="vc",
            transcripts_dir=tmp_path,
        )
        writer.add_bot_reply("Hello everyone", label="greeting")
        writer.set_total_tokens(99)
        payload = json.loads(writer.path.read_text(encoding="utf-8"))
        assert len(payload["bot_replies"]) == 1
        assert payload["bot_replies"][0]["label"] == "greeting"
        assert payload["usage"]["total_tokens"] == 99
        assert payload["usage"]["genai_input_tokens"] == 99
        assert payload["usage"]["genai_output_tokens"] == 0

    def test_add_token_usage_increments(self, tmp_path: Path) -> None:
        writer = TranscriptSessionWriter(
            guild_id=1,
            guild_name="G",
            channel_id=2,
            channel_name="c",
            transcripts_dir=tmp_path,
        )
        writer.add_token_usage(10)
        writer.add_token_usage(5)
        payload = json.loads(writer.path.read_text(encoding="utf-8"))
        assert payload["usage"]["total_tokens"] == 15
        assert payload["usage"]["genai_input_tokens"] == 0
        assert payload["usage"]["genai_output_tokens"] == 15
