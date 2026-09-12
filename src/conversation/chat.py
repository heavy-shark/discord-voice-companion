"""GenAI chat session management."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Optional

from google import genai
from google.genai import types

from ..models import Persona, UsageCounters
from ..storage.reply_locale import is_auto_reply_locale

logger = logging.getLogger(__name__)

# The gate answers one word, but "one word" is not the same as "few tokens" on a
# thinking model: reasoning tokens come out of the same budget and leave nothing
# visible behind. Thinking is switched off for the usual call; the retry that
# allows it needs room for the reasoning *and* the answer.
_GATE_MAX_OUTPUT_TOKENS = 8
_GATE_THINKING_MAX_OUTPUT_TOKENS = 512


def _response_text(response: Any) -> str:
    return (getattr(response, "text", None) or "").strip().upper()


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "unknown"
    return str(getattr(candidates[0], "finish_reason", "unknown"))


def _hit_output_cap(response: Any) -> bool:
    """True when the model stopped because it ran out of output budget."""
    return _finish_reason(response).upper().endswith("MAX_TOKENS")

# Human-readable names for the tags the companion is expected to hear. Prompting
# with "Ukrainian" is far more reliable than prompting with "uk-UA".
_LANGUAGE_NAMES: dict[str, str] = {
    "uk": "Ukrainian",
    "ru": "Russian",
    "en": "English",
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "da": "Danish",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "cs": "Czech",
    "tr": "Turkish",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}

DEFAULT_SPOKEN_LANGUAGES: tuple[str, ...] = ("uk-UA", "ru-RU", "en-US")


def language_name(tag: str) -> str:
    """Map a BCP-47 tag to an English language name, falling back to the tag."""
    cleaned = (tag or "").strip()
    if not cleaned:
        return ""
    primary = cleaned.replace("_", "-").split("-", 1)[0].lower()
    return _LANGUAGE_NAMES.get(primary, cleaned)


def _language_name_list(tags: Sequence[str]) -> str:
    """"Ukrainian, Russian or English" from a list of BCP-47 tags."""
    names: list[str] = []
    for tag in tags:
        name = language_name(tag)
        if name and name not in names:
            names.append(name)
    if not names:
        return "the language the speaker used"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" or {names[-1]}"


def _mirror_language_rule(spoken_languages: Sequence[str]) -> str:
    """The 'answer in whatever they spoke' rule used when no locale is pinned."""
    allowed = _language_name_list(spoken_languages)
    return (
        "[OUTPUT LANGUAGE — HIGHEST PRIORITY] Reply in the same language the last "
        f"speaker used. The people here speak {allowed}, so expect to switch "
        "between them mid-conversation. Match the speaker turn by turn: a "
        "Ukrainian question gets a Ukrainian answer, a Russian question gets a "
        "Russian answer, an English question gets an English answer. Never "
        "translate or comment on which language you are using, and never default "
        "to English just because these instructions are in English."
    )


def _pinned_language_rule(locale_tag: str) -> str:
    """The 'always answer in one locale' rule used when a tag is configured."""
    name = language_name(locale_tag)
    return (
        "[OUTPUT LANGUAGE — HIGHEST PRIORITY] Write every assistant message only in "
        f'{name} (locale "{locale_tag}"). Do not mirror another language from the '
        "room or from these instructions, even if people speak something else."
    )


def language_rule(
    reply_locale: str | None,
    spoken_languages: Sequence[str] = DEFAULT_SPOKEN_LANGUAGES,
) -> str:
    """Language instruction for the current configuration."""
    if is_auto_reply_locale(reply_locale):
        return _mirror_language_rule(spoken_languages)
    return _pinned_language_rule(str(reply_locale).strip())


def format_transcript_user_message(
    transcript_block: str,
    *,
    reply_locale: str | None = None,
    spoken_languages: Sequence[str] = DEFAULT_SPOKEN_LANGUAGES,
) -> str:
    """Build the user turn sent after each reply trigger.

    The chat session keeps multi-turn history; each trigger sends one block
    describing what was said since the bot last spoke.
    """
    block = transcript_block.strip()
    intro = (
        "Here is what people in the voice channel said since your last reply "
        '(each line is "Username: text"). Give a brief spoken reply in '
        "1–2 short sentences that fits naturally in the conversation."
    )
    if is_auto_reply_locale(reply_locale):
        closing = (
            "Answer in the same language as the last line above. Do not switch "
            "languages and do not mention the language."
        )
    else:
        tag = str(reply_locale).strip()
        name = language_name(tag)
        intro = (
            f"Your reply must be in {name} (locale {tag}). Do not mirror the "
            "transcript language if it differs. " + intro
        )
        closing = f"Reply only in {name}; no other language."
    return "\n".join([intro, "", block, "", closing])


def join_greeting_chat_history(greeting_prompt: str, greeting_text: str) -> list[types.Content]:
    """Seed multiturn chat so later replies see the join exchange."""
    return [
        types.Content(role="user", parts=[types.Part(text=greeting_prompt)]),
        types.Content(role="model", parts=[types.Part(text=greeting_text)]),
    ]


def _greeting_language_instruction(
    reply_locale: str | None,
    spoken_languages: Sequence[str],
) -> str:
    """Language line for greetings, where there is no speaker turn to mirror yet."""
    if is_auto_reply_locale(reply_locale):
        default_name = language_name(
            spoken_languages[0] if spoken_languages else "en-US"
        )
        return (
            f"Nobody has spoken yet, so greet them in {default_name}. After this, "
            "switch to whatever language each speaker uses."
        )
    tag = str(reply_locale).strip()
    return f"Write the entire greeting only in {language_name(tag)} (locale {tag})."


def format_join_greeting_prompt(
    names_str: str,
    *,
    reply_locale: str | None = None,
    spoken_languages: Sequence[str] = DEFAULT_SPOKEN_LANGUAGES,
) -> str:
    """Build the user turn for the join greeting (first model output in the session)."""
    body = (
        "You just joined this Discord voice channel. "
        f"Users currently present: {names_str}. "
        "Give a brief, friendly spoken-style hello in 1-2 sentences."
    )
    return f"{body}\n\n{_greeting_language_instruction(reply_locale, spoken_languages)}"


def format_user_join_greeting_prompt(
    display_name: str,
    *,
    reply_locale: str | None = None,
    spoken_languages: Sequence[str] = DEFAULT_SPOKEN_LANGUAGES,
) -> str:
    """User turn for greeting someone who just connected to the voice channel (by nickname)."""
    name = (display_name or "").strip() or "them"
    body = (
        f'A user just joined this Discord voice channel. Their display name is "{name}". '
        "Give one brief, friendly spoken-style welcome (1-2 sentences) that feels unique to them "
        f'(use their name naturally — "{name}"). Do not copy a generic template verbatim.'
    )
    return f"{body}\n\n{_greeting_language_instruction(reply_locale, spoken_languages)}"


class GenAIChatManager:
    """
    Manages GenAI chat sessions for multi-turn conversation.

    Maintains conversation state, applies persona instructions,
    tracks token usage via usage_metadata.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        reply_locale: Optional[str] = None,
        spoken_languages: Sequence[str] = DEFAULT_SPOKEN_LANGUAGES,
    ) -> None:
        """
        Initialize GenAI client.

        Args:
            api_key: Gemini API key (optional, reads GOOGLE_GEMINI_API_KEY if
                not provided)
            reply_locale: BCP-47 tag (e.g. en-US) to pin every reply to one
                language, or ``"auto"`` to mirror whatever each speaker used.
            spoken_languages: Languages expected in the channel; used to tell the
                model which ones to switch between in ``"auto"`` mode.
        """
        resolved = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        if not resolved:
            raise ValueError(
                "GOOGLE_GEMINI_API_KEY must be set for GenAI (environment or constructor)."
            )
        self._client = genai.Client(api_key=resolved)
        self._chat: Any = None
        self.current_persona: Optional[Persona] = None
        self.usage_counters = UsageCounters()
        self._reply_locale = (reply_locale or "").strip() or None
        self._spoken_languages = tuple(spoken_languages) or DEFAULT_SPOKEN_LANGUAGES
        logger.info(
            "GenAIChatManager initialized (reply language: %s)",
            self._reply_locale or "auto",
        )

    @property
    def spoken_languages(self) -> tuple[str, ...]:
        """Languages the companion expects to hear and answer in."""
        return self._spoken_languages

    def _accumulate_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_token_count", None)
        candidates = getattr(usage, "candidates_token_count", None)
        total = getattr(usage, "total_token_count", None)
        used_breakdown = False
        if isinstance(prompt, int) and prompt > 0:
            self.usage_counters.input_tokens += prompt
            used_breakdown = True
        if isinstance(candidates, int) and candidates > 0:
            self.usage_counters.output_tokens += candidates
            used_breakdown = True
        if not used_breakdown and isinstance(total, int) and total > 0:
            self.usage_counters.output_tokens += total

    def get_token_usage_breakdown(self) -> tuple[int, int]:
        """Cumulative GenAI prompt (input) and candidates (output) tokens."""
        return (self.usage_counters.input_tokens, self.usage_counters.output_tokens)

    async def create_chat_session(
        self,
        persona: Persona,
        *,
        history: Optional[list[types.Content]] = None,
    ) -> None:
        """
        Create a new async chat session with persona instructions.

        Args:
            persona: Persona configuration with system instruction and model id
            history: Optional prior turns (e.g. join greeting) for multiturn context.
        """
        self.current_persona = persona
        rule = language_rule(self._reply_locale, self._spoken_languages)
        # The language rule brackets the persona so it outranks both the persona's
        # own tone and any language the transcript happens to be dominated by.
        system_instruction = "\n\n".join(
            part for part in (rule, persona.system_instruction or "", rule) if part
        )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.65,
        )
        # google-genai: aio.chats.create returns AsyncChat synchronously; await send_message instead.
        self._chat = self._client.aio.chats.create(
            model=persona.genai_model,
            config=config,
            history=list(history) if history else [],
        )
        logger.info("Chat session created with persona '%s'", persona.display_name)

    async def send_message(self, text: str) -> str:
        """
        Send message to chat and get response text.

        Args:
            text: Message text

        Returns:
            Response text from model (may be empty if blocked)
        """
        if self._chat is None:
            raise RuntimeError("Chat session not initialized; call create_chat_session first.")

        logger.debug("Sending message (%s chars)", len(text))
        response = await self._chat.send_message(text)
        self._accumulate_usage(response)

        reply = (getattr(response, "text", None) or "").strip()
        logger.debug("Received reply (%s chars)", len(reply))
        return reply

    async def should_volunteer(
        self,
        transcript_text: str,
        *,
        model: str | None = None,
    ) -> bool:
        """Decide whether an unprompted interjection is actually worth making.

        Runs as a standalone one-token call rather than a chat turn, so declined
        interjections leave no trace in the conversation history. This is what
        makes a high talkativeness setting usable: the timing dial decides *how
        often the bot considers speaking*, and this decides *whether it should*.
        """
        block = transcript_text.strip()
        if not block:
            return False

        persona_name = (
            self.current_persona.display_name if self.current_persona else "the assistant"
        )
        prompt = (
            f"You are {persona_name}, sitting in a Discord voice call. Nobody "
            "addressed you directly. Here is what was just said:\n\n"
            f"{block}\n\n"
            "Would joining in right now add something the others would actually "
            "want to hear - answering an open question, catching a mistake, or a "
            "genuinely fitting remark? Interrupting a flowing conversation with "
            "filler, agreement or small talk is not worth it.\n"
            'Answer with exactly one word: "YES" or "NO".'
        )
        gate_model = model or (
            self.current_persona.genai_model if self.current_persona else "gemini-2.5-flash"
        )

        try:
            response = await self._gate_request(prompt, gate_model, thinking=False)
            verdict = _response_text(response)
            if not verdict and _hit_output_cap(response):
                # A thinking model spent the whole budget reasoning and emitted
                # nothing. Left unhandled this reads as "NO" every single time,
                # which silently mutes the bot for good.
                logger.warning(
                    "Volunteer gate produced no text within %d tokens (finish=%s); retrying "
                    "with room for the model to think.",
                    _GATE_MAX_OUTPUT_TOKENS,
                    _finish_reason(response),
                )
                response = await self._gate_request(prompt, gate_model, thinking=True)
                verdict = _response_text(response)
        except Exception as exc:
            # Staying quiet is the safe failure mode for an uninvited remark.
            logger.warning("Volunteer gate failed (%s); staying quiet.", exc)
            return False

        if not verdict:
            logger.warning(
                "Volunteer gate returned no verdict (finish=%s); staying quiet.",
                _finish_reason(response),
            )
            return False

        decision = verdict.startswith("YES")
        logger.info("Volunteer gate: %s -> %s", verdict, decision)
        return decision

    async def _gate_request(self, prompt: str, gate_model: str, *, thinking: bool) -> Any:
        """One volunteer-gate call. ``thinking=False`` is the cheap, usual path."""
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=_GATE_THINKING_MAX_OUTPUT_TOKENS if thinking else _GATE_MAX_OUTPUT_TOKENS,
            system_instruction=(
                "You gate an AI companion's unprompted interjections. "
                "Reply with YES or NO and nothing else. When unsure, say NO."
            ),
        )
        if not thinking:
            # Without this, a thinking model burns the whole output budget on
            # reasoning tokens and returns empty text.
            config.thinking_config = types.ThinkingConfig(thinking_budget=0)

        try:
            response = await self._client.aio.models.generate_content(
                model=gate_model,
                contents=prompt,
                config=config,
            )
        except Exception:
            if thinking:
                raise
            # Some models refuse thinking_budget=0; fall back to letting them think.
            # That path accounts for its own usage, hence the early return.
            logger.debug("Gate model %s rejected thinking_budget=0; retrying.", gate_model)
            return await self._gate_request(prompt, gate_model, thinking=True)

        self._accumulate_usage(response)
        return response

    async def count_tokens(self, text: str) -> int:
        """
        Rough character-based estimate (optional; preflight counting is SDK-specific).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        logger.debug("Token estimate for message: %s chars", len(text))
        return max(1, len(text) // 4)

    def get_token_usage(self) -> int:
        """Get total tokens used in this session."""
        return self.usage_counters.total_tokens

    def format_conversation_prompt(self, transcript_text: str) -> str:
        """Format merged transcript into the user message for the model."""
        return format_transcript_user_message(
            transcript_text,
            reply_locale=self._reply_locale,
            spoken_languages=self._spoken_languages,
        )

    def end_chat_session(self) -> None:
        """End current chat session."""
        self._chat = None
        logger.info("Chat session ended")
