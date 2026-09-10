"""Conversation package.

Manages multi-turn chat sessions, conversation state, reply triggers,
and conversation history logging.
"""

from .chat import GenAIChatManager
from .log import ConversationLog
from .persona import PersonaManager
from .policy import (
    DEFAULT_TALKATIVENESS,
    ReplyTriggerPolicy,
    TalkativenessProfile,
    TriggerDecision,
    TriggerReason,
    TriggerState,
)

__all__ = [
    "DEFAULT_TALKATIVENESS",
    "GenAIChatManager",
    "ReplyTriggerPolicy",
    "TalkativenessProfile",
    "TriggerDecision",
    "TriggerReason",
    "TriggerState",
    "ConversationLog",
    "PersonaManager",
]
