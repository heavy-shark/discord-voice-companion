"""Storage and persistence package.

Handles settings management, configuration persistence, and data storage.
"""

from .consent import ConsentAcceptEvent, ConsentStore
from .settings import (
    AUTO_REPLY_LOCALE,
    DEFAULT_STT_BACKEND,
    DEFAULT_STT_BATCH_MODEL,
    DEFAULT_STT_LANGUAGE_HINTS,
    DEFAULT_STT_LIVE_MODEL,
    SettingsStore,
)
from .transcripts import (
    TranscriptScrubStats,
    TranscriptSessionWriter,
    scrub_user_data_from_transcripts,
)

__all__ = [
    "AUTO_REPLY_LOCALE",
    "ConsentAcceptEvent",
    "ConsentStore",
    "DEFAULT_STT_BACKEND",
    "DEFAULT_STT_BATCH_MODEL",
    "DEFAULT_STT_LANGUAGE_HINTS",
    "DEFAULT_STT_LIVE_MODEL",
    "SettingsStore",
    "TranscriptScrubStats",
    "TranscriptSessionWriter",
    "scrub_user_data_from_transcripts",
]
