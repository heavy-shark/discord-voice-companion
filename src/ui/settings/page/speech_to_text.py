"""Speech-to-Text settings page widget."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Select, Switch

from src.storage import (
    DEFAULT_STT_BATCH_MODEL,
    DEFAULT_STT_LANGUAGE_HINTS,
    DEFAULT_STT_LIVE_MODEL,
)


class SpeechToTextSettingsPage(Vertical):
    """Scrollable Speech-to-Text settings with a fixed action row."""

    def __init__(self) -> None:
        super().__init__(classes="settings_page")

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="settings_scroll"):
            with Vertical(classes="settings_form"):
                with Vertical(classes="settings_field_group"):
                    yield Label("Transcription backend")
                    yield Select(
                        [
                            ("live (streaming, lowest latency)", "live"),
                            ("batch (one request per utterance)", "batch"),
                        ],
                        id="stt_backend",
                        prompt="Select transcription backend...",
                    )
                with Vertical(classes="settings_field_group"):
                    yield Label("Language hints (BCP-47, comma-separated; blank = auto-detect)")
                    yield Input(
                        id="stt_language_hints",
                        placeholder=", ".join(DEFAULT_STT_LANGUAGE_HINTS),
                    )
                with Vertical(id="stt_live_fields"):
                    with Vertical(classes="settings_field_group"):
                        yield Label("Live transcription model")
                        yield Input(
                            id="stt_live_model",
                            placeholder=DEFAULT_STT_LIVE_MODEL,
                        )
                    with Vertical(classes="settings_field_group"):
                        yield Label("Show partial transcripts while someone is speaking")
                        yield Switch(id="stt_emit_interim")
                with Vertical(id="stt_batch_fields"):
                    with Vertical(classes="settings_field_group"):
                        yield Label("Batch transcription model")
                        yield Input(
                            id="stt_batch_model",
                            placeholder=DEFAULT_STT_BATCH_MODEL,
                        )
        with Horizontal(classes="settings_action_row"):
            yield Button("Save Speech-to-Text", id="btn_save_stt", variant="success")
