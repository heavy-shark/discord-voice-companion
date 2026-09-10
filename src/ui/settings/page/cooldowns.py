"""Conversation behaviour settings page widget."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Switch


class CooldownsSettingsPage(Vertical):
    """Scrollable conversation-behaviour settings with a fixed action row."""

    def __init__(self) -> None:
        super().__init__(classes="settings_page")

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="settings_scroll"):
            with Vertical(classes="settings_form"):
                with Vertical(classes="settings_field_group"):
                    yield Label("Talkativeness (0-100)")
                    yield Input(id="cooldown_talkativeness", placeholder="65")
                    yield Label(
                        "0 = speaks only when addressed by name. Higher values let it "
                        "join in on its own; every uninvited remark still has to pass a "
                        '"is this worth saying?" check.',
                        classes="settings_help",
                    )
                with Vertical(classes="settings_field_group"):
                    yield Label("Reply silence seconds (blank = from talkativeness)")
                    yield Input(id="cooldown_reply_silence_seconds", placeholder="auto")
                with Vertical(classes="settings_field_group"):
                    yield Label("Reply cooldown seconds (blank = from talkativeness)")
                    yield Input(id="cooldown_reply_cooldown_seconds", placeholder="auto")
                with Vertical(classes="settings_field_group"):
                    yield Label("Mention window seconds")
                    yield Input(id="cooldown_mention_window_seconds", placeholder="30")
                with Horizontal(
                    classes="settings_field_group settings_toggle_row", id="cooldown_barge_in_row"
                ):
                    yield Switch(value=True, id="cooldown_barge_in")
                    with Vertical(classes="settings_side_label"):
                        yield Label("Stop talking when someone speaks over the bot")
                with Horizontal(
                    classes="settings_field_group settings_toggle_row", id="cooldown_greet_row"
                ):
                    yield Switch(value=True, id="cooldown_greet_on_join")
                    with Vertical(classes="settings_side_label"):
                        yield Label("Greet users when they join the voice channel")
        with Horizontal(classes="settings_action_row"):
            yield Button("Save Behaviour", id="btn_save_cooldowns", variant="success")
