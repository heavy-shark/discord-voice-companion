"""Slash commands that drive the companion from inside Discord.

On a server there is no terminal to pick a voice channel from and no easy way to
look up a channel id, so ``/join`` takes the caller's own voice channel. That is
the whole setup flow: invite the bot, sit in a voice channel, type ``/join``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import discord
from discord import app_commands

from ..session.service import CompanionService

logger = logging.getLogger(__name__)

# Shown once when the bot joins, so people in the channel know what is happening
# to their audio even when per-user consent is not enforced.
JOIN_NOTICE = (
    "Listening in **{channel}**. Voice in this channel is transcribed by Google "
    "Gemini and answered out loud via ElevenLabs. Use `/leave` to stop."
)


def register_companion_commands(
    tree: app_commands.CommandTree,
    service: CompanionService,
    *,
    activity_sink: Callable[[str], None] | None = None,
) -> None:
    """Register ``/join``, ``/leave`` and ``/status`` on *tree*."""

    def emit(message: str) -> None:
        logger.info("[companion] %s", message)
        if activity_sink is not None:
            try:
                activity_sink(message)
            except Exception:
                logger.exception("activity sink failed")

    @tree.command(name="join", description="Join your voice channel and start listening.")
    async def join(interaction: discord.Interaction) -> None:
        channel = _caller_voice_channel(interaction)
        if channel is None:
            await interaction.response.send_message(
                "Join a voice channel first, then run `/join` again.", ephemeral=True
            )
            return

        permissions = channel.permissions_for(channel.guild.me)
        missing = [
            name
            for name, allowed in (("Connect", permissions.connect), ("Speak", permissions.speak))
            if not allowed
        ]
        if missing:
            await interaction.response.send_message(
                f"I need the {' and '.join(missing)} permission in **{channel.name}**.",
                ephemeral=True,
            )
            return

        # Joining and negotiating voice takes longer than the 3s interaction window.
        await interaction.response.defer(thinking=True)
        try:
            await service.join(channel.id, channel.name)
        except Exception as exc:
            emit(f"/join failed for {channel.name}: {exc}")
            logger.exception("/join failed")
            await interaction.followup.send(f"Could not join **{channel.name}**: {exc}")
            return

        emit(f"Joined {channel.name} ({channel.id})")
        await interaction.followup.send(JOIN_NOTICE.format(channel=channel.name))

    @tree.command(name="leave", description="Stop listening and leave the voice channel.")
    async def leave(interaction: discord.Interaction) -> None:
        if not service.is_running:
            await interaction.response.send_message("I am not in a voice channel.", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        try:
            await service.leave()
        except Exception as exc:
            logger.exception("/leave failed")
            await interaction.followup.send(f"Could not leave cleanly: {exc}")
            return

        emit("Left the voice channel")
        await interaction.followup.send("Left the voice channel.")

    @tree.command(name="status", description="Show what the companion is doing.")
    async def status(interaction: discord.Interaction) -> None:
        snapshot = service.status()
        if not snapshot.is_active:
            message = "Idle — not in a voice channel. Use `/join` to start."
            if snapshot.last_error:
                message += f"\nLast error: `{snapshot.last_error}`"
        else:
            message = (
                f"In **{snapshot.channel_name}** as **{snapshot.persona_name or 'companion'}** "
                f"— {snapshot.status_line}."
            )
        await interaction.response.send_message(message, ephemeral=True)


def _caller_voice_channel(interaction: discord.Interaction) -> discord.VoiceChannel | None:
    """The voice channel the person running the command is sitting in."""
    member = interaction.user
    if not isinstance(member, discord.Member):
        return None
    voice_state = member.voice
    if voice_state is None:
        return None
    channel = voice_state.channel
    if not isinstance(channel, discord.VoiceChannel):
        return None
    return channel
