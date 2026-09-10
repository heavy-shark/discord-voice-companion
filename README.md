# Discord Voice Companion

A Discord bot that sits in a voice channel and takes part in the conversation: it
listens continuously, transcribes each speaker with **Gemini**, decides whether it
has something worth saying, and answers out loud through **ElevenLabs** — in
Ukrainian, Russian or English, matching whoever just spoke.

Forked from [Magnus5405/discord-audio-bot](https://github.com/Magnus5405/discord-audio-bot)
(MIT), which supplied the DAVE-capable voice receive path and the Textual control UI.

---

## Pipeline

```text
Discord voice receive (per-user Opus, DAVE-decrypted)
  → per-speaker utterance chunking (48 kHz stereo → 16 kHz mono PCM)
  → Gemini transcription (streaming Live API, or one request per utterance)
  → conversation log + reply policy
  → Gemini chat (persona + context)          [+ a cheap "worth saying?" gate]
  → ElevenLabs Flash v2.5 → 48 kHz PCM → straight into the voice channel
```

Everything except speech synthesis runs on a single Gemini API key.

---

## What makes it feel like a companion

**It never goes deaf.** Voice receive stays on while the bot is thinking and while
it is talking, so it hears follow-ups and interruptions instead of missing the
half of the conversation that happens during its own turn.

**It can be interrupted.** The moment a human starts speaking over it, playback
stops mid-sentence (`BOT_BARGE_IN`, on by default).

**It decides whether to speak, not just when.** One `talkativeness` dial (0-100)
sets how often the bot *considers* joining in; a separate one-token Gemini call
then decides whether this particular moment is worth it. That check runs outside
the chat session, so declined interjections leave no trace in the conversation
history. At `0` the bot is purely reactive and only answers when addressed by name.

**It follows the language of the room.** Transcription auto-detects, and replies
mirror the speaker turn by turn — a Ukrainian question gets a Ukrainian answer.
Set `BOT_REPLY_LANGUAGE` to a BCP-47 tag to pin one language instead.

**It starts speaking before it has finished synthesizing.** ElevenLabs streams
48 kHz PCM directly into a Discord audio source: no MP3 decode, no temp file, no
FFmpeg process per reply.

---

## Setup — three keys, then run

You need a **Discord bot account** (a user account / "selfbot" is a ToS violation
and, since DAVE, not workable anyway), and two API keys.

| # | Variable | Where to get it |
|---|---|---|
| 1 | `DISCORD_TOKEN` | [Developer Portal](https://discord.com/developers/applications) → your app → **Bot** → Reset Token |
| 2 | `GOOGLE_GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/) → Get API key |
| 3 | `ELEVENLABS_API_KEY` | [ElevenLabs API keys](https://elevenlabs.io/app/settings/api-keys) |

Put all three in **`.env`** (already in the repo, with the fields empty and
commented). Nothing else is required.

> ⚠️ `.env` is committed on purpose so a deployment that clones the repo picks up
> the config. **Keep the repository private.** If it is ever public, rotate all
> three keys.

**Inviting the bot.** In the Developer Portal → OAuth2 → URL Generator, tick
scopes **`bot`** and **`applications.commands`**, and bot permissions **Connect**
and **Speak**. Open the generated URL and add it to your server.

Set `DISCORD_SERVER_ID` in `.env` too — without it, slash commands register
globally and can take up to an hour to show up.

### Then, in Discord

1. Join a voice channel.
2. Type **`/join`**.

That is the whole flow. `/leave` disconnects, `/status` says what it is doing.
To skip the command entirely, set `DISCORD_VOICE_CHANNEL_ID` in `.env` and it
joins that channel on startup.

---

## Running it

### On a server / process manager

Point it at the repo and run:

```
python main.py
```

- **Working directory**: the repo root (where `main.py` and `.env` are).
- **Install step**: `pip install -r requirements.txt`.
- No arguments, no terminal interaction. It stays online and idles until `/join`.
- Handles SIGTERM cleanly, so restarts are safe.

### Locally, with a terminal

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python main.py                       # same headless mode
python -m src.main --tui             # dashboard with live metrics
python -m src.main --list-voice-channels
```

### Notes

- **Python 3.11+.**
- **FFmpeg is not needed** for the normal reply path — only for the
  `--audio-path` playback smoke test.
- Installing needs no Windows long-path support: the ElevenLabs Python SDK is
  deliberately not a dependency (it ships filenames long enough to break
  `pip install` on default Windows setups), and there is nothing to compile from
  source.

---

## Configuration

All of it lives in `.env`, and anything saved from the settings editor in
`settings.json` takes precedence.

### Transcription

| Key | Default | Meaning |
|---|---|---|
| `GEMINI_STT_BACKEND` | `live` | `live` streams into the Gemini Live API (lowest latency, emits partial text); `batch` sends one request per utterance |
| `GEMINI_STT_LIVE_MODEL` | `gemini-3.5-transcribe-live` | Model for the `live` backend |
| `GEMINI_STT_BATCH_MODEL` | `gemini-2.5-flash` | Model for the `batch` backend |
| `GEMINI_STT_LANGUAGE_HINTS` | `uk-UA,ru-RU,en-US` | Biases auto-detection; blank means unbiased |
| `STT_IDLE_TIMEOUT_SECONDS` | `1.2` | Silence gap that ends one speaker's utterance |

`gemini-3.5-transcribe-live` was in public preview at the time of writing. If your
key does not have it, startup fails with the list of models it *does* have — set
`GEMINI_STT_LIVE_MODEL` to one of those, or switch to the `batch` backend.

### Behaviour

| Key | Default | Meaning |
|---|---|---|
| `BOT_TALKATIVENESS` | `65` | 0-100. `0` = only answers when addressed |
| `BOT_REPLY_SILENCE_SECONDS` | derived | Override the silence gap before replying |
| `BOT_REPLY_COOLDOWN_SECONDS` | derived | Override the minimum gap between replies |
| `BOT_MENTION_WINDOW_SECONDS` | `30` | Max wait after being named before answering anyway |
| `BOT_BARGE_IN` | `true` | Stop talking when a human speaks over the bot |
| `BOT_REPLY_LANGUAGE` | *(auto)* | Pin replies to one BCP-47 tag |
| `BOT_GREET_ON_JOIN` | `true` | Greet on join and when someone enters mid-session |
| `BOT_REQUIRE_CONSENT` | `false` | Require `/consent accept` per speaker before transcribing |

The two timing overrides are blank by default and derived from `talkativeness`
instead. The mapping is geometric rather than linear, because the midpoint between
a 180 s and an 8 s cooldown is not "moderately chatty" at 94 s:

| Talkativeness | Silence before replying | Cooldown |
|---|---|---|
| 0 | 6.0 s | never volunteers |
| 50 | 3.1 s | 38 s |
| 65 (default) | 2.5 s | 24 s |
| 100 | 1.6 s | 8 s |

### Persona

Characters live in `settings.json` (or the **Characters** tab of the settings
editor): system instruction, Gemini model, ElevenLabs voice id, and
`alternative_names` — the aliases that count as addressing the bot when they show
up in a transcript. Set those to how people will actually say the name out loud.

---

## Privacy

When the bot joins, it posts a notice in the channel saying that voice is
transcribed by Gemini and answered via ElevenLabs.

Per-speaker consent is **opt-in** (`BOT_REQUIRE_CONSENT`, default off) — with it
on, a speaker's audio is ignored until they run `/consent accept`, which is
correct for a server where you need recorded proof of disclosure but reads as "the
bot is broken" on a private server. `/consent revoke` and `/data delete` work
either way. The disclosure text is in `src/discord/consent_commands.py`; keep it
accurate if you change which services the pipeline touches.

Conversation state is not persisted across runs. Session transcripts are written
to `transcripts/` for debugging.

---

## Costs

Roughly, per hour of *active conversation*:

- **Transcription** — Gemini bills audio at 32 tokens/second, so on a Flash-tier
  model this is cents per hour, not dollars.
- **Chat + volunteer gate** — small; the gate is one token of output per check.
- **Speech** — ElevenLabs bills per character, and this is the dominant cost.

Live estimates appear in the TUI; the rate card is under `pricing` in
`settings.json` / the `*_USD_*` variables in `.env`.

---

## Development

```bash
ruff check src tests
mypy src --ignore-missing-imports --strict
pytest
```

### Package map

```
main.py                          # Server entry point: python main.py
src/
├── main.py                      # CLI: headless modes and --tui dispatch
├── session/
│   ├── conversation_runner.py   # The voice session: policy, chat, TTS, playback
│   └── service.py               # Start/stop/supervise sessions for a long-lived process
├── conversation/
│   ├── chat.py                  # Gemini chat, language rules, volunteer gate
│   ├── policy.py                # Talkativeness dial and reply triggers
│   ├── log.py                   # Transcript merge / "since last bot turn"
│   └── persona.py
├── discord/
│   ├── client.py                # Gateway, voice join, receive lifecycle
│   ├── voice_sink.py            # Incoming audio → queue
│   ├── pcm_stream.py            # Streaming PCM audio source (+ barge-in)
│   ├── playback.py              # Playback lifecycle
│   └── consent_commands.py      # /consent, /data
├── transcription/
│   ├── gemini_stt.py            # Gemini live + batch transcription
│   ├── coordinator.py           # Per-user utterance chunking
│   └── preprocessing.py         # 48 kHz stereo → 16 kHz mono
├── tts/elevenlabs.py            # REST/streaming client (no SDK)
├── storage/                     # settings.json, consent, transcripts
└── ui/                          # Textual dashboard and settings editor
```

### Concurrency notes

Discord runs on asyncio, but voice sink callbacks and the audio player run on
their own threads. Forward work to an `asyncio.Queue` via
`loop.call_soon_threadsafe` rather than blocking, and never call Discord APIs
from a sink callback.

### Known trade-off

The `live` backend opens one Live API WebSocket per utterance, because utterance
segmentation happens on the Discord side. In a fast back-and-forth that is a
connection per turn. A persistent per-speaker session would avoid it; the current
shape was kept because it matches the existing coordinator contract. Switch to
`batch` if you hit rate limits.

---

## Voice encryption (DAVE)

Discord has enforced DAVE end-to-end encryption on all voice channels since
2 March 2026, and a bot that does not implement it cannot receive audio at all.
This is why the project is Python: `discord.py` 2.6+ supports DAVE via the `davey`
package, while `@discordjs/voice` still cannot decrypt inbound audio
([discord.js#11419](https://github.com/discordjs/discord.js/issues/11419)).

Keep `discord.py`, `davey`, `PyNaCl` and `discord-ext-voice-recv` current. Note
that `discord.py[voice]` requires `PyNaCl < 1.6`, so do not pin above that or
dependency resolution fails.

---

## License

MIT — see [`LICENSE`](LICENSE).
