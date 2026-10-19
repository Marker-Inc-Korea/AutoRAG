# Discord remote control

`autorag tui --discord` lets you ask the librarian questions from a Discord channel while a TUI session is running on your machine. It loads [`pi-discord-remote`](https://pi.dev/packages/pi-discord-remote), the most-downloaded Pi extension for Discord (MIT).

On launch it creates one text channel for the session. A message you send there runs through the same Jev routing and two-phase search as the same text typed in the terminal. When the TUI exits, the channel is deleted.

## Status

| Messenger | Status |
|---|---|
| Discord | Supported, no steering yet |
| Telegram | Supported, see [telegram.md](telegram.md) |

## Requirements

- A running `autorag tui --discord` session. The bridge is an extension of that process: if the TUI exits, the bot stops answering. Discord cannot start a session.
- A Discord bot token and a server where the bot may manage channels.
- Outbound HTTPS and WebSocket only. No inbound port is opened.

## Setup

1. Create an application at the [Discord developer portal](https://discord.com/developers/applications), open **Bot**, and copy the token.
2. Under **Bot**, enable **Message Content Intent**. Without it login fails with `disallowed intents`.
3. Invite the bot to your server (**OAuth2 → URL Generator**, scope `bot`) with these permissions: View Channels, Send Messages, Add Reactions, **Manage Channels**.
4. Turn on Developer Mode (Discord settings → Advanced), then copy your **server ID** and your **user ID** from the right-click menu.
5. Write `~/.pi/agent/pi-discord-remote/config.json` and restrict it:

   ```json
   {
     "token": "<bot token>",
     "guildId": "<server id>",
     "allowedUserIds": ["<your user id>"]
   }
   ```

   ```bash
   chmod 600 ~/.pi/agent/pi-discord-remote/config.json
   ```

   Or run `/pi-discord-remote setup` inside the TUI. `DISCORD_TOKEN` and `DISCORD_GUILD_ID` can stand in for the token and server ID.
6. Start the TUI:

   ```bash
   autorag tui --discord
   ```

The status line shows `🔌 Discord: #<channel>`, and the channel appears in your server.

> **Set `allowedUserIds`.** The session has shell access. AutoRAG adds no allow-list of its own, so an empty list lets every member of the server drive it.

## Using it

- Send a question in the session channel. The bot reacts with ⏳ while it works.
- If the search has two phases, the fast answer is posted as soon as verification starts. The verified final answer follows with a mention of the asker. A one-phase search posts one reply.
- Tool calls are posted as short labels while the search runs.
- When the model needs to ask you something, it asks in the channel and reads your next message as the answer.
- A message sent while a search is running gets `Still processing`. Steering a running search from Discord is not supported yet.
- `/pi-discord-remote status | stop | start` work in the TUI.

`discord_send_image` is registered by the extension but kept inactive: it uploads a local file the model chooses.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Discord remote: no config` | Missing config file, or invalid JSON |
| `disallowed intents` | Message Content Intent is off |
| `Could not create channel` | The bot lacks Manage Channels |
| Channel appears but nothing answers | The bot cannot read the channel, or your user ID is not in `allowedUserIds` |
| `Could not post the fast answer` | TUI warning only; the final answer still posts |

## Tests

`test/agent/discord-remote.test.ts` runs the real extension and discord.js against a local fake Discord (`test/helpers/fake-discord.ts`). `bun scripts/manual-qa/discord-remote-qa.ts` runs the real `autorag tui --discord` on a pty against the same fake with an isolated home. Neither talks to the real Discord service.
