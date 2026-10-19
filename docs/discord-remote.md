# Discord remote control

`autorag tui --discord` hosts the community [`pi-discord-remote`](https://pi.dev/packages/pi-discord-remote)
extension (MIT, the most downloaded Discord remote-control package for Pi) inside the
AutoRAG TUI session. On launch it creates one text channel for the session. Allow-listed
Discord users type there; AutoRAG answers in the channel. Shutting the TUI down deletes
the channel.

## What Discord can do

| From Discord | Effect |
|---|---|
| Message in the session channel | Runs as a question through AutoRAG's two-phase search. While a search runs the extension replies "Still processing"; it does not steer |
| Reply to an agent question | Answers `discord_ask_user_question` |

The reply is posted once, when the whole search settles, so it carries the verified final
answer rather than the first phase's. Discord cannot start a session from nothing: a local
`autorag tui --discord` process must be running. Image upload (`discord_send_image`) is
registered by the extension but deliberately **not** enabled: it uploads a local file path
chosen by the model.

## Setup

1. Create a bot in the Discord developer portal, enable **Message Content**, and invite it
   with Read/Send Messages, Add Reactions and Manage Channels.
2. Write `~/.pi/agent/pi-discord-remote/config.json` (or run `/pi-discord-remote setup` in the TUI):

   ```json
   { "token": "…", "guildId": "…", "allowedUserIds": ["<your user id>"] }
   ```

3. `autorag tui --discord`

`allowedUserIds` is the extension's own allow-list (empty means every guild member can
send messages). AutoRAG adds no check on top of it. `DISCORD_TOKEN` / `DISCORD_GUILD_ID`
can replace token and guild.

## Tests

- `test/agent/discord-remote.test.ts` runs the real extension and real discord.js against
  `test/helpers/fake-discord.ts` (local REST + gateway) and a faux model.
- `bun scripts/manual-qa/discord-remote-qa.ts` runs the real `autorag tui --discord`
  process on a pty against the same fakes with an isolated HOME and no inherited credentials.

Neither talks to the real Discord service. A run against a real bot is the one thing they
cannot prove.
