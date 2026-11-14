# Messenger connections

Ask the librarian from a chat app.

## Messenger Connections

| Messenger | Status | Start here |
|---|---|---|
| **Telegram** | Supported | [`autorag tui --telegram`](telegram.md) |
| **WhatsApp** | Supported (unofficial client) | [`autorag tui --whatsapp`](whatsapp.md) |
| **Discord** | Supported (no steering yet) | [`autorag tui --discord`](discord.md) |
| Slack, others | Not supported yet | |

Telegram runs through the [`@llblab/pi-telegram`](https://github.com/llblab/pi-telegram) bridge inside a running `autorag tui` session: questions you send the bot get the same routing and two-phase search as the terminal. Setup, commands, and keeping it running are in [docs/telegram.md](telegram.md).

WhatsApp runs through the pinned [`whatsapp-pi`](https://www.npmjs.com/package/whatsapp-pi) extension the same way; see [docs/whatsapp.md](whatsapp.md).

Discord runs through the [`pi-discord-remote`](https://pi.dev/packages/pi-discord-remote) extension and creates one channel per TUI session; the fast answer and the verified final answer are posted as they land. Setup, permissions, and limits are in [docs/discord.md](discord.md).
