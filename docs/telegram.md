# Telegram remote control

`autorag tui --telegram` lets you ask the librarian questions from a Telegram chat while a TUI session is running on your machine. It loads [`@llblab/pi-telegram`](https://github.com/llblab/pi-telegram), the most-installed Pi extension for this (about 13.8k npm downloads a month; the leading Discord extension has about 200).

A question you send the bot runs through the same Jev routing and two-phase search as the same text typed in the terminal, and the answer comes back in the chat. A message sent while a search is running steers that search, exactly like pressing Enter in the TUI.

## Status

| Messenger | Status |
|---|---|
| Telegram | Supported |
| Discord, Slack, others | Not supported |

## Requirements

- A running `autorag tui --telegram` session. The bridge is an extension of that process: if the TUI exits, Telegram stops answering.
- A Telegram bot token from [@BotFather](https://t.me/BotFather) (`/newbot`).
- Outbound HTTPS only. No inbound port is opened.

## Setup

1. Create a bot with @BotFather and copy the token.
2. Find your numeric Telegram user id. Send your bot any message, then run:

   ```bash
   curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates" | jq '.result[].message.from | {id, username}'
   ```

   Do this before connecting; a connected bridge consumes the same updates.
3. Write `~/.pi/agent/telegram.json` (keep any existing content), referencing the token by environment variable:

   ```json
   {"profiles": {"default": {"botToken": "$TELEGRAM_BOT_TOKEN", "allowedUserId": 123456789}}}
   ```

4. Start the TUI with the token exported in the same shell:

   ```bash
   export TELEGRAM_BOT_TOKEN='123456:ABC...'
   autorag tui --telegram
   ```

5. In the TUI, run `/telegram-connect`. You should see `Telegram bridge connected.`
6. Message the bot from your own account.

Alternatively, run `/telegram-setup` in the TUI and paste the token. Without `allowedUserId`, the first account to send `/start` becomes the only allowed owner, so set `allowedUserId` up front.

## Commands

In the TUI:

| Command | Purpose |
|---|---|
| `/telegram-setup` | Save the bot token |
| `/telegram-connect` | Start polling and bind this session |
| `/telegram-status [--debug]` | Show connection state and recent events |
| `/telegram-disconnect` | Stop polling and release ownership |

In the bot chat: `/start` (menu and pairing), `/new`, `/compact`, `/abort`, `/stop`, `/queue`, `/model`, `/settings`.

## Behavior

- Telegram sessions have the same tools as the TUI, including `bash`, `edit`, and `write`. Treat the bot token and `allowedUserId` as credentials for a shell on your machine.
- The bot token and pairing live in `~/.pi/agent/telegram.json` (under `PI_CODING_AGENT_DIR` if set), never in AutoRAG config.
- The prompt queue is per session. Anything waiting when the process exits is lost and not replayed.
- `print` and `json` modes never poll Telegram.

## Keeping it running

The bridge does not start a background process of its own. To keep the session alive after closing the terminal, run the TUI under tmux:

```bash
tmux new -s autorag
autorag tui --telegram      # then /telegram-connect
# detach: Ctrl+b d    reattach: tmux attach -t autorag
```

The session stops if the machine sleeps or reboots.

## Troubleshooting

- Footer shows `disconnected`: the session is configured but not polling. Run `/telegram-connect`.
- `not configured`: the token is missing or `$TELEGRAM_BOT_TOKEN` is not set in the shell that started the TUI.
- `Telegram connection failed`: run `/telegram-status --debug` and read `recent runtime events`.
- Verify the token: `curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe" | jq '{ok, username: .result.username}'`.

## Testing

- `test/agent/telegram-remote.test.ts`: offline end-to-end against a local fake Bot API.
- `test/live-e2e/telegram-remote.test.ts`: real model through the fake chat; run with `AUTORAG_TELEGRAM_LIVE=1` and `OPENROUTER_API_KEY`.
