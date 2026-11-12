# Sending messages (`send_message`)

The agent can send a message out of your machine when you ask it to: "tell the ops channel the deploy is done",
"email Minji the summary". The tool is `send_message`. It exists only when `config.json` has at least one channel.

## Configuring channels

```json
{
  "messaging": {
    "channels": {
      "ops": { "kind": "telegram", "tokenEnv": "TELEGRAM_BOT_TOKEN", "chatId": "-1001234567890" }
    }
  }
}
```

- A channel id is 1-40 letters, digits, `_` or `-`. The model passes it back verbatim.
- `kind` selects the transport; the other fields belong to that kind. Unknown fields are rejected.
- **Secrets never go in the config.** Fields named `...Env` hold the *name* of an environment variable. A value
  that is not an upper-case variable name (`A-Z`, `0-9`, `_`) is refused, because a pasted key is mixed case; the
  error never echoes it.
- Secrets are read when a message is sent, not at startup.

Channel kinds each document their own fields in their own section below.

## `telegram`: a bot posting to one chat

```json
{ "kind": "telegram", "tokenEnv": "TELEGRAM_BOT_TOKEN", "chatId": "-1001234567890" }
```

| Field | |
|---|---|
| `tokenEnv` | Name of the variable holding the bot token from @BotFather. |
| `chatId` | Numeric chat id (groups and channels are negative) or `@channelname`. The bot must already be in the chat; a user must have started the bot before it can message them. |
| `threadId` | Optional forum topic to post into. |

The chat is fixed by the config, so the model cannot redirect a message. Text is sent as plain text (no Markdown or
HTML parsing). A subject, if given, is put on a first line. Over 4096 characters the text is split into ordered
parts; if a part fails, the error says how many went out so nothing is sent twice. Success needs Telegram to return
a message id.

## Approval

Every send is shown to you before it goes, one message at a time, with the channel, recipients, subject, and the
full text. There is no "allow all": a grant you gave for file changes does not cover sending either.

| Mode | A send |
|---|---|
| `ask`, `auto` | You approve each one. Jev is not asked. Without a UI (nobody to answer) it is blocked. |
| `full` | Goes without asking. Choose it yourself with `--permission-mode full`, `AUTORAG_PERMISSION_MODE=full`, or `permissionMode` in the file. |

`autorag search` and a bridged TUI default to `full` for file changes. That default is **not** consent to send
unattended, so there `send_message` is not offered unless you chose `full` explicitly.

A denied send is not retried within the request. A failed send is reported, not repeated: a send cannot be undone.

## Out of scope

Automation agents (`manage_automations`) and the MCP server never get `send_message`.
