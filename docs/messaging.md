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

## `agentmail`: an email address of the agent's own

The agent gets its own inbox at [AgentMail](https://www.agentmail.to) and sends from it. It never opens your
mailbox and no SMTP password is stored; people see mail from `my-agent@agentmail.to`, not from you.

```json
{ "kind": "agentmail", "apiKeyEnv": "AGENTMAIL_API_KEY", "inboxId": "my-agent@agentmail.to" }
```

| Field | |
|---|---|
| `apiKeyEnv` | Name of the variable holding the AgentMail API key (`am_...`). |
| `inboxId` | The inbox's address, as AgentMail returns it for `inbox_id`. |
| `region` | `us` (default) or `eu`. |

Set up once (AgentMail's own flow, not AutoRAG's): create an account at console.agentmail.to or let the agent
sign itself up with your email (`POST /v0/agent/sign-up`), verify the one-time code it mails you, and export the key.
Until verified, an inbox can only mail the human attached to it, and new accounts ramp up how many distinct
recipients they may reach.

Each send takes `to` (1-10 addresses) and a one-line `subject`. A reply to the message arrives in the inbox, not
in AutoRAG: this channel only sends. Success needs AgentMail to return a message id; an answer without one is
reported as unconfirmed.

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
