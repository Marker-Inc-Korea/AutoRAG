---
title: "WhatsApp"
description: "Control a running autorag tui session from WhatsApp using the pinned whatsapp-pi extension."
---

# WhatsApp

`autorag tui --whatsapp` lets you ask the librarian questions from WhatsApp while a TUI session is
running on your machine. It installs (first run) and loads
[`whatsapp-pi`](https://www.npmjs.com/package/whatsapp-pi) (pinned `1.0.70`), a Pi extension that
drives WhatsApp Web through Baileys.

A message from an allowed chat runs through the same routing and two-phase search as the same text
typed in the terminal, and the answer is sent back to that chat. A message sent while a search is
running is handled like input typed during a search in the TUI.

## Requirements

- A running `autorag tui --whatsapp` session. The extension lives in that process: if the TUI exits,
  WhatsApp stops answering.
- A phone with WhatsApp to link as a device.
- Network access to npm (first run installs the package into Pi's agent dir) and to WhatsApp.

## Setup

1. Start the session:

   ```bash
   autorag tui --whatsapp
   ```

   To answer only one group, add `--whatsapp-group <group-jid>` (for example
   `120363012345@g.us`). `--whatsapp-group` requires `--whatsapp`.

2. In the TUI run `/whatsapp`, choose **Connect / Reconnect WhatsApp**, and scan the QR code in
   WhatsApp under *Settings > Linked devices > Link a device*. This is needed once; the login is
   kept for later runs.

   If your terminal cannot render the QR code, set `WHATSAPP_PI_QR_TEXT=1` to use the text QR.

3. In the same `/whatsapp` menu, add your own number (or the group) to **Allowed Contacts** /
   **Allowed Groups**. Messages from anyone else are ignored.

4. Send a question from the allowed chat. The answer comes back in the same chat.

The `/whatsapp` menu also manages **Disconnect** and **Logoff (Delete Session)**, browses
**Recents** and message history, and sends a one-off message without invoking the agent.

## Notes

- Anything an allowed chat sends drives a session that still has shell and file-write tools. Allow
  only chats you control, and treat chat content as untrusted input to the model.
- All allowed chats share the **same model context**; reply routing is not conversation privacy
  isolation.
- A bridged `tui` defaults to `full` permission because there is no terminal dialog; an explicit
  `--permission-mode`, `AUTORAG_PERMISSION_MODE`, or config value still wins.
- WhatsApp chat commands `/compact` and `/abort` compact the current Pi context and abort the running
  operation.
- `whatsapp-pi` uses an unofficial WhatsApp Web client; WhatsApp can restrict accounts that use one.
- The model's progress notes, fast answer, and final answer may arrive as separate WhatsApp messages,
  and answers to questions typed in the terminal can also go to the last WhatsApp chat. Both are
  behaviors of the extension, not of AutoRAG.
- A successful socket send means **accepted**, not delivered or read. An undefined result after
  transmission starts is reported as `delivery: "unknown"`; do not resend automatically, since the
  first send may still arrive.

## Local data

The extension stores WhatsApp credentials, configuration, recents, logs, archive, and media under:

```text
~/.pi/agent/extensions/whatsapp-pi/
```

Keep this directory private: it includes authentication credentials and conversation data.

## Keeping it running

The extension does not start a background process of its own. Keep the TUI alive with tmux the same
way as [Telegram](telegram.md#keeping-it-running). On Ubuntu the extension also ships an optional
systemd user service installer (`scripts/install-whatsapp-pi-service.sh`) that runs it under tmux at
login; it requires `tmux`, `systemd`, and an installed `pi` command.
