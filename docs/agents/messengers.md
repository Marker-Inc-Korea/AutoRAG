---
title: "Overview"
description: "Ask the librarian from Telegram, WhatsApp, or Discord while a TUI session runs on your machine."
---

# Overview

The librarian can be reached from a chat app while an `autorag tui` session runs on your machine.
This page compares the three supported bridges and points at each page's setup.

## Messenger connections

| Messenger | Status | Start here |
| --- | --- | --- |
| Telegram | Supported | [`autorag tui --telegram`](telegram.md) |
| WhatsApp | Supported (unofficial client) | [`autorag tui --whatsapp`](whatsapp.md) |
| Discord | Supported (no steering yet) | [`autorag tui --discord`](discord.md) |
| Slack, others | Not supported | — |

Each bridge is an extension of the running TUI process: a question you send the bot runs through the
same Jev routing and two-phase search as the same text typed in the terminal, and the answer comes
back in the chat. If the TUI exits, the bridge stops answering. A chat message sent while a search
is running steers that search where the bridge supports it (Telegram, WhatsApp); Discord does not
support steering yet.

A bridged session has the same shell and file tools as the terminal, so treat every chat that can
drive it as a credential for a shell on your machine, and allow only chats you control. Chat
content is untrusted input to the model. A bridged `tui` defaults to `full` permission because there
is no terminal to answer an approval dialog; see [Permission modes](permissions.md).

## How the bridges are loaded

| Bridge | Extension | Pinned version |
| --- | --- | --- |
| Telegram | [`@llblab/pi-telegram`](https://github.com/llblab/pi-telegram) | `0.58.3` |
| WhatsApp | [`whatsapp-pi`](https://www.npmjs.com/package/whatsapp-pi) | `1.0.70` |
| Discord | [`pi-discord-remote`](https://www.npmjs.com/package/pi-discord-remote) | `0.2.4` |

The Telegram bridge loads on demand from the installed package. The WhatsApp extension is installed
into Pi's agent directory by `autorag tui --whatsapp` on first run, then loaded. The Discord
extension ships with AutoRAG and is hosted in-process; it creates one text channel per session and
the fast answer and the verified final answer are posted as they land.
