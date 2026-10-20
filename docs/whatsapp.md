# WhatsApp remote control

`autorag tui --whatsapp` lets you ask the librarian questions from WhatsApp while a TUI session is running on your machine. It installs and loads [`whatsapp-pi`](https://www.npmjs.com/package/whatsapp-pi) (pinned version), a Pi extension that drives WhatsApp Web through Baileys.

A message from an allowed chat runs through the same routing and two-phase search as the same text typed in the terminal, and the answer is sent back to that chat. A message sent while a search is running is handled like input typed during a search in the TUI.

## Requirements

- A running `autorag tui --whatsapp` session. The extension lives in that process: if the TUI exits, WhatsApp stops answering.
- A phone with WhatsApp to link as a device.
- Network access to npm (first run installs the package into Pi's agent dir) and to WhatsApp.

## Setup

1. Start the session:

   ```bash
   autorag tui --whatsapp
   ```

   To answer only one group, add `--whatsapp-group <group-jid>` (for example `120363012345@g.us`).
2. In the TUI run `/whatsapp`, choose **Connect WhatsApp**, and scan the QR code in WhatsApp under *Settings > Linked devices > Link a device*. This is needed once; the login is kept for later runs.
3. In the same `/whatsapp` menu, add your own number (or the group) to **Allowed Contacts** / **Allowed Groups**. Messages from anyone else are ignored.
4. Send a question from the allowed chat. The answer comes back in the same chat.

## Notes

- Anything an allowed chat sends drives a session that still has shell and file-write tools. Allow only chats you control.
- Chat content is untrusted input to the model.
- `whatsapp-pi` uses an unofficial WhatsApp Web client; WhatsApp can restrict accounts that use one.
- The model's progress notes, fast answer, and final answer may arrive as separate WhatsApp messages, and answers to questions typed in the terminal can also go to the last WhatsApp chat. Both are behaviors of the extension, not of AutoRAG.
