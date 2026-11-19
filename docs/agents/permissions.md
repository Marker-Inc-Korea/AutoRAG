---
title: "Permission modes"
description: "How the approval gate classifies file-changing calls, the ask/auto/full modes, and how to switch them."
---

# Permission modes

The librarian can organize your files, so changes to **existing files** go through an approval
gate. The gate is a Pi extension (`src/agent/permission-gate.ts`) that sits on every `bash`,
`write`, `edit`, and saved-automation call. Reading, searching, copying, and creating new files
never ask.

## Modes

| Mode | Behavior |
| --- | --- |
| `auto` (default) | Jev classifies each gated call first. A call that changes no existing file runs. A flagged one is approved silently when Jev is at least 80% sure the user's request covers exactly that change. Otherwise — or when Jev is unreachable — the question goes to you, with Jev's reason shown. |
| `ask` | You approve every `bash`, `write`, `edit`, and saved-automation call, including harmless ones like `ls`. Jev is never consulted, so it works with no Jev backend. "Allow all such calls this session" ends the questions until the mode changes. |
| `full` | No gate. The agent does everything itself. |

In `auto`, calls that Jev finds do not delete, move, rename, overwrite, or edit an existing file
run without asking — for example `ls`, `cat`, `grep`, `cp` to a new path, `mkdir`, `touch`, and
`write` to a new path. Files and folders the agent created itself stay free to edit, move, and
delete, so the intended workflow is copy first, then change the copy.

## Switching modes

| Where | How |
| --- | --- |
| TUI | **Shift+Tab** cycles `ask` → `auto` → `full`. `/permissions` opens a picker; `/permissions auto` sets one. The footer shows `permissions: <mode>`. |
| CLI | `autorag tui --permission-mode auto`, `autorag search "…" --permission-mode full` |
| Environment | `AUTORAG_PERMISSION_MODE=full` |
| Config | `"permissionMode": "auto"` in `~/.autorag/config.json` |

Precedence is flag, then environment, then config file, then the command's own default. An unknown
value is an error, never a silent fallback to a weaker gate.

The default is `auto` for the interactive `autorag tui`, and `full` for `autorag search` and for a
`tui` started with `--telegram`, `--discord`, or `--whatsapp` (nobody can answer a terminal dialog
there). Shift+Tab replaces Pi's thinking-level cycle key inside `autorag tui`.

Switching mode clears any "allow for this session" grants: a grant only covers the mode it was
given in.

## How a call is classified

This section is about `auto`; `ask` skips classification and asks about every call. There is no
command parser. Before every gated call, Jev answers six category questions in one batched request,
each a probability that the call does that to a file that existed before it ran:

| Category | Question |
| --- | --- |
| `delete` | Removes a file or folder, by any means (`rm`, `trash`, `find -delete`, a one-line perl/python/ruby/node call, …). |
| `move` | Moves one to a different directory. |
| `rename` | Renames one within its directory. |
| `overwrite` | Replaces a file's whole content (redirect, download, `cp` over it, `git reset --hard`). |
| `edit` | Changes part of a file in place (`sed -i`, an editor, a script that patches it). |
| `unclear` | Runs code whose effect cannot be told (an unseen script, `make`, an encoded command). |

A category at probability 0.2 or above, or one Jev escalates, counts as flagged and the call is
asked about; the dialog names the categories. Jev is shown the call, what you said in this
conversation, the agent's earlier successful calls, and for each path the call names whether it
exists right now and whether the agent created it.

Because Jev reads what a call does rather than matching its text, a one-line `python3 -c` that
trashes a file is caught, and a script Jev cannot read is `unclear` and asked about. If Jev cannot be
reached, times out after 15 seconds, or returns no usable answer, the call is treated as
unclassified and goes to you in `auto` — nothing is let through on a failure. `ask` and `full` never
call Jev.

Jev does not see file contents, only whether a named path exists. Before anything leaves the
machine, secret-like strings (API keys, URL passwords) are redacted.

::: warning **Warning:** The gate is against mistakes and runaway agents, not a sandbox.
Jev is a model, so it can misjudge, and a determined script can still reach files in ways it cannot
see. Use `full` only when you trust the task. External MCP-server tools (`mcp__*`) are outside the
gate: AutoRAG cannot know what they do.
:::

## The approval dialog

```text
Approve this change to your files?
  delete  ~/Downloads/old.pdf
    $ rm ~/Downloads/old.pdf
> Allow once
  Allow all delete operations this session
  Deny
  Deny and tell the model what to do instead
```

The session option names the flagged categories (in `ask`, one "Allow all such calls this session").
Deny blocks the call and tells the model not to retry and to work on a copy. "Deny and tell the
model…" lets you add a note that is passed on. Questions are asked one at a time even when the model
issues parallel calls.

The agent's own configuration file is always writable so its self-configuration turn can edit it
without an approval.

## Without a terminal

`autorag search` (and any automation that runs it) and a `tui` with a Discord/Telegram/WhatsApp
bridge default to `full`, since nobody can answer a dialog there. If you set a mode yourself, it
behaves like this when no terminal is attached:

- `auto`: Jev's approval lets it through; anything else is blocked, and the model is told to work on
  a copy.
- `ask`: every `bash`, `write`, `edit`, and saved-automation call is blocked, so nothing that touches
  files runs.
- `full`: nothing is blocked.

To gate an unattended job anyway, pass `--permission-mode auto` (Jev decides; anything it will not
approve is blocked) or set the environment variable / config.

## Jev in `auto`

In `auto`, the same batch also carries one allow/deny question. The state Jev judges is:

- **Everything you said in this conversation**, numbered, oldest first, exactly as typed: the first
  question, every follow-up (including ones queued with Alt+Enter), and every steer typed while the
  agent was working. AutoRAG's own internal phase prompts (the fast-answer, verification, and
  direct-answer wrappers, steer continuations) are never included.
- The working directory, the call itself, the agent's earlier successful gated calls, and the path
  existence facts above.

That state is saved in the session next to the transcript, so `/resume` brings the earlier messages
back and `/new` starts empty. Jev's context allows 30,000 tokens (about 108,000 characters) of
state; past that, the opening request and the newest messages are kept and the middle is dropped with
a count. A run of 1,000+ characters without whitespace (a pasted blob) is replaced by a note of its
size before redaction, because it says nothing about intent.

Jev needs a backend: the `jev` config section, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`,
`TYPESAFE_API_KEY`, `AI_GATEWAY_API_KEY`, `CLOUDFLARE_API_KEY`, or `OPENCODE_API_KEY`, or signing in
to Dazzi Models with `/login`. Without one, every `bash`/`write`/`edit` call is unclassified, so
`auto` asks about each one like `ask` does — `full` is the mode for running without approvals.

## MCP interplay

`autorag-mcp` (Lite) exposes retrieval, report, refresh, and evidence tools only; it has no tool
that deletes, moves, or edits your files, and it does not run the agent loop, so the file gate does
not apply to it. See [MCP server (Lite)](mcp-server.md#tool-surface).
