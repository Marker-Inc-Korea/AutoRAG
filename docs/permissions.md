# Permission modes

The AutoRAG agent can organize your files, so changes to **existing files** go through an approval gate.
The gate is a pi extension (`src/agent/permission-gate.ts`) that sits on every `bash`, `write`, `edit`, and saved
automation call. By default (`auto`) Jev reads each call and only what could change an existing file reaches you;
reading, searching, copying, and creating new files never ask.

## Modes

| Mode | Behavior |
| --- | --- |
| `auto` (default) | Jev (the judgment model) classifies each call first. A call that changes no existing file runs. A flagged one is approved silently when Jev is at least 80% sure the user's request covers exactly that change. Otherwise, or when Jev is unreachable, the question goes to you, with Jev's reason shown. |
| `ask` | You approve every `bash`, `write`, `edit`, and saved-automation call, including harmless ones like `ls`. Jev is never consulted, so it works with no Jev backend. Reads, searches, and AutoRAG's own tools never ask. "Allow all such calls this session" ends the questions until the mode changes. |
| `full` | No gate. The agent does everything itself. |

Not asked about in `auto`, because Jev finds they do not delete, move, rename, overwrite, or edit an existing file: `ls`, `cat`, `grep`,
`cp` to a new path, `mkdir`, `touch`, `write` to a new path. Files and folders the agent created itself stay free
to edit, move, and delete, so the intended workflow is copy first, then change the copy. The system prompt tells
the model this in `ask` and `auto` (in `ask` it also tells the model every such call needs your approval, so it
batches steps).

## How a call is classified

This section is about `auto`; `ask` skips classification and asks about every call. There is no command parser. Before every `bash`, `write`, `edit`, and `manage_automations save` call, Jev answers six
questions in one batched request, each a probability that the call does that to a file that existed before it ran:

| Category | Question |
| --- | --- |
| `delete` | removes a file or folder, by any means (`rm`, `trash`, `find -delete`, a one-line perl/python/ruby/node call, ...) |
| `move` | moves one to a different directory |
| `rename` | renames one within its directory |
| `overwrite` | replaces a file's whole content (redirect, download, `cp` over it, `git reset --hard`) |
| `edit` | changes part of a file in place (`sed -i`, an editor, a script that patches it) |
| `unclear` | runs code whose effect cannot be told (an unseen script, `make`, an encoded command) |

A category at 0.2 or above, or one Jev is unsure about, counts as flagged and the call is asked about; the dialog
names the categories. Jev is shown the call, what you said in this conversation, the agent's earlier successful
calls, and for each path the call names whether it exists right now and whether the agent created it. That is
how `cp report.md report.edited.md` then `sed -i.bak ... report.edited.md` runs without asking: the copy is the
agent's own.

Because Jev reads what a call does rather than matching its text, a one-line `python3 -c` that trashes a file, or
`f=rm; $f x`, is caught, and a script Jev cannot read is `unclear` and asked about. If Jev cannot be reached,
times out after 15 seconds, or returns no usable answer, the call is treated as unclassified and goes to you in
`auto` (nothing is let through on a failure). `ask` and `full` never call Jev.

This is a gate against mistakes and runaway agents, not a sandbox: Jev is a model, so it can misjudge, and a
determined script can still reach files in ways it cannot see. Use `full` only when you trust the task.

External MCP servers' tools (`mcp__*`) are outside the gate: AutoRAG cannot know what they do.

## Switching modes

| Where | How |
| --- | --- |
| TUI | **Shift+Tab** cycles `ask` → `auto` → `full` (the session starts in `auto`). `/permissions` shows a picker, `/permissions auto` sets one. The footer shows `permissions: <mode>`. |
| CLI | `autorag tui --permission-mode auto`, `autorag search "..." --permission-mode full` |
| Environment | `AUTORAG_PERMISSION_MODE=full` |
| Config | `"permissionMode": "auto"` in `~/.autorag/config.json` |

Precedence: flag, then environment, then config file, then `auto`. An unknown value is an error, never a silent
fallback. Shift+Tab replaces pi's thinking-level cycle key inside `autorag tui`.

Shift+Tab and `/permissions` apply to the very next tool call and clear any "allow for this session" grants.

## The approval dialog

```
Approve this change to your files?
  delete  /Users/me/Downloads/old.pdf
    $ rm /Users/me/Downloads/old.pdf
> Allow once
  Allow all delete operations this session
  Deny
  Deny and tell the model what to do instead
```

Deny blocks the call and tells the model not to retry and to work on a copy. "Deny and tell the model..." lets you
add a note that is passed on. Questions are asked one at a time even when the model issues parallel calls.

## Without a terminal

`autorag search`, automations, and the Discord/Telegram/WhatsApp bridges have nobody to answer a dialog:

- `auto`: Jev's approval lets it through; anything else is blocked, and the model is told to use a copy.
- `ask`: every `bash`, `write`, `edit`, and saved-automation call is blocked, so nothing that touches files runs.
- `full`: nothing is blocked.

Pass `--permission-mode full` (or set the env/config) for an unattended job that must delete or move files.

## Jev in `auto`

In `auto`, the same batch also carries one allow/deny question. The state Jev judges, for both, is:

- **Everything you said in this conversation**, numbered, oldest first, exactly as typed: the first question,
  every follow-up (also ones queued with Alt+Enter), and every steer typed while the agent was working (marked
  `(correction while the agent was working)`). AutoRAG's own internal prompts (the fast-answer, verification,
  and direct-answer wrappers, steer continuations) are never included: the log is fed where your words arrive,
  not read back from the transcript.
- The working directory, the call itself, the agent's earlier successful calls, and the existence facts above.

It is saved in the session next to the transcript, so `/resume` brings the earlier messages back and `/new`
starts empty.

Nothing is shortened while it fits. Jev's context allows 30,000 tokens of state, so the conversation gets
about 108,000 characters minus the operations. Past that, the opening request and the newest messages are kept
and the middle is dropped with a count (`[12 earlier messages omitted]`); a single message longer than the
whole budget is cut in its middle. A run of 1,000+ characters without whitespace (a pasted blob) is replaced by
a note of its size, because it says nothing about intent and jev-use's secret redaction is quadratic on it.

Never file contents. `jev-use` redacts secrets such as API keys and URL passwords before anything leaves the
machine. Jev needs a backend (`jev` config, `OPENROUTER_API_KEY`, `TYPESAFE_API_KEY`, or a Dazzi login). Without one, every
bash/write/edit call is unclassified, so `auto` asks about each one like `ask` does (`full` is the mode for running
without approvals).

## MCP

`autorag-mcp` (Lite) is always full permission: it exposes retrieval, report, and refresh tools only, and has no
tool that deletes, moves, or edits your files, so the gate does not apply to it.
