# Permission modes

The AutoRAG agent can organize your files, so changes to **existing files** go through an approval gate.
Reading, searching, copying, and creating new files never ask. The gate is a pi extension
(`src/agent/permission-gate.ts`) that sits on every `bash`, `write`, and `edit` call.

## Modes

| Mode | Behavior |
| --- | --- |
| `ask` (default) | Asks before deleting, moving, renaming, overwriting, or editing an original. |
| `auto` | Jev (the judgment model) answers first. It approves silently when it is at least 80% sure the user's request covers exactly that change. Otherwise, or when Jev is unreachable, the question goes to you, with Jev's reason shown. |
| `full` | No gate. The agent does everything itself. |

Not gated, in every mode: `ls`, `cat`, `grep`, `find`, `cp` to a new path, `mkdir`, `touch`, `write` to a new
path, `sed` without `-i`, and every AutoRAG search tool. Files and folders the agent created itself stay free
to edit, move, and delete, so the intended workflow is copy first, then change the copy. The system prompt
tells the model this in `ask` and `auto`.

What counts as a change to an original:

- `delete`: `rm`, `rmdir`, `unlink`, `trash`, `shred`, `find -delete` / `-exec rm`, `rsync --remove-source-files` / `--delete`, `git rm`, `git clean`
- `rename` / `move`: `mv`, `git mv` (same folder and name change is a rename, anything else a move)
- `overwrite`: `>`, `>>`, `tee`, `dd of=`, `cp` onto an existing file, `write` onto an existing file
- `edit`: `edit` on an existing file, `sed -i`, `perl -i`, `truncate`
- `script`: `python -c` / `node -e` / `perl -e` / shell heredocs whose source removes or renames files
- A saved `manage_automations` rule whose command does any of the above

A target only known at run time (`rm $X`, `rm *.tmp`, `xargs rm`) is gated as well. Wrappers (`sudo`, `env`,
`nohup`, `bash -c`, `eval`, `$(...)`) are looked through. This is a gate against mistakes and runaway agents, not
a sandbox: a determined script can still reach files in ways no command parser sees. Use `full` only when you
trust the task.

External MCP servers' tools (`mcp__*`) are outside the gate: AutoRAG cannot know what they do.

## Switching modes

| Where | How |
| --- | --- |
| TUI | **Shift+Tab** cycles `ask` → `auto` → `full`. `/permissions` shows a picker, `/permissions auto` sets one. The footer shows `permissions: <mode>`. |
| CLI | `autorag tui --permission-mode auto`, `autorag search "..." --permission-mode full` |
| Environment | `AUTORAG_PERMISSION_MODE=full` |
| Config | `"permissionMode": "ask"` in `~/.autorag/config.json` |

Precedence: flag, then environment, then config file, then `ask`. An unknown value is an error, never a silent
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

- `ask`: gated changes are blocked, and the model is told to use a copy.
- `auto`: Jev's approval lets it through; anything else is blocked.
- `full`: nothing is blocked.

Pass `--permission-mode full` (or set the env/config) for an unattended job that must delete or move files.

## Jev in `auto`

Jev is asked one allow/deny question per gated operation. The state it judges is:

- **Everything you said in this conversation**, numbered, oldest first, exactly as typed: the first question,
  every follow-up (also ones queued with Alt+Enter), and every steer typed while the agent was working (marked
  `(correction while the agent was working)`). AutoRAG's own internal prompts (the fast-answer, verification,
  and direct-answer wrappers, steer continuations) are never included: the log is fed where your words arrive,
  not read back from the transcript.
- The working directory and one line per operation (`delete /path`).

It is saved in the session next to the transcript, so `/resume` brings the earlier messages back and `/new`
starts empty.

Nothing is shortened while it fits. Jev's context allows 30,000 tokens of state, so the conversation gets
about 108,000 characters minus the operations. Past that, the opening request and the newest messages are kept
and the middle is dropped with a count (`[12 earlier messages omitted]`); a single message longer than the
whole budget is cut in its middle. A run of 1,000+ characters without whitespace (a pasted blob) is replaced by
a note of its size, because it says nothing about intent and jev-use's secret redaction is quadratic on it.

Never file contents. `jev-use` redacts secrets such as API keys and URL passwords before anything leaves the
machine. Jev needs a backend (`jev` config, `OPENROUTER_API_KEY`, `TYPESAFE_API_KEY`, or a Dazzi login); without
one `auto` behaves like `ask`. A 15 second timeout falls back to asking you.

## MCP

`autorag-mcp` (Lite) is always full permission: it exposes retrieval, report, and refresh tools only, and has no
tool that deletes, moves, or edits your files, so the gate does not apply to it.
