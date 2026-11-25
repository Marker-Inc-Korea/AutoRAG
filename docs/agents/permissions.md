---
title: "Permission modes"
description: "Binary readonly and full permission modes with OS-level read-only filesystem sandboxing for bash and TUI controls."
---

# Permission modes

AutoRAG uses a two-mode permission model — **`readonly`** (the default) and **`full`** — with no
decision model (Jev) and no per-call approval popups. Enforcement happens in the Pi extension
`src/agent/permission-gate.ts` backed by OS-level read-only filesystem sandboxing in
`src/agent/readonly-sandbox.ts`.

## Modes

| Mode | Behavior |
| --- | --- |
| `readonly` (default) | The agent can read, search, and inspect freely, but cannot modify or delete files on disk. Every `bash` command (including TUI `!` escapes, and Windows `powershell`) runs inside an OS-level read-only filesystem sandbox that blocks all disk writes outside a dedicated session temp directory (`$TMPDIR`/`%TEMP%`). Write-capable tools (`write`, `edit`, `convert_files`, and mutating actions of `manage_automations` and `manage_ltr`) are blocked. |
| `full` | Unrestricted read and write access. Shell commands run directly without the read-only sandbox, and all file-modifying tools are enabled. |

## OS-level read-only enforcement

In `readonly` mode, AutoRAG does not rely on command-name parsing or an LLM classifier to guess
whether a shell command is safe. Instead, the operating system denies filesystem mutations at the
kernel boundary for the shell and every child process it spawns (`rm`, `mv`, `sed -i`, shell
redirects, `python3`, `perl`, `node`, `/usr/bin/trash`, `cmd /c del`, compiled binaries, or
symlink/hardlink escapes):

- **macOS (`darwin`)**: `/usr/bin/sandbox-exec` (Seatbelt) with `(deny file-write*)` and explicit
  write access only for the session's fresh scratch directory (`$TMPDIR`) and virtual devices
  (`/dev/null`, `/dev/zero`, `/dev/tty`, `/dev/fd/*`, `/dev/ttys*`).
- **Linux (`linux`)**: `bwrap` (bubblewrap) with `--ro-bind / /`, `--dev /dev`, `--proc /proc`, and
  `--bind` for the session scratch directory.
- **Windows (`win32`)**: Win32 Mandatory Integrity Control (Low Integrity `S-1-16-4096`) +
  `CreateRestrictedToken(DISABLE_MAX_PRIVILEGE | LUA_TOKEN)` + kill-on-close Job Object, with only
  the session scratch directory (and any explicitly trusted paths) labeled Low Integrity
  `(OI)(CI)L` via `icacls.exe`. Windows evaluates No-Write-Up before DACLs, blocking writes and
  deletes to all Medium/High Integrity files and folders without administrator privileges.
- **Hosts without an available sandbox**: fails closed — shell execution refuses to run unsandboxed
  in `readonly` mode and explains how to use read-only tools (`read`, `grep`, `find`, `ls`, search
  tools) or switch to `full` mode.

The session scratch directory is allocated fresh per session under the OS temp folder, exported as
`TMPDIR`/`TEMP`/`TMP`, and removed when the session shuts down. Even writes to sibling directories
under `/tmp` outside that scratch directory are rejected.

The active AutoRAG config file passed via self-configuration remains writable on its exact path so
the agent can still edit its own config when asked.

## Switching modes

| Where | How |
| --- | --- |
| TUI | **Shift+Tab** toggles `readonly` ↔ `full`. `/permissions` opens a picker; `/permissions readonly` or `/permissions full` sets a mode directly. The footer shows `permissions: read-only` or `permissions: full`. |
| CLI | `autorag tui --permission-mode full`, `autorag search "…" --permission-mode readonly` |
| Environment | `AUTORAG_PERMISSION_MODE=full` |
| Config | `"permissionMode": "readonly"` in `~/.autorag/config.json` |

Precedence is flag, then environment, then config file, then the built-in default (`readonly`).
`read-only` is accepted as a case-insensitive alias for `readonly`. Any unknown value is a
configuration error — never a silent fallback to `full`.

Shift+Tab replaces Pi's thinking-level cycle key inside `autorag tui`.

## Headless and bridged sessions

`autorag search` and bridged `autorag tui` sessions (`--telegram`, `--discord`, `--whatsapp`) also
default to `readonly` so an unattended or chat-driven run cannot mutate files on disk unless you
explicitly opt in with `--permission-mode full`, `AUTORAG_PERMISSION_MODE=full`, or
`"permissionMode": "full"` in config.

## MCP interplay

`autorag-mcp` (Lite) exposes retrieval, report, refresh, and evidence tools only; it has no tool
that deletes, moves, or edits your files, and it does not run the agent loop, so the permission
gate does not apply to it. See [MCP server (Lite)](mcp-server.md#tool-surface).
