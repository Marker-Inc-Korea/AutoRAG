---
title: "Hermes plugin"
description: "Install the AutoRAG Agent Plugins package in Hermes for local hybrid retrieval, evidence, and a desktop search pane."
---

# Hermes plugin

AutoRAG ships a portable Agent Plugins v1 package for Hermes at
[`plugins/hermes/`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/plugins/hermes/README.md).
It runs the model-free AutoRAG Lite MCP server from `@autorag/librarian`, so
Hermes keeps its own model and AutoRAG supplies retrieval. This page covers
install, what the plugin exposes, and the dashboard and desktop pieces that ship
in the package.

## Install

```bash
hermes plugins install autorag
hermes plugins enable autorag
```

Then start a new session and say: **"set up AutoRAG for my Documents folder"**.
Hermes loads the bundled `setup` skill, which writes the config, connects the
Slack, Discord, Telegram, WhatsApp, KakaoTalk, and email archives it finds on your
machine, builds the first index, and checks that a search works.

Hermes needs its MCP support (the `mcp` extra, part of Hermes's `[all]` install).
If the tools are missing after enabling the plugin, run `hermes plugins list` to
confirm it is enabled, then `hermes pm install --extra mcp`. The MCP server reads
its config when it starts: after the first setup, run `/reload-mcp` (or open a new
session) once.

Requires Node.js 24 or newer.

## What it exposes

The plugin declares a stdio MCP server named `autorag` in
[`mcp.json`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/plugins/hermes/mcp.json):

```json
{
  "mcpServers": {
    "autorag": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "-p", "@autorag/librarian@2.13.12", "autorag-mcp"],
      "cwd": "${PLUGIN_DATA}"
    }
  }
}
```

Hermes prefixes these tools with `mcp__autorag__`:

| Hermes tool | Purpose |
| --- | --- |
| `mcp__autorag__autorag_status` | Index freshness and health |
| `mcp__autorag__autorag_refresh` | Incremental or forced re-index (absent in read-only mode) |
| `mcp__autorag__autorag_search` | Fused search over files and datasources |
| `mcp__autorag__autorag_search_files` | File and folder name search |
| `mcp__autorag__autorag_search_datasource_<id>` (for example `mcp__autorag__autorag_search_datasource_slack`) | Search one connected archive |
| `mcp__autorag__autorag_datasources_list` / `..._get` | What is connected |
| `mcp__autorag__autorag_duplicates` | Duplicate and near-duplicate scan |
| `mcp__autorag__autorag_convert` | Convert files to another format with Filee (absent in read-only mode); never replaces a file |
| `mcp__autorag__autorag_convert_formats` | What a source format can be converted to |
| `mcp__autorag__autorag_report` / `..._evidence` | Persist and re-open cited evidence |

The underlying server tool names are the same as in
[MCP server (Lite)](mcp-server.md); Hermes rewrites the dots to underscores and
adds the prefix. The plugin also carries four skills, namespaced `autorag:`:

- `search` — how to query, name sources in plain text, and compare across sources.
- `setup` — folders, datasources, first index, verification.
- `automate` — run a Hermes agent task whenever files appear in a folder.
- `schedule` — Hermes cron jobs that re-index on a schedule, only on AC power or
  inside time windows, and report new duplicate documents.

## Search pane (Hermes Desktop)

In Hermes Desktop the plugin adds a search pane: press `Cmd+Alt+F` (`Ctrl+Alt+F`
elsewhere), use the status-bar magnifier, or run **AutoRAG search** from the
command palette. `Enter` searches file names, local document contents, and every
connected datasource at once; each source gets its own tab, filled as it answers.
Results are filtered for relevance: an exact or Korean-stem word match is kept,
and the rest are judged by your configured Jev/sLLM decision model (if none is
reachable, nothing is filtered out).

Select results and press **Add to chat**. Files are staged through Hermes' own
`file.attach` and an `@file:` reference is placed in the current chat's input;
chat and mail hits are inserted as quoted blocks naming their source, author, and
time. Nothing is sent until you send the message.

The backend is
[`dashboard/plugin_api.py`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/plugins/hermes/dashboard/plugin_api.py);
it runs the pinned `autorag lite find --json` as a child process with a reduced
environment (no provider API keys from the gateway). The pane is Desktop-only;
the CLI and messaging gateways use the MCP tools.

The first time the plugin loads (or the pane opens), the backend installs the
pinned `@autorag/librarian` once into
`$HERMES_HOME/plugin-data/autorag/runtime` with `npm install` (about 1 GB of
dependencies). Searches then run with `node` from that folder, so a search never
waits on npm. Changing the version pinned in `mcp.json` reinstalls on the next
load. An installation failure is shown with its cause.

Desktop opt-in: Hermes ships a package's desktop half switched off. Turn on
**AutoRAG search** once in **Capabilities → Plugins**; the pane then sits next to
the file tree in the right sidebar and opens with `Cmd+Alt+F`. The desktop
controller is
[`desktop/plugin.js`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/plugins/hermes/desktop/plugin.js);
it talks only to this plugin's own `/api/plugins/autorag/` routes and to the
gateway's `file.attach`.

## File automations

Hermes cron has no file trigger. The `automate` skill adds one with AutoRAG's own
file automations: AutoRAG's daemon watches a folder, and each batch of new files
queues its paths and runs one Hermes cron job. The job's pre-run script hands the
agent the file list, or skips the run with no model call when nothing is queued.
Example request: "When a PDF lands in Downloads, summarize it and send it to me on
Telegram."

The helper
[`skills/automate/scripts/autorag_automation.py`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/plugins/hermes/skills/automate/scripts/autorag_automation.py)
(Python standard library only) writes a rule to `~/.autorag/automations.json`
(backed up first), a generated `fire.sh` under
`$HERMES_HOME/plugin-data/autorag/automations/`, and a gate script in
`$HERMES_HOME/scripts/`. `fire.sh` runs `hermes cron run` for that job. The daemon
is registered with launchd or systemd `--user` only if you run
`autorag automate install`. Nothing is created at install time, and `remove`
deletes everything the helper made.

## Configuration

All optional. Hermes forwards these from your shell or profile `.env` only when
they are set:

| Variable | Effect |
| --- | --- |
| `AUTORAG_CONFIG` | Path to a specific `config.json` |
| `AUTORAG_HOME` | Directory holding `config.json` and `memory.json` (default `~/.autorag`) |
| `AUTORAG_SEARCH_PATHS` | Comma-separated folders, overrides the config file |
| `AUTORAG_WORKSPACE` | Workspace that owns the `.autorag/` index |
| `AUTORAG_MCP_READ_ONLY` | `true` omits `autorag_refresh` and `autorag_report` |
| `AUTORAG_MCP_TOOLS` | Comma-separated exact tool allowlist |

The server declares no credentials. Datasource config stores only environment
variable names, never values.

**Safety.** The MCP process and the Finder force `AUTORAG_AUTH_MODE=environment`:
stored model credentials, Pi model/settings files, and Codex config discovery are
disabled, so the plugin does not reuse a Pi/Dazzi subscription login. It never
reads `.env` files, keychains, browser profiles, or another tool's token files.
The plugin itself writes only under `$HERMES_HOME/plugin-data/autorag`, the
workspace `.autorag/` directory, Jikji's data directory, and `~/.autorag` (config,
memory, report sessions). It never writes to your documents.

## Develop and verify

```bash
hermes plugins validate plugins/hermes
hermes plugins doctor plugins/hermes --ci
uv run --with pytest --with fastapi --with httpx pytest plugins/hermes-tests -q   # backend
node --test plugins/hermes-tests/desktop/                                          # desktop controller
plugins/hermes-tests/e2e/run.sh   # live Electron E2E; needs a built hermes-agent checkout
```

Health check without Hermes:

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  | npx -y -p @autorag/librarian@2.13.12 autorag-mcp
```

The plugin is Apache-2.0. For the document-scanning half, see
[Datasources](../datasources/overview.md).
