# AutoRAG for Hermes Agent

Search your own files and chat/mail archives from Hermes. Indexing and retrieval run on your machine.
This is a portable [Agent Plugins v1](https://agent-plugins.org) package
(`plugin.json` + `mcp.json` + `skills/`) that runs the model-free AutoRAG Lite
MCP server from [`@autorag/librarian`](https://www.npmjs.com/package/@autorag/librarian).
Hermes keeps its own model; AutoRAG supplies retrieval.

- Local files in place: PDF, DOCX, PPTX, XLSX, CSV, Markdown, email, and Korean
  HWP/HWPX. No copy or upload step; the index lives under the AutoRAG workspace.
- Chat and mail archives through local crawler CLIs: Slack, Discord, Telegram,
  WhatsApp, KakaoTalk, email, plus Notion, Obsidian, GitHub, and RSS connectors.
- Hybrid retrieval: BM25, dense (Qwen3-Embedding-0.6B), and sparse search, plus
  instant file-name search.
- Evidence, not vibes: every result carries its source; `autorag.report` and
  `autorag.evidence` persist and re-open the exact chunks behind an answer.
- `autorag.duplicates` finds exact and near-duplicate documents; it never
  deletes anything.

## Install

```bash
hermes plugins install autorag
hermes plugins enable autorag
```

Then start a new session and say: **"set up AutoRAG for my Documents folder"**.
Hermes loads the bundled `setup` skill, which writes the config, connects the
Slack, Discord, Telegram, WhatsApp, KakaoTalk and email archives it finds on
your machine, builds the first index, and checks that a search works. After
that, just ask questions: "what did the vendor say in Slack, and does it match
the contract?"

Timing on a machine with no cargo and an empty cache: about 40 s to download
the pinned package, then about 2.5 minutes for the first index of a small
folder (it downloads about 1.5 GB of local models). If `cargo` is installed,
AutoRAG compiles one helper itself (`minsync`), which added about 6 minutes on
the test machine. Later refreshes are incremental. Requires Node.js 24 or newer.

Without the setup skill:

```bash
mkdir -p ~/.autorag/workspace
npx -y -p @autorag/librarian@2.13.8 autorag setup \
  --search-paths "$HOME/Documents" --workspace ~/.autorag/workspace
npx -y -p @autorag/librarian@2.13.8 autorag lite refresh
```

Hermes needs its MCP support (the `mcp` extra, part of Hermes's `[all]`
install). If the tools are missing after enabling the plugin, run
`hermes plugins list` to confirm it is enabled, then `hermes pm install --extra mcp`.
The MCP server reads its config when it starts: after the first setup, run
`/reload-mcp` (or open a new session) once.

## What you get

Nine tools, plus one `search_datasource_<id>` tool per connected archive, all
prefixed `mcp__autorag__` in Hermes:

| Tool | Purpose |
| --- | --- |
| `autorag_status` | Index freshness and health |
| `autorag_refresh` | Incremental or forced re-index (absent in read-only mode) |
| `autorag_search` | Fused search over files and datasources |
| `autorag_search_files` | File and folder name search |
| `autorag_search_datasource_<id>` | Search one connected archive |
| `autorag_datasources_list` / `_get` | What is connected |
| `autorag_duplicates` | Duplicate and near-duplicate scan |
| `autorag_report` / `autorag_evidence` | Persist and re-open cited evidence |

Skills (namespaced `autorag:`): `search` (how to query, cite, and compare
across sources) and `setup` (folders, datasources, first index, verification).

## Search pane (Hermes Desktop)

In Hermes Desktop the plugin adds a search pane: press `Cmd+Alt+F` (`Ctrl+Alt+F` elsewhere), use the status-bar
magnifier, or run **AutoRAG search** from the command palette. `Enter` searches file names, local document contents
and every connected datasource at once; each source gets its own tab, filled as it answers. Results are filtered for
relevance: an exact or Korean-stem word match is kept, and the rest are judged by your configured Jev/sLLM decision
model (if none is reachable, nothing is filtered out).

Select results and press **Add to chat**. Files are staged through Hermes' own `file.attach` and an `@file:` reference
is placed in the current chat's input; chat and mail hits are inserted as quoted blocks naming their source, author and
time. Nothing is sent until you send the message. The backend (`dashboard/plugin_api.py`) runs the pinned
`autorag lite find --json` as a child process with a reduced environment (no provider API keys from the gateway).
It needs Node.js 24+ and a built index, like the MCP tools. The pane is Desktop-only; the CLI and messaging gateways
use the MCP tools.

The first time the plugin loads (or the pane opens), the backend installs the pinned `@autorag/librarian` once into
`$HERMES_HOME/plugin-data/autorag/runtime` with `npm install` (about 1 GB of dependencies and about 25 s on a warm
cache; the pane shows "Getting AutoRAG ready" until it finishes). Searches then run with `node` from that folder, so a
search never waits on npm. Changing the version pinned in `mcp.json` reinstalls on the next load.

Desktop opt-in: Hermes ships a package's desktop half switched off. Turn on **AutoRAG search** once in
Capabilities → Plugins; it then sits next to the file tree in the right sidebar, and opens with `Cmd+Alt+F`.

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

## Disclosure

- **First run downloads software.** `npx` fetches the pinned
  `@autorag/librarian@2.13.8` from npm (about 11 MB unpacked, plus its
  dependencies). The first `autorag_refresh` (or `autorag lite refresh`) then
  provisions local helpers: `minsync` (`cargo install` when Rust is present,
  otherwise a sha256-verified GitHub release asset from NomaDamas/MinSync),
  `jikji` (`cargo install`), and the local models into
  `~/.cache/minsync/models` from Hugging Face: Qwen3-Embedding-0.6B (about
  1.2 GB) and KURE-Sparse-nano (about 290 MB). `dupey` is used when installed.
  Downloads happen during refresh, never during a search.
- **Reads outside the plugin's own data.** It reads the folders in your AutoRAG
  config and, for each datasource you configure, the local archive of that
  crawler CLI (for example `~/.slacrawl`, `~/.telecrawl`). It never writes to
  your documents. It writes indexes under the workspace's `.autorag/` directory,
  Jikji's data directory, and `~/.autorag` (config, memory, and report sessions).
- **Child processes.** `autorag-mcp` runs as a long-lived stdio child of Hermes.
  Refresh spawns local helpers: `minsync`, `jikji`, `dupey`, `fsearch-cli` (or
  Spotlight/Everything where applicable), and the crawler CLIs you configured.
- **Where your data goes.** AutoRAG itself does not upload documents. But the
  chunks a search returns are read by your Hermes model, so they go wherever
  that model runs (a cloud provider unless you use a local model). Remote
  embedders or rerankers are used only if you configure them in `config.json`.
  Network-backed connectors (Notion, GitHub, RSS, Hister) and crawler CLIs you
  sync yourself talk to their own services under their own login.
- **No telemetry, no self-update.** The package contains no analytics. The
  server never replaces its own files; the version is pinned in `mcp.json` and
  changes only through a reviewed catalog SHA bump.
- **No credentials.** The plugin does not read `.env` files, keychains, browser
  profiles, or another tool's token files.
- **Hermes approvals.** The server's tools are annotated `readOnlyHint` except
  `autorag_refresh` and `autorag_report`, which write only AutoRAG's own index
  and report state.
- **Search pane.** `desktop/plugin.js` talks only to this plugin's own `/api/plugins/autorag/` routes and to the
  gateway's `file.attach`. It does not read files itself; `lite find` reads the folders and archives you configured.
  The backend runs `npm install` of the pinned package once (see above), from the registry configured for npm on this
  machine, and writes only under `$HERMES_HOME/plugin-data/autorag`.
  The relevance step may send result snippets to your configured decision model (Jev or sLLM), the same data path as
  any search whose results your model reads.

## Develop and verify

```bash
hermes plugins validate plugins/hermes
hermes plugins doctor plugins/hermes --ci
uv run --with pytest --with fastapi --with httpx pytest plugins/hermes-tests -q   # backend
node --test plugins/hermes-tests/desktop/                                          # desktop controller
plugins/hermes-tests/e2e/run.sh   # live Electron E2E; needs a built hermes-agent checkout (see the script header)
```

Health check without Hermes:

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  | npx -y -p @autorag/librarian@2.13.8 autorag-mcp
```

Source of truth for the tools and schemas:
[`src/mcp/server.ts`](../../src/mcp/server.ts). License: Apache-2.0.
