---
name: setup
description: One-pass setup for the AutoRAG plugin - pick folders, write the config, auto-connect the Slack, Discord, Telegram, WhatsApp, KakaoTalk, and email archives already on this machine, build the first index, and prove a search works. Use when autorag.status or a search says index-not-ready, no folders are configured, a datasource is missing, search is keyword-only, or the autorag tools are absent.
license: Apache-2.0
compatibility: Node.js >= 24 and npx on PATH. macOS, Linux, or Windows.
---

# AutoRAG setup

Do the whole thing in one pass and report at the end. Ask the user at most one
question: which folders to index (skip it when they already named folders).
Everything below was measured on a clean machine; the timings are real.

## Rules

- Never read, print, or store credential values. Config holds env-var names only.
- Never use `--force` on an existing config. Never edit or delete the user's files.
- Index and report state go under the workspace `.autorag/` directory and
  AutoRAG's own data directories only.
- Run long steps (first index) in the background and tell the user it is
  running. Do not wait silently.

## 0. Know what is already there

Call `autorag.status` and `autorag.datasources.list` first. If a config exists,
inspect only `searchPaths`, `workspacePath`, and the datasource names, keep it,
and jump to step 4.

`CLI` below means: `npx -y -p @autorag/librarian@2.13.7 autorag`.

## 1. Folders

Use folders the user named. If none, propose the one or two most
document-dense of `~/Documents`, `~/Desktop`, `~/Downloads` that exist, state
them, and continue unless the user objects. Never index a system tree, app
bundle, cache, `node_modules`, `.git`, or a credential store. Use absolute
paths.

## 2. Config in one command

```bash
mkdir -p "$WORKSPACE"
CLI setup --search-paths "/abs/one,/abs/two" --workspace "$WORKSPACE" --format json
```

`setup` writes a default config when none exists, records the local embedding
model, and reports what is ready. Pick `$WORKSPACE` outside the indexed folders
(for example `~/.autorag/workspace`). `ok: false` is not fatal: read
`remediation` and carry on; the usual cause is a missing embedding daemon,
handled in step 5.

## 3. Connect the archives on this machine

Probe, then connect every archive that has both its CLI and a local archive:

```bash
command -v slacrawl discrawl telecrawl wacrawl lazykatok mailcrawl
```

| Source | Check the archive exists | Config entry to add under `datasources` |
| --- | --- | --- |
| Telegram | `~/.telecrawl/telecrawl.db` | `"telegram": {"instanceId": "default", "connector": {}}` |
| Slack | `~/.slacrawl/slacrawl.db`; id: `sqlite3 -readonly ~/.slacrawl/slacrawl.db "select id from workspaces"` | `"slack": {"instanceId": "default", "connector": {"syncSource": "wiretap", "workspace": "<id>"}}` |
| Discord | `discrawl status` shows `guilds>0`; id: `discrawl --json sql "select id,name from guilds"` | `"discord": {"type": "discord", "instanceId": "<name>", "connector": {"guildId": "<id>"}}` |
| WhatsApp | `~/.wacrawl/wacrawl.db` | `"whatsapp": {"instanceId": "default", "connector": {}}` |
| KakaoTalk, email | run `CLI setup --format json`, read the `datasources` states | use the entry shape `docs/datasource-skills.md` shows |

Merge into the existing config; never replace it. This keeps every entry the
user already has, adds only missing ones, and writes a backup first:

```bash
python3 - <<'PY'
import json, os, shutil, time
cfg = os.environ.get("AUTORAG_CONFIG") or os.path.join(
    os.environ.get("AUTORAG_HOME") or os.path.expanduser("~/.autorag"), "config.json")
c = json.load(open(cfg))
shutil.copy(cfg, f"{cfg}.bak-{int(time.time())}-before-hermes-plugin")
ds = c.setdefault("datasources", {})
# ds.setdefault("telegram", {...}); ds.setdefault("slack", {...}); one line per source found
json.dump(c, open(cfg, "w"), indent=2)
PY
```

Give each entry a short `"description"` saying what the archive contains; the
model uses it to pick the right archive. List every source you skipped and why
(CLI missing, no archive). Do not install or log in to a crawler unasked.

The running MCP server reads its config once, at start. Measured: a config
written after the server started was ignored (refresh did nothing, search
returned no results). So after step 2 or 3 changes the config, tell the user to
run `/reload-mcp` (or start a new session) once, then continue from step 4.

## 4. First index

Call `autorag.refresh` with `{}`. If that tool is absent (read-only server) or
you prefer not to block, run it in the terminal in the background:

```bash
CLI lite refresh --json
```

Measured on a machine with no cargo and an empty cache: the pinned package
downloads in about 40 s, then the first refresh takes about 2.5 minutes for a
small folder (it downloads the local models, about 1.5 GB). With `cargo`
installed, AutoRAG compiles `minsync` itself, which took about 6 minutes
extra. Tell the user the number that applies and that later refreshes are
incremental and fast.

## 5. Turn on semantic search

A search that returns the diagnostic `embedding-unavailable` is keyword-only
and still works. It stays that way because nothing starts the local embedding
daemon for MCP clients. Start it once; it keeps running until stopped:

```bash
BIN="$(command -v minsync || find "$WORKSPACE/.autorag/bin" -name minsync -type f | head -1)"
nohup "$BIN" serve --idle-timeout 0 >/dev/null 2>&1 &
```

Measured: with the daemon up, the `embedding-unavailable` diagnostic disappears
and search uses dense retrieval too. On a four-document corpus the top result
was the same with and without it, so do not promise better ranking. The
daemon held about 1.3 GB of memory shortly after start (MinSync documents about
2.8 GB once fully loaded); stopping that process stops it. Skip this step if
the user declines.

## 6. Prove it

1. `autorag.status` returns no `stale` and no blocking diagnostic.
2. `autorag.search` with a phrase from one of the user's own documents returns
   it in the top results.
3. For each connected archive, call its `autorag.search_datasource_<id>` tool
   with a phrase from it.

Finish with one short report: folders indexed, archives connected, archives
skipped (with reason), whether semantic search is on, and one example question
the user can ask now. Offer `autorag.duplicates` as a next step. Do not ask for
a GitHub star unless the user brings it up.

## Troubleshooting

- The `autorag` tools are missing: Hermes lacks its MCP support. Run
  `hermes plugins list` to confirm the plugin is enabled, then
  `hermes pm install --extra mcp` and start a new session. Node.js below 24 or
  a still-running first `npx` download also cause this.
- `index-not-ready`: run step 4.
- `MinSync is required`: run a full `CLI lite refresh` once and report the error.
- Nothing found where the user expects it: check the folder is in `searchPaths`,
  not under `excludePaths`, then refresh.
- Datasource tool absent after editing config: `/reload-mcp` or a new session.
- `hermes pm install --extra mcp` stops with "tools not on PATH before venv sync":
  Hermes's installer did not find a tool it manages (node, npm, ffmpeg, pnpm, or
  ripgrep) on PATH. Install the named ones with `hermes pm install <names>`, then
  retry. This happened on a machine with a minimal PATH; a normal shell may not hit it.
- Discord reports "no compatible message embeddings" and `discrawl embed --rebuild`
  says embeddings are disabled: discrawl's own config has embeddings off, so
  Discord search stays keyword-only. Report it; do not edit the user's discrawl config.
