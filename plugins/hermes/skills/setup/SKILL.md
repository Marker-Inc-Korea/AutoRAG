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

`CLI` below means: `npx -y -p @autorag/librarian@2.13.11 autorag`.

## 1. Folders: index the documents that matter, not everything

Use folders the user named. If none, propose the one or two most
document-dense of `~/Documents`, `~/Desktop`, `~/Downloads` that exist, state
them, and continue unless the user objects. Never index a system tree, app
bundle, cache, or a credential store. Use absolute paths.

**1a. Make sure the folder can be read at all.** On macOS, `~/Documents`,
`~/Desktop` and `~/Downloads` are privacy-protected: the first read from a
background process raises a permission prompt, and until someone answers it every
read of that folder hangs (measured: a plain `ls` stayed blocked for minutes and
the tools timed out one after another). Probe first, with a short limit:

```bash
perl -e 'alarm 10; exec @ARGV' ls "$FOLDER" >/dev/null 2>&1; echo $?
```

Any non-zero result (alarm kill is 142, a refusal is 1) means do **not** keep
retrying. Tell the user, in one message: "macOS is asking whether Hermes may read
<folder>. Click Allow in the dialog, or choose a folder outside
Documents/Desktop/Downloads, and say 'continue'." Then stop and wait. A folder
that probes fine needs no message.

**1b. Look before you index.** Run the read-only inventory for each folder. It
opens no file and writes nothing:

```bash
python3 "${HERMES_SKILL_DIR}/scripts/autorag_inventory.py" "$FOLDER"
```

It prints how many files AutoRAG can index (by type), and two lists of folders:

- `noise`: the file types alone say these are not documents (a project's source
  code, exported media). Exclude them without asking; `recommended_excludes` holds
  their absolute globs such as `/Users/me/projects/webapp/src/**`.
- `ask`: a pile of small text files with one or two naming patterns (`IMG_0001.txt`,
  `receipt_12.txt`, `weekly_w21.md`). The files cannot say whether this is an app's
  output or the person's own numbered reports, so **do not exclude it on your own**.
  Put it to the user in one line with the count and an example name, and leave it in
  the index unless they say to skip it. If they say yes, add its `suggested_exclude`.

Report in two or three lines, for example: "Documents has 4,200 files; 310 are
real documents. I am skipping `projects/webapp/src` (source code, 3,900 files).
`Downloads` has 180 text files named like `IMG_0116.txt` - skip it too?" Then:

- Index the folders the inventory keeps; add every `recommended_excludes` entry
  to the config's `excludePaths` (step 2 does this in one command).
- A folder of distinctly named notes is never flagged, so a person's own notes are
  not at risk. Keeping clutter is cheaper than hiding something the user wrote.
- Ask only about `ask` folders, or when the inventory says `truncated`.
  Skipped is not deleted: the user can include a folder later by removing its
  exclude and refreshing.
- Never exclude a folder the user named by hand.

`node_modules`, `.git`, `dist`, `build`, `target` and other cache folders are
already skipped by AutoRAG; the inventory lists them under `skipped_dirs`
without scanning them.

## 2. Config in one command

```bash
mkdir -p "$WORKSPACE"
# No config yet, and the inventory recommended excludes: write the config with them first.
CLI init --search-paths "/abs/one,/abs/two" --workspace "$WORKSPACE" \
  --exclude '/abs/one/Downloads/**' --exclude '/abs/one/projects/webapp/src/**'
# Then probe and record the embedding model. It keeps the excludes.
CLI setup --search-paths "/abs/one,/abs/two" --workspace "$WORKSPACE" --format json
```

Pass one `--exclude` per `recommended_excludes` entry, exactly as printed (a glob
is kept as written; a plain folder path is resolved against the current
directory). Skip `init` and run only `setup` when the inventory found nothing to
exclude. If a config already existed (step 0), do not run `init`: merge the new
entries into its `excludePaths`, keep the user's own, and back it up first.

`setup` writes a default config when none exists, records the local embedding
model, and reports what is ready. Pick `$WORKSPACE` outside the indexed folders
(for example `~/.autorag/workspace`). `ok: false` is not fatal: read
`remediation` and carry on; the usual cause is a missing embedding daemon,
handled in step 5.

## 3. Connect only archives that actually work here

Never wire an archive just because its CLI is installed. A source is connectable
only when its crawler CLI exists AND its local archive already holds data. Do
not install a crawler, log in to anything, or run an import from the source app.

Probe each one (all read-only):

| Source | CLI | Archive that must exist | Config entry |
| --- | --- | --- | --- |
| Telegram | `telecrawl` | `~/.telecrawl/telecrawl.db` with `chats` > 0 (`telecrawl status`) | `"telegram": {"instanceId": "default", "connector": {}}` |
| Slack | `slacrawl` | `~/.slacrawl/slacrawl.db`; id: `sqlite3 -readonly ~/.slacrawl/slacrawl.db "select id from workspaces"` | `"slack": {"instanceId": "default", "connector": {"syncSource": "wiretap", "workspace": "<id>"}}` |
| Discord | `discrawl` | `discrawl status` shows `guilds` > 0; id: `discrawl --json sql "select id,name from guilds"` | `"discord": {"type": "discord", "instanceId": "<name>", "connector": {"guildId": "<id>"}}` |
| WhatsApp | `wacrawl` | `~/.wacrawl/wacrawl.db` | `"whatsapp": {"instanceId": "default", "connector": {}}` |
| KakaoTalk, email, others | `lazykatok`, `mailcrawl`, ... | run `CLI setup --format json`; connect only entries whose state is not `skipped` | entry shape in `docs/datasource-skills.md` |

`CLI setup --format json` also reports `skipped` with a reason (`missing <cli>
binary`, `native store is not present`); trust it over your own guess.

Add the entries to the config by merging, never replacing. Keep every entry the
user already has, add only missing ones, and back up first:

```bash
python3 - <<'PY'
import json, os, shutil, time
cfg = os.environ.get("AUTORAG_CONFIG") or os.path.join(
    os.environ.get("AUTORAG_HOME") or os.path.expanduser("~/.autorag"), "config.json")
c = json.load(open(cfg))
shutil.copy(cfg, f"{cfg}.bak-{int(time.time())}-before-hermes-plugin")
ds = c.setdefault("datasources", {})
# ds.setdefault("telegram", {...}); one line per source that passed the probe
json.dump(c, open(cfg, "w"), indent=2)
PY
```

Give each new entry a short `"description"` of what the archive contains; the
model uses it to pick the right archive.

Then prove each new entry syncs, and remove the ones that do not. A present
archive can still fail (for example Telegram's archive exists but `telecrawl
import` needs Telegram Desktop's data folder). Run
`CLI lite refresh --method datasources --json`, read `datasources[]`, and for
every entry you added with `ok: false` delete it from the config (back up
first) and report the error text. Never remove an entry the user had before you
started. Measured on a sandbox: Telegram failed with `telecrawl nonzero-exit`,
was removed, Slack and Discord stayed, and the next refresh was all `ok`.

Finish by listing every source you skipped or removed and why.

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

**No model key is needed for any of this.** The MCP tools never call AutoRAG's
own decision model (Jev/sLLM); your Hermes model reads the results. Measured on a
clean machine with no provider key set for AutoRAG: setup, first index, and a
search that returned the right file with its source all worked. Do not ask the
user for an OpenRouter or OpenAI key, and do not treat a missing one as a failure.
A decision model only matters for the desktop search pane's relevance filter, and
when none is reachable the pane filters nothing out.

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
