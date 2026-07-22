---
name: autorag-setup
description: Configure AutoRAG for first use or repair its setup. Detect the current agent's usable subscription-backed runtime or API provider without exposing private provider identities, select orchestrator and explorer models, propose OS-aware document folders for approval, initialize configuration, and build indexes (parsed mirrors, BM25, MinSync vectors, optional Jikji maps).
---

# AutoRAG Setup Skill

Use this skill for first-time setup, missing or broken model configuration,
provider/authentication discovery, changing indexed folders, or rebuilding the
initial indexes. After setup succeeds, use the separate `autorag` skill for
normal searches and feedback.

## Safety boundaries

- Inspect only non-secret provider/model metadata and credential availability.
- Never print, copy, migrate, compare, or persist credential values.
- Never expose private provider aliases or internal model catalogs in generated
  configuration, logs, or user-facing explanations.
- Do not scan the whole filesystem or home directory without explicit approval.
- Never move, rename, edit, or delete source documents.
- Recommend document-dense folders only. Do not index system trees, app bundles,
  caches, or credential stores.

## Detect an authenticated model runtime

Do not begin by asking the user to manually name a provider. Determine the best
usable configuration from available evidence.

1. Preserve explicit user choices and a working `~/.autorag/config.json`.
2. Inspect the current agent's exposed provider/model capabilities.
3. Inspect compatible non-secret local metadata, including configured model
   registries and `~/.autorag/pi-agent/models.json`. Provider-specific local
   configuration may be used only to determine endpoint compatibility, model
   IDs, credential environment-variable names, and whether credentials exist.
4. Check Pi authentication entries by provider identity without reading or
   showing their secret payloads.
5. Treat ChatGPT, Claude, Gemini, and other consumer subscriptions as usable
   only when the active runtime can demonstrably delegate that authenticated
   session to AutoRAG or compatible authentication already exists in AutoRAG's
   Pi state. A subscription is not automatically an API entitlement.
6. Do not infer usability from an installed CLI, a config filename, or an
   environment-variable name alone. Authentication and protocol compatibility
   must both be established.

If no compatible authenticated runtime exists, report the exact missing public
provider/authentication requirement. Do not write a configuration that cannot
run. Ask one concise question only when multiple equally suitable public
providers remain and runtime evidence cannot choose between them.

## Select role models

Select only models actually advertised by the authenticated runtime:

- `agents.orchestrator`: strongest reliable reasoning and high-context model.
- `agents.explorer`: faster, cheaper high-recall model with sufficient context.
- If only one usable model exists, configure it for both roles.
- Preserve an existing working explicit pair.
- Never invent model IDs or write a private provider alias into distributed or
  user-facing configuration.

Provider and model ID must always be supplied together for each configured role.
If role models are intentionally omitted, leave them unset so search-time
resolution can still pick up an authenticated local runtime when available.

## Discover and approve document folders

Propose a **short list of recommended folders** tailored to the user's OS, then
let the user accept, drop, or replace entries before any indexing. Explicit user
paths always win. Reuse previously approved folders without re-asking.

### 1. Detect OS and home roots

Resolve the user home once:

| OS | User home |
|---|---|
| macOS | `$HOME` (e.g. `/Users/<name>`) |
| Linux | `$HOME` (e.g. `/home/<name>`; honor `XDG_*` when set) |
| Windows | `%USERPROFILE%` / `$HOME` under Git Bash (e.g. `C:\Users\<name>`) |

Do not walk the entire home tree. Check only the recommended candidates below
for existence and approximate document density (counts of supported extensions),
then present the findings.

### 2. Recommend document-dense anchors per OS

Offer candidates that commonly hold PDFs, Office docs, notes, and downloads.
Mark each as **recommended** or **optional**, and never enable a path that does
not exist.

**macOS**

| Priority | Path | Why |
|---|---|---|
| Recommended | `~/Documents` | Default document library |
| Recommended | `~/Downloads` | Fresh PDFs, reports, attachments |
| Recommended | `~/Desktop` | Dropped working files |
| Optional | `~/Notes`, `~/Obsidian`, `~/iCloud Drive/Documents` when present | Personal knowledge bases |
| Optional | Current project docs roots the user already named | Repo/wiki collections |

Prefer user-visible folders over iCloud container UUIDs or Library internals.
Skip `~/Library`, `~/Applications`, and Time Machine volumes unless the user
explicitly points there.

**Linux**

| Priority | Path | Why |
|---|---|---|
| Recommended | `~/Documents` or `$XDG_DOCUMENTS_DIR` | Document library |
| Recommended | `~/Downloads` or `$XDG_DOWNLOAD_DIR` | Incoming files |
| Recommended | `~/Desktop` or `$XDG_DESKTOP_DIR` | Working files |
| Optional | `~/Notes`, `~/Sync`, Nextcloud/Syncthing folders when present | Synced knowledge |
| Optional | Current project docs roots the user already named | Repo/wiki collections |

When XDG user-dirs are configured, prefer those resolved paths over bare
`~/Documents`-style guesses.

**Windows**

| Priority | Path | Why |
|---|---|---|
| Recommended | `%USERPROFILE%\Documents` | Document library (also OneDrive Documents if that is the real shell folder) |
| Recommended | `%USERPROFILE%\Downloads` | Incoming PDFs and exports |
| Recommended | `%USERPROFILE%\Desktop` | Working files |
| Optional | OneDrive `Documents` / `Desktop` when they differ from the local shell folders | Cloud-backed corp libraries |
| Optional | Current project docs roots the user already named | Repo/wiki collections |

Prefer the shell-known special folders. Do not crawl `AppData`, `Program Files`,
or system roots.

### 3. Density and format filter

Prefer folders dense with formats AutoRAG parses into `.autorag/parsed` for
BM25/MinSync:

- text notes: `md`, `markdown`, `txt`, `text`
- documents: `pdf`, `docx`, `pptx`, `xlsx`, `hwpx`, `eml`
- optional OCR images only when enabled: `jpg`, `jpeg`, `png`, `bmp`, `tiff`

Do not present legacy binary shells that currently fail pure-JS parsing as fully
supported: bare `doc`, `xls`, and legacy `hwp` (use `docx` / `xlsx` / `hwpx`
instead). Explorers can still open any readable text under approved paths with
`read`/`grep`/`find`/`ls`, but those files will not contribute to BM25/MinSync
indexes.

Skip generated/vendor directories including `node_modules`, `.git`, `dist`,
`build`, `target`, `.cache`, `.autorag`, and `.jikji`.

### 4. Present a proposal, then require approval

Summarize a concrete proposal, for example:

```text
Recommended index roots (macOS):
  [R] ~/Documents      (~120 pdf/md/docx)
  [R] ~/Downloads      (~45 pdf/pptx)
  [R] ~/Desktop        (~12 md/pdf)
  [O] ~/Notes          (~80 md) — optional personal vault

Reply with: accept all / keep only Documents+Downloads / custom list
```

Rules:

1. Prefer an explicit directory already named by the user over any suggestion.
2. Do not silently index large or sensitive trees.
3. Keep the first-run set small (typically 1–3 approved roots). Users can add
   more after the initial index build.
4. Wallpaper/background folders are rarely useful; only suggest Desktop itself
   (where people leave docs), not OS wallpaper asset caches.

## Initialize

Write the approved paths and selected role models:

```bash
autorag init \
  --search-paths "/path/to/docs,/path/to/notes" \
  --orchestrator-model-provider PROVIDER \
  --orchestrator-model-id ORCHESTRATOR_MODEL \
  --explorer-model-provider PROVIDER \
  --explorer-model-id EXPLORER_MODEL
```

This writes `~/.autorag/config.json` by default. Use `--config PATH` (or
`AUTORAG_CONFIG`) for an explicit location. Optional:

- `--workspace DIR` for the workspace that owns `.autorag` indexes
- `--memory-path FILE` for retrieval-memory storage
- `--force` only when intentionally replacing an existing config

### Indexing method defaults

BM25 and MinSync are **enabled by default**. `autorag init` writes
`bm25: { enabled: true }` and `minSync: { enabled: true, autoInstall: false }`
into the config even when no method flags are supplied. To disable a method,
set it to `false` in the config file (`"bm25": false` or `"minSync": false`).

MinSync auto-install is off by default (`autoInstall: false`); the binary must
be pre-installed or available on `PATH`. AutoRAG never forces TEI or any
external embedding service.

### MinSync embedder flags

`autorag init` accepts non-secret embedder configuration flags that are written
into `minSync.embedder` in the config:

```bash
autorag init \
  --search-paths "/path/to/docs" \
  --embedder-id "text-embedding-3-small" \
  --embedder-base-url "https://api.openai.com/v1" \
  --embedder-api-key-env "OPENAI_API_KEY" \
  --embedder-dimension 1536 \
  --embedder-batch-size 64
```

All fields are optional; only provided fields are written. `--embedder-api-key-env`
accepts the **environment variable name** (e.g. `OPENAI_API_KEY`), never the key
value itself. `--embedder-dimension` and `--embedder-batch-size` must be positive
integers.

If role models are intentionally omitted, `autorag init` must leave `agents`
unset; it must never inject a private provider default. Legacy cwd
`autorag.config.json` is a migration source only and is never deleted by init.

Per-run overrides (flags and env take precedence over the file) are available
via the role-specific flags or:

- `AUTORAG_ORCHESTRATOR_MODEL_PROVIDER` / `AUTORAG_ORCHESTRATOR_MODEL_ID`
- `AUTORAG_EXPLORER_MODEL_PROVIDER` / `AUTORAG_EXPLORER_MODEL_ID`
- legacy single-model aliases: `AUTORAG_MODEL_PROVIDER` / `AUTORAG_MODEL_ID`
- `AUTORAG_SEARCH_PATHS`, `AUTORAG_WORKSPACE`, `AUTORAG_MEMORY_PATH`

## Configure datasource skills (optional)

External datasources (Slack, Discord, Notion, GitHub Issues/PRs, Google
Drive, Gmail, local mail exports, Obsidian vaults, RSS/news) are configured
as **datasource skills** in the trusted config file — never via model/tool
arguments. Add a `datasources` section (skill name → config) plus a trusted
`datasourceAccess` allow-list to `config.json`:

```jsonc
{
  "datasources": {
    "slack":    { "connector": { "tokenEnv": "SLACK_BOT_TOKEN", "channels": ["eng"] } },
    "discord":  { "connector": { "tokenEnv": "DISCORD_BOT_TOKEN", "guildId": "..." } },
    "notion":   { "connector": { "tokenEnv": "NOTION_TOKEN" } },
    "github":   { "connector": { "repos": ["owner/repo"], "tokenEnv": "GITHUB_TOKEN" } },
    "gdrive":   { "connector": { "backend": "rclone", "remote": "gdrive:" } },
    "gmail":    { "connector": { "backend": "himalaya", "account": "gmail", "folder": "INBOX" } },
    "mail-export": { "connector": { "paths": ["/path/to/exports"] } },
    "obsidian": { "connector": { "vaultPath": "/path/to/vault" } },
    "rss":      { "connector": { "feeds": [{ "url": "https://example.com/feed.xml" }] } }
  },
  "datasourceAccess": {
    "allowedTags": ["slack", "github", "rss"],
    "allowedScopes": ["/slack/**", "/github/**", "/rss/**"]
  }
}
```

Rules:

- Tokens are configured as **environment variable names** (`tokenEnv`), never
  raw secrets in the file. Each skill has a default env var
  (`SLACK_BOT_TOKEN`, `DISCORD_BOT_TOKEN`, `NOTION_TOKEN`, `GITHUB_TOKEN`,
  `GDRIVE_ACCESS_TOKEN`, `GMAIL_ACCESS_TOKEN`).
- **CLI-bridge backends** (recommended where available): `gmail` accepts
  `"backend": "himalaya"` to index any IMAP/Maildir account the external
  [himalaya](https://pimalaya.org) CLI has configured (no OAuth plumbing);
  `gdrive` accepts `"backend": "rclone"` to index a Google Drive remote (or
  any of rclone's 70+ backends) configured via `rclone config` — Docs/Sheets
  are exported as text through `--drive-export-formats`. Auth lives entirely
  in the external tool's own config, matching the katok pattern.
- Access is **default-deny**: without `datasourceAccess.allowedTags`, no
  datasource skill is announced or searchable, even when configured.
  `allowedScopes` are opaque slash-hierarchical roots like `/slack/<instance>/**`.
- Per-skill options: `instanceId`, `pollingIntervalMs`, `tags`, and the
  connector-specific `connector` object. `false` or `{ "enabled": false }`
  disables an entry. Unknown skill names fail config resolution.
- `autorag refresh --method datasources` indexes them; chunks persist under
  `<workspace>/.autorag/datasources/<skill>/<instance>/`.
- Verify with the manual QA harness: `bun scripts/manual-qa/run-qa.ts`
  (see `docs/manual-qa-datasources.md`).

## Build indexes after install

Installation + `init` alone does **not** make search useful. Immediately after
the approved config is written, build the local indexes before handing off to
the `autorag` skill.

### What `autorag refresh` builds

`autorag refresh` is the post-setup indexing step. It is model-free and:

1. **Parses** approved source files into workspace-local `.autorag/parsed`
   markdown mirrors.
2. **Prepares BM25** lexical indexes over those mirrors.
3. **Embeds into the MinSync vector DB** (semantic index) over the same mirrors
   when MinSync is configured/available.
4. **Prepares Jikji maps/caches** under each approved source's `.jikji/` when
   Jikji is configured (indexing only — find answers come later via
   `jikji_find` at search time).
5. **Indexes authorized datasources** when datasource skills are configured.

Order of operations for first-time setup:

```bash
# 1) write approved folders + role models
autorag init \
  --search-paths "$HOME/Documents,$HOME/Downloads,$HOME/Desktop" \
  --orchestrator-model-provider PROVIDER \
  --orchestrator-model-id ORCHESTRATOR_MODEL \
  --explorer-model-provider PROVIDER \
  --explorer-model-id EXPLORER_MODEL

# 2) verify model/provider auth and explorer subagent preflight
autorag health

# 3) parse sources, build BM25, embed MinSync, prepare Jikji maps
autorag refresh

# 4) verify corpus freshness / index health (path-opaque)
autorag status
```

Interpret results:

- `refresh` prints parse counts and BM25 readiness (and datasource rows when
  present). Re-run with `autorag refresh --force` only when a dirty/partial
  index needs a full rebuild path and a lighter incremental refresh is not
  enough.
- `--method <csv>` restricts which methods refresh runs:
  `bm25,minsync,parsed,datasources,jikji,all`. When omitted, all methods run.
  Parsed mirrors are always synced when BM25 or MinSync is selected (they index
  over the parsed mirrors). Example: `autorag refresh --method bm25,minsync`.
- `status` is model-free and path-opaque: inspect freshness and component
  health only; do not expect absolute source paths in the output.
- `health` checks model/provider auth and explorer subagent setup: it resolves
  both role models, verifies credential presence, and (unless `--skip-probes`)
  probes a completion call per role. Use it to confirm both role models resolve
  from the authenticated runtime or written config without displaying
  credentials or private provider details.
- Do not claim setup succeeded when authentication, indexes, role-model
  resolution, or subagent dispatch remain unverified.
- Prefer bounded `refresh` over destructive resets. Reserve
  `autorag index rebuild --yes` for a full wipe+reindex of workspace
  `.autorag` parsed/BM25/MinSync dirs only — never against source documents.
  `index reset` and `index rebuild` also accept `--method` to scope which
  indexes are removed/rebuilt (e.g. `autorag index reset --method bm25 --yes`
  removes only the BM25 index).

Large first-time corpora can take several minutes for parse + embed. Stay on
the approximate status of the running refresh rather than starting concurrent
index jobs.

## Customize and extend indexed folders

Users can change the indexed set after the initial install. Always get explicit
approval before adding paths, then re-init (or rewrite config) and refresh.

### Add or replace folders

1. Collect the desired absolute paths (user paste, or another short recommended
   proposal using the OS tables above).
2. Prefer merging with the existing approved list rather than silently dropping
   working roots, unless the user asks to replace everything.
3. Write the new list:

```bash
# Replace/extend the configured search roots, preserve role models already set
autorag init \
  --force \
  --search-paths "/existing/docs,/new/research,/path/to/notes"
```

If role models must be preserved and you are rewriting via `init --force`, pass
the same orchestrator/explorer provider+id pair already in
`~/.autorag/config.json`. Alternatively edit `searchPaths` in that config when
the agent can safely update JSON without touching credentials/private fields.

4. Rebuild indexes for the new roots:

```bash
autorag refresh
autorag status
```

### Narrow temporarily without re-init

For a one-off search scope, prefer `autorag search "…" --scope <virtual-subpath>`
through the `autorag` skill after setup. Scope narrows retrieval; it does not
add new roots to the permanent index set.

### Env/flag overrides for automation

Machine-driven agents may point at a custom list without rewriting the default
home config:

```bash
AUTORAG_SEARCH_PATHS="/docs,/notes" autorag refresh
# or
autorag refresh --search-paths "/docs,/notes"
```

Remember that CLI flags / env vars take precedence for that invocation only;
persist planned long-term folders with `autorag init` so later searches and
refreshes share the same set.

### Ongoing maintenance

```bash
autorag status                 # corpus freshness / index health
autorag refresh                # resync after documents change
autorag watch --once           # preferred single tick for cron / Task Scheduler / systemd timer
autorag watch                  # long-running fs watch for interactive sessions
autorag refresh --force        # force a full re-sync when needed
autorag index rebuild --yes    # wipe parsed/BM25/MinSync under .autorag, then forced refresh
autorag memory inspect         # read-only retrieval memory snapshot
```

After setup, install an **OS-appropriate periodic job** so indexes stay fresh without a permanent daemon. Agents MUST create or verify this schedule before claiming continuous indexing works.

| OS | Scheduler | Suggested tick (every 30 min) |
|---|---|---|
| macOS | `cron` or LaunchAgent in `~/Library/LaunchAgents` | `*/30 * * * * PATH=... NODE_OPTIONS=--max-old-space-size=16384 autorag watch --once` |
| Linux | `cron` or `systemd --user` timer | same crontab pattern, or a oneshot unit + timer |
| Windows | Task Scheduler | repeating task every 30 minutes running `autorag watch --once` |

Prefer `watch --once`/`refresh` for scheduled ticks. Reserve long-running `watch` for interactive foreground use. Log under `~/.autorag/` (or user temp), never into source document trees. Avoid overlapping concurrent refreshes.

Never run reset/rebuild against source document trees. Indexes live under the
configured workspace's `.autorag/`; Jikji prepare caches live under
per-source `.jikji/` when enabled.

## Hand off

After models authenticate, `autorag health` confirms model/provider auth and
explorer subagent dispatch, folders are approved, `init` has written the config,
`refresh` has built parsed + BM25 + MinSync (+ optional Jikji/datasource)
indexes, and `status` looks healthy:

- stop the setup skill
- use the `autorag` skill for normal queries (`autorag search`, then
  `autorag feedback` with the returned `sessionId`)
- return to this skill when the user adds folders, fixes providers/models, or
  needs a deliberate reindex
