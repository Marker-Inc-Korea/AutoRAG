---
name: autorag-setup
description: Configure AutoRAG for first use or repair its setup. Copy this agent's current LLM provider/model/endpoint/auth-env into ~/.autorag/config.json (orchestrator = current or strongest reasoning model; explorer = fastest-TPS sibling on the same auth), verify with autorag health, propose OS-aware document folders for approval, initialize configuration, and build indexes (parsed mirrors, BM25, MinSync vectors, optional Jikji maps).
---

# AutoRAG Setup Skill

Use this skill for first-time setup, missing or broken model configuration,
provider/authentication discovery, changing indexed folders, or rebuilding the
initial indexes. After setup succeeds, use the separate `autorag` skill for
normal searches and feedback.

## Safety boundaries

- Inspect only non-secret provider/model metadata and credential availability.
- Never print, copy, migrate, compare, or persist credential values. Write only
  environment-variable *names* (`apiKeyEnv`), never key material.
- `~/.autorag/config.json` must contain the real `provider`, `id`, `api`, and
  `baseUrl` AutoRAG will call. Do not dump internal model catalogs or secret
  payloads in chat, logs, or other files.
- Do not scan the whole filesystem or home directory without explicit approval.
- Never move, rename, edit, or delete source documents.
- Recommend document-dense folders only. Do not index system trees, app bundles,
  caches, or credential stores.

## Copy this agent's LLM setup into AutoRAG

AutoRAG does **not** inherit the host agent's session. The parent orchestrator
and explorer are separate Pi calls. If you omit `agents`, search may fall back
to a Codex Responses provider in `~/.codex/config.toml` and still default the
ids to `gpt-5.6-sol` / `gpt-5.6-luna` — not whatever model *you* are running.
First-time setup must therefore **translate this agent's live LLM setup into
explicit AutoRAG role models** and write them to `~/.autorag/config.json`.

Do not begin by asking the user to name a provider. Read the runtime you are
already using.

1. Preserve explicit user choices and a working `~/.autorag/config.json`.
2. Read **this agent's** current provider, model id, wire API, base URL, and
   credential env-var name from session metadata, advertised models, and the
   config files this agent already uses. Typical non-secret sources:
   - the model/provider you were launched with (session, CLI flags, env);
   - Codex: `~/.codex/config.toml` (`model_provider`, `model_providers.*.base_url`,
     `wire_api`, `env_key`; the top-level `model` field is the host default, not
     an AutoRAG role);
   - Anthropic/Claude Code-style runs: catalog provider `anthropic` plus
     `ANTHROPIC_API_KEY` presence, current model id, wire `anthropic-messages`;
   - OpenAI-compatible proxies (OpenRouter, Fireworks, LiteLLM, corp gateways):
     `baseUrl` + `api` + `apiKeyEnv`;
   - `~/.autorag/pi-agent/models.json` and Pi auth *identities* (not payloads).
3. A consumer ChatGPT / Claude / Gemini subscription is usable only when this
   runtime can actually call that provider as an API (key or delegated session
   already available to AutoRAG). A subscription is not automatically an API
   entitlement.
4. Do not infer usability from an installed CLI or a filename alone.
   Authentication and protocol compatibility must both be established.

Map what you find onto AutoRAG's `AgentModelConfig`:

| Host fact | AutoRAG field |
|---|---|
| Provider name this agent already calls | `provider` |
| Model id this agent already sends | `id` (orchestrator; explorer may differ) |
| Chat Completions / Responses / Anthropic Messages / Codex Responses / Azure | `api` |
| OpenAI-compatible or custom gateway URL | `baseUrl` (required when the provider is not in the pi-ai catalog) |
| Env var that already holds the key | `apiKeyEnv` (name only) |

Allowed `api` values: `openai-completions`, `openai-responses`,
`anthropic-messages`, `openai-codex-responses`, `azure-openai-responses`.
When `baseUrl` is set and `api` is omitted, AutoRAG defaults to
`openai-completions`. Codex `wire_api = "responses"` must be written as
`api: "openai-responses"`.

If nothing compatible is authenticated, report the missing public
provider/key/protocol. Do not write a config that cannot run. Ask one concise
question only when two public providers are equally usable and evidence cannot
choose.

## Select role models

Write **both** roles explicitly. Same provider, endpoint, `api`, and
`apiKeyEnv` unless the user asked otherwise.

- `agents.orchestrator`: this agent's current model when it is a capable
  reasoning/high-context model; otherwise the strongest reliable sibling the
  same auth can call.
- `agents.explorer`: the **fastest tokens-per-second** (lowest-latency / mini /
  flash / haiku-class) model the same authenticated runtime can actually call,
  with enough context to read documents. Prefer throughput over flagship
  quality. Explorers do high-recall `read`/`grep`/`find`/`ls`, not final
  judgment.
- If the runtime only exposes one callable model, use it for both roles.
- Do not leave `agents` unset hoping search-time Codex fallback will "use my
  model" — it will not.
- Do not invent ids. Only write models this runtime can already call.
- Preserve an existing working explicit pair unless the user asked to change
  models or health fails.

Provider and id must always be supplied together for each role.

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

Write approved folders plus the **copied** role models into
`~/.autorag/config.json` (not the workspace `.autorag/` index dir).

Prefer `autorag init` when provider+id are enough (pi-ai catalog models, or a
Codex provider whose `baseUrl` already lives in `~/.codex/config.toml`):

```bash
autorag init \
  --search-paths "/path/to/docs,/path/to/notes" \
  --orchestrator-model-provider PROVIDER \
  --orchestrator-model-id ORCHESTRATOR_MODEL \
  --explorer-model-provider PROVIDER \
  --explorer-model-id EXPLORER_MODEL
```

When this agent uses a custom/OpenAI-compatible endpoint, `init` flags only
store provider+id. After init, add `api`, `baseUrl`, and `apiKeyEnv` on each
role in `~/.autorag/config.json` so AutoRAG does not need a catalog entry:

```json
{
  "searchPaths": ["/path/to/docs"],
  "agents": {
    "orchestrator": {
      "provider": "openrouter",
      "id": "anthropic/claude-sonnet-4",
      "api": "openai-completions",
      "baseUrl": "https://openrouter.ai/api/v1",
      "apiKeyEnv": "OPENROUTER_API_KEY"
    },
    "explorer": {
      "provider": "openrouter",
      "id": "openai/gpt-4o-mini",
      "api": "openai-completions",
      "baseUrl": "https://openrouter.ai/api/v1",
      "apiKeyEnv": "OPENROUTER_API_KEY"
    }
  }
}
```

That file is the durable model setting. Use `--config PATH` (or `AUTORAG_CONFIG`)
for a non-default location. Optional:

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

Do not omit role models during setup. Leaving `agents` unset is only for an
explicit user request to rely on search-time resolution; that path does not
copy this agent's current model ids. `autorag init` must never inject a
provider default of its own. Legacy cwd `autorag.config.json` is a migration
source only and is never deleted by init.

## Verify the copied models

This is the model-connection test. Run it after writing `agents`, before
claiming setup worked and before a long `refresh` when models were just
changed:

```bash
autorag health
autorag health --skip-probes   # resolution + credential presence only, no network
autorag health --timeout-ms 20000
```

`health` resolves both roles from `~/.autorag/config.json` (or `--config` /
`AUTORAG_CONFIG`), checks that each role's API key env is present, then unless
`--skip-probes` is set issues a real completion per role and a lightweight
explorer subagent probe. It does not touch indexes.

Pass criteria: both orchestrator and explorer resolve, auth is present, probes
succeed. On failure, fix `provider` / `id` / `api` / `baseUrl` / `apiKeyEnv`
(or the env var itself) and re-run `health`. Do not proceed to search, and do
not tell the user the host LLM was "copied", until this passes.

Per-run overrides (flags and env take precedence over the file) are available
via the role-specific flags or:

- `AUTORAG_ORCHESTRATOR_MODEL_PROVIDER` / `AUTORAG_ORCHESTRATOR_MODEL_ID`
- `AUTORAG_EXPLORER_MODEL_PROVIDER` / `AUTORAG_EXPLORER_MODEL_ID`
- legacy single-model aliases: `AUTORAG_MODEL_PROVIDER` / `AUTORAG_MODEL_ID`
- `AUTORAG_SEARCH_PATHS`, `AUTORAG_WORKSPACE`, `AUTORAG_MEMORY_PATH`

## Configure datasource skills (optional)

External datasources (WhatsApp, Telegram, Slack, Discord, Notion, GitHub Issues/PRs, Google
Drive, Gmail, local mail exports, Obsidian vaults, RSS/news) are configured
as **datasource skills** in the trusted config file — never via model/tool
arguments. Add a `datasources` section (skill name → config) plus a trusted
`datasourceAccess` allow-list to `config.json`:

```jsonc
{
  "datasources": {
    "whatsapp": { "instanceId": "personal", "connector": { "binaryPath": "wacrawl" } },
    "telegram": { "instanceId": "personal", "connector": { "binaryPath": "telecrawl" } },
    "slack":    { "connector": { "binaryPath": "slacrawl", "configPath": "/path/to/slacrawl.yaml", "syncSource": "primary" } },
    "discord":  { "connector": { "tokenEnv": "DISCORD_BOT_TOKEN", "guildId": "..." } },
    "notion":   { "connector": { "binaryPath": "notcrawl", "configPath": "/path/to/notcrawl.yaml" } },
    "github":   { "connector": { "repos": ["owner/repo"], "tokenEnv": "GITHUB_TOKEN" } },
    "gdrive":   { "connector": { "backend": "rclone", "remote": "gdrive:" } },
    "gmail":    { "connector": { "backend": "himalaya", "account": "gmail", "folder": "INBOX" } },
    "mail-export": { "connector": { "paths": ["/path/to/exports"] } },
    "obsidian": { "connector": { "vaultPath": "/path/to/vault" } },
    "rss":      { "connector": { "feeds": [{ "url": "https://example.com/feed.xml" }] } }
  },
  "datasourceAccess": {
    "allowedTags": ["whatsapp", "telegram", "slack", "github", "rss"],
    "allowedScopes": ["/whatsapp/**", "/telegram/**", "/slack/**", "/github/**", "/rss/**"]
  }
}
```

Rules:

- REST connector tokens are configured as **environment variable names** (`tokenEnv`), never
  raw secrets in the file. Each skill has a default env var
  (`DISCORD_BOT_TOKEN`, `GITHUB_TOKEN`,
  `GDRIVE_ACCESS_TOKEN`, `GMAIL_ACCESS_TOKEN`).
- **CLI-bridge backends** (recommended where available): `gmail` accepts
  `"backend": "himalaya"` to index any IMAP/Maildir account the external
  [himalaya](https://pimalaya.org) CLI has configured (no OAuth plumbing);
  `gdrive` accepts `"backend": "rclone"` to index a Google Drive remote (or
  any of rclone's 70+ backends) configured via `rclone config` — Docs/Sheets
  are exported as text through `--drive-export-formats`. Auth lives entirely
  in the external tool's own config, matching the katok pattern.
- `whatsapp` requires the external
  [wacrawl](https://github.com/openclaw/wacrawl) binary
  (`brew install openclaw/tap/wacrawl`). Optional trusted connector fields are
  `binaryPath`, `databasePath`, and `sourcePath`. Live desktop ingestion is
  macOS-only; the crawler owns Full Disk Access and archive setup.
- `telegram` requires the external
  [telecrawl](https://github.com/openclaw/telecrawl) binary
  (`brew install openclaw/tap/telecrawl`). Optional trusted connector fields
  are `binaryPath`, `databasePath`, and `sourcePath`. Live desktop ingestion
  is macOS-only; the crawler owns Full Disk Access and archive setup.
- `slack` requires the external
  [slacrawl](https://github.com/openclaw/slacrawl) binary
  (`brew install openclaw/tap/slacrawl`). Optional trusted connector fields
  are `binaryPath`, `configPath`, and `syncSource`; Slack credentials and
  remote source definitions stay in slacrawl's own configuration.
- `notion` requires the external
  [notcrawl](https://github.com/openclaw/notcrawl) binary
  (`brew install openclaw/tap/notcrawl`). Optional trusted connector fields
  are `binaryPath` and `configPath`; Notion credentials and workspace
  definitions stay in notcrawl's own configuration.
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
- `health` is the model test (see **Verify the copied models**): both roles
  must resolve and probe successfully without printing credentials.
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
