---
name: autorag-setup
description: Install and configure AutoRAG, register its Lite MCP server for external agents, or repair its single search model, approved roots, indexes, datasources, and health checks without exposing credentials. Use when AutoRAG or MCP is missing, init/refresh/health fails, indexes are stale, or the user wants to add folders or datasources.
license: Apache-2.0
---

# AutoRAG setup

Use this skill when AutoRAG is unconfigured, the `autorag` CLI is missing, model
resolution fails, indexes are missing or stale, or the user wants to change the
document collection or datasources.

For model-free external-agent search, use `autorag-lite-setup` to configure
and register `autorag-mcp`; do not configure a search model or install a search
skill for Lite. This skill's model setup and CLI search apply to the full
librarian, not the Lite MCP server.

## Safety

- Inspect only non-secret provider/model metadata and credential availability.
- Never print, copy, migrate, compare, or persist credential values. Store only
  environment-variable names such as `apiKeyEnv`.
- Do not scan the whole filesystem or home directory without explicit approval.
- Setup, indexing, and refresh never move, rename, edit, or delete source
  documents. The librarian itself may organize files when the user asks it to.
- Do not index system trees, app bundles, caches, credential stores,
  `node_modules`, `.git`, `dist`, `build`, `target`, `.cache`, `.autorag`, or
  `.jikji`.

## Install the CLI if needed

The CLI is `@autorag/librarian` (`autorag`). Runtime is Node.js ≥ 24 or Bun.

```bash
command -v autorag >/dev/null || bun install -g @autorag/librarian
autorag --help
```

If Bun is unavailable, `npm install -g @autorag/librarian` is acceptable.

## Inspect existing configuration

Check `--config`, `AUTORAG_CONFIG`, `$AUTORAG_HOME/config.json`, or
`~/.autorag/config.json`. Relevant fields are:

- `searchPaths`, `workspacePath`, and `memoryPath`
- `model.provider`, `model.id`, `model.api`, `model.baseUrl`, `model.apiKeyEnv`
- `bm25`, `minSync`, and `jikji`
- `limits` (retrieval, baseline-prefetch, and model-facing candidate caps)
- `datasources` and `ui`

Preserve explicit user choices and a working config unless the user asks to
replace them or health checks fail.

## Configure one search model

AutoRAG is the specialized librarian agent. One configured model plans the
search, calls retrieval and filesystem tools, reads sources, judges evidence,
and curates the answer in one loop. There are no orchestrator/explorer roles.

Prefer a model with reliable tool calling and structured output, enough context
for source excerpts, high output TPS, and low first-token latency. Use a larger
reasoning model only when difficult synthesis or domain judgment matters more
than latency.

Start from a provider/model the current runtime can actually call. A Pi-usable
ChatGPT, Claude, Gemini, or other authenticated subscription is valid; an
installed CLI or subscription that Pi cannot invoke is not. For custom
OpenAI-compatible endpoints, record the real wire API, base URL, and credential
environment-variable name.

Allowed `api` values:

- `openai-completions`
- `openai-responses`
- `anthropic-messages`
- `openai-codex-responses`
- `azure-openai-responses`

If no callable setup can be established, ask the user for the provider, model
id, API protocol, base URL when custom, and credential environment-variable
name. Do not invent provider identities or model ids.

## Propose and approve document roots

Explicit user paths win. Otherwise inspect only these likely document-dense
candidates for existence and approximate supported-file counts, then ask for
approval before indexing:

| OS | Recommended | Optional |
|---|---|---|
| macOS | `~/Documents`, `~/Downloads`, `~/Desktop` | `~/Notes`, `~/Obsidian`, user-named project docs |
| Linux | XDG Documents/Downloads/Desktop or their `~/` defaults | `~/Notes`, `~/Sync`, Nextcloud/Syncthing roots |
| Windows | Documents, Downloads, Desktop shell folders | OneDrive document roots, user-named project docs |

Supported parsed formats are `md`, `markdown`, `txt`, `text`, `pdf`, `docx`,
`pptx`, `xlsx`, `xls`, `hwp`, `hwpx`, and `eml`. OCR for `jpg`, `jpeg`, `png`, `bmp`,
and `tiff` is optional (`parserOptions.ocr.enabled`). Do not present legacy
`.doc` as a supported parsed format.

Keep the first-run set small, usually one to three roots. Present a concrete
proposal and require `yes`, a narrowed keep-list, a custom list, or `skip`
before running `refresh`.

## Initialize

```bash
autorag init \
  --search-paths "/path/to/documents,/path/to/notes" \
  --workspace "/path/to/workspace" \
  --model-provider PROVIDER \
  --model-id MODEL
```

Model resolution runs through the pi runtime. A configured `provider`/`id` is
resolved against the pi runtime catalog — the built-in providers plus any
`~/.pi/agent/models.json`, custom, or extension providers — and credentials come
from stored pi auth (`~/.pi/agent/auth.json`, API key or OAuth established with
`pi /login`), then provider environment variables. When no `model` is
configured, AutoRAG uses pi's `defaultProvider`/`defaultModel` when that
provider has configured credentials; otherwise it falls back to the
authenticated local codex runtime. So model flags may be omitted when either pi
or the local runtime already supplies the intended model. Inspect what pi can
resolve with `autorag models list` (`--available` shows only providers with
configured credentials); it never prints credential values.

For a custom endpoint, add `api`, `baseUrl`, and `apiKeyEnv` to the single
`model` object in the trusted config:

```json
{
  "searchPaths": ["/path/to/documents"],
  "model": {
    "provider": "openrouter",
    "id": "anthropic/claude-sonnet-5.5",
    "api": "openai-completions",
    "baseUrl": "https://openrouter.ai/api/v1",
    "apiKeyEnv": "OPENROUTER_API_KEY"
  }
}
```

When `provider`/`id` names a pi runtime catalog model, the catalog entry stays
the base: `baseUrl`, `api`, and any declared `reasoning`, `input`,
`contextWindow`, or `maxTokens` override only those fields, and the catalog's
reasoning, thinking, and compat settings are kept. Only an id outside the
catalog (private proxy, Ollama, LiteLLM) gets a generic text model with a 128k
context window unless those fields are declared.

Use `--force` only when intentionally replacing an existing config, and target it
with an explicit path (`--config` or `AUTORAG_CONFIG`); `--force` refuses to
replace the implicit `~/.autorag/config.json`. Legacy cwd
`autorag.config.json` is a migration source only and is never deleted by init.

### Retrieval defaults

MinSync and Jikji are enabled by default. Leave them enabled unless the
user explicitly asks otherwise. Indexing never happens while answering: a
question only reads indexes that `autorag refresh` (or `autorag watch`) built,
so run a refresh after setup and whenever documents change. A never-refreshed
workspace still answers, but without MinSync/Jikji evidence.

Refresh (never a query) auto-installs the binaries: MinSync installs a verified
GitHub release into `<workspace>/.autorag/bin` (`minSync.autoInstall` defaults
to true), and Jikji installs `jikji-cli` through cargo (`jikji.autoInstall`
defaults to true; requires the Rust toolchain). Set `"autoInstall": false` only
when managing the binary yourself. Refresh is incremental: MinSync syncs only
changed parsed mirrors, and `jikji prepare` reuses unchanged documents. Jikji
roots prepare one at a time (they share one central SQLite file that rejects
concurrent writers). A search root nested inside, or duplicating, another root
is skipped for Jikji because the outer root's index already covers it.

Jikji keeps its prepared corpus (search index, parsed text, wiki, routes) in one
central store, not inside your documents: `<OS data dir>/jikji/index.sqlite`
plus `<OS data dir>/jikji/roots/<id>/` (macOS `~/Library/Application Support`,
Linux `$XDG_DATA_HOME` or `~/.local/share`, Windows `%LOCALAPPDATA%`; override
with the `JIKJI_DATA_DIR` environment variable). The only file `refresh` adds
inside an approved document root is the `.jikji_agent_map.md` guide. Tell the
user that before the first `refresh`, and never delete or edit Jikji's central
store by hand; removing it only forces a full Jikji re-prepare on the next
refresh. Older Jikji versions wrote a `<root>/.jikji` directory instead; such a
leftover is now unused and safe for the user to delete.

Exact duplicate exclusion is enabled by default. AutoRAG invokes the external
`dupey` CLI before parsed-mirror indexing, keeps the newest filesystem copy for
each exact canonical-text hash, and excludes older copies from the mirror.
Install dupey during setup when it is missing (`command -v dupey || cargo
install dupey --locked`) and tell the user the feature is available; when installation
is impossible, refresh continues without this optimization and the user is
told duplicate exclusion is off. Set `"excludeExactDuplicates": false` to
index every copy.

AutoRAG owns no embedding model. MinSync provisions the embedder
(`Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions) in its own cache
(`~/.cache/minsync/models`); there is no local model download and no bundled
runtime. One shared embedder serves everything that embeds inside
AutoRAG: MinSync indexing/query, memory similar-question search, the
chat-crawler sidecars, the GitHub Gist vector index, and the endpoint external
CLIs are pointed at. The default needs no embedder flags and no API key. It
resolves in order:

1. A configured remote endpoint (`minSync.embedder`), used for everything; no
   daemon is started and no local model is loaded.
2. A running `minsync serve` daemon (loopback only; MinSync 0.4.11+).
   AutoRAG never starts it: run `minsync serve --idle-timeout 0` yourself, as
   a launchd/systemd service (see `docs/minsync-setup.md`), or press
   Ctrl+Shift+E in `autorag tui`. `autorag setup` only reports whether it is
   alive.
3. MinSync's in-process engine, through the optional Node-API addon
   `@nomadamas/minsync-native` (same engine and weights, so bit-for-bit
   identical vectors; not bundled). Without a daemon and without the addon,
   embedding fails with an error naming the missing package.

Override the embedder only when intentionally using a remote provider:

```bash
autorag init \
  --embedder-id "voyageai/voyage-4-lite" \
  --embedder-base-url "https://openrouter.ai/api/v1" \
  --embedder-api-key-env "OPENROUTER_API_KEY" \
  --embedder-dimension 1024 \
  --embedder-batch-size 64
```

A remote `minSync.embedder` names an `id` (`openai:<model>`, `tei:<model>`, or
any id together with a `baseUrl`) plus optional `apiKeyEnv` (the **name** of the
environment variable holding the bearer token), `dimension`, `batchSize`, and
`timeoutMs`. OpenAI-protocol endpoints POST `{baseUrl}/v1/embeddings`; a `tei:`
endpoint POSTs `{baseUrl}/embed`; `openai:` with no `baseUrl` means
`https://api.openai.com`. Only store the environment-variable name, never its
value. `dimension` and `batchSize` must be positive integers, and `dimension`
must match the endpoint (the local default is 1024).

### Jev routing and question decomposition (on by default)

Leave both enabled. They are the recommended setup: they make simple questions
fast and multi-part questions thorough.

- **Jev** (`jev`, default `{ "backend": "dazzi" }` when signed in to Dazzi
  Models, else `{ "backend": "openrouter" }`; model `jev-1.13` on Dazzi,
  `typesafe/jev-1.13` on OpenRouter) runs before the fast answer. It routes each question to
  local search, web search, a direct answer (general knowledge or small
  talk skips retrieval entirely), or the `config` branch, and decides whether
  to decompose it. On
  local search it also decides, per registered datasource, whether to search
  it before the fast answer, using each datasource's `description` and where
  similar past questions were answered (retrieval memory). When setting up a
  datasource, always write a `description` from what it actually holds:
  channels or rooms, people, topics, time range (for example `"Team Slack,
  2024-2026: #release and #on-call channels, dependabot notifications"`).
  Jev is told descriptions are short, non-exhaustive summaries, so list the
  main content and do not try to list everything. After the fast answer it
  decides whether verification is needed, so a complete, evidence-backed fast
  answer ends the run.
- **Question decomposition** (`queryDecomposition`, default model
  `dazzi/openai/gpt-6-luna` when signed in to Dazzi Models, else
  `openrouter/qwen/qwen3.7-flash`) splits a multi-part question into at most
  five search queries that run in parallel.

Both need a credential for the chosen backend: a Dazzi Models `/login` (check with
`autorag models list --provider dazzi --available`), or the user's
`OPENROUTER_API_KEY` (`test -n "$OPENROUTER_API_KEY"`, never print it). Tell the
user Jev routing is on. Without a credential, routing falls back to a single
local search and the run always verifies, so searches still work. A
`query-route-fallback` diagnostic (`autorag search --debug`) shows that state.

```json
{
  "jev": { "backend": "openrouter" },
  "queryDecomposition": { "model": { "provider": "openrouter", "id": "qwen/qwen3.7-flash" } }
}
```

`autorag init` writes these defaults into new configs. To change them:

- Jev backend: `"backend": "typesafe"` (`TYPESAFE_API_KEY`) or `"vercel"`
  (`AI_GATEWAY_API_KEY`).
- Decomposition model: any catalog `provider`/`id`, with the same fields as
  the top-level `model`.
- `"queryDecomposition": false` decomposes with the search model itself.
- `"jev": false` turns routing off entirely. Do this only when the user
  explicitly opts out, for example because questions must never leave the
  machine (Jev and decomposition send the question text to OpenRouter).

When a user asks the running agent itself to change its settings (switch the
default model, add a provider, check that a provider works), Jev's `config`
branch loads this whole skill into that turn. The agent edits only the active
config file (and `models.json` for a custom provider), verifies with
`autorag health --json` and `autorag models list --available`, and reports each
change as old → new in its final message. It never prints a
credential value and never uses `init --force`.

### Retrieval and ingest caps

`limits` bounds retrieval, baseline prefetch, and the candidate lists handed to
the model. Every field is optional — an omitted field keeps the shipped default,
so add the section only to tighten or widen a specific cap. Values must be
positive integers; unknown keys (and unknown `prefetch` keys) fail config
resolution.

| Field | Default | Controls |
|---|---|---|
| `mergedEvidenceCeiling` | 500 | `search_all_documents` / model-free merge ceiling when the model omits `topK` |
| `singleDatasourceTopK` | 50 | `search_datasource_*` merge default when the model omits `topK` |
| `minSyncTopK` | 50 | MinSync semantic default `topK` |
| `minSyncScopedQueryTopK` | 100 | MinSync fetch cap applied when a scope narrows the query |
| `toolDescriptionInstanceScopes` | 8 | Instance scopes listed in one datasource tool description |
| `prefetch.jikjiTopK` | 30 | Jikji find candidate count |
| `prefetch.minSyncTopK` | 100 | MinSync retrieve candidate count |
| `prefetch.jikjiPathLimit` | 100 | Max Jikji answer paths rendered into the baseline |
| `prefetch.sectionLimit` | 100 | Max results rendered per baseline section |

```json
{
  "limits": {
    "mergedEvidenceCeiling": 1000,
    "prefetch": { "jikjiTopK": 12, "sectionLimit": 30 }
  }
}
```

The baseline renders each prefetched result's full chunk content (not a
per-result excerpt) so the model can see where the hit came from. Bound baseline
size with `prefetch.sectionLimit` / `prefetch.minSyncTopK` /
`prefetch.jikjiPathLimit`, not with a per-result content truncation.

Ingest caps are trusted connector options under `datasources.<name>.connector`
and are never settable from model/tool arguments: `maxDocuments`,
`maxItemsPerFeed`, and `maxContentChars` (RSS); `maxDocuments`,
`maxContentChars`, `maxResultsPerQuery`, and `maxBytesPerFile` (Spotlight);
`maxDocuments`, `maxContentChars`, `maxBytesPerFile`, `concurrency`,
`bandwidthLimit`, and `dryRun` (cloud-drive/rclone). `maxContentChars` defaults
to 20000 for RSS and 100000 for Spotlight and cloud-drive.

### Refresh safety limits

`refresh` guards the parsed-mirror refresh against deleting a corpus because a
search root became unavailable. Every field is optional; unknown keys fail
config resolution.

| Field | Default | Controls |
|---|---|---|
| `maxDeleteRatio` | 0.3 | Largest share (0–1) of indexed documents one refresh may delete; a bigger deletion above 100 documents is suppressed (`mass-delete-suppressed`) unless the refresh is forced |
| `maxScannedEntries` | 2000000 | Directory entries examined per search root before `walk-truncated` |
| `maxDepth` | 64 | Deepest directory level walked below a root before `walk-truncated` |

A root that is missing, unlistable, or empty while it previously held documents
keeps its mirrors and reports `search-root-unavailable`; see
`docs/minsync-setup.md`.

## Probe and configure datasource skills (setup wizard)

Use `autorag setup` to probe the embedding backend, MinSync, and known
datasources automatically:

```bash
autorag setup --format json
```

A datasource setup UI is not shipped in this build — do not recommend it for datasource
setup. Configure datasources directly in trusted config, wizard-style:

1. Probe every datasource for setup feasibility before asking the user
   anything: the backing CLI exists (`lazykatok`, `discrawl`, `slacrawl`,
   `wacrawl`, `telecrawl`, `notcrawl`, `mailcrawl`, `rclone`, `lark-cli`) and its
   local store or archive is present. CLI-backed datasources own their own
   archive, index, and authentication, so environment credentials (such as bot
   tokens) are never required or checked for them. Non-CLI connectors
   (such as `github`) require their credential environment variable
   (`GITHUB_TOKEN`).
2. Auto-configure every datasource that probes feasible — write its trusted
   `datasources` entries without asking. For example,
   when Slack (`slacrawl`) and Discord (`discrawl`) are installed with local
   stores present, set both up automatically. Discord uses discrawl's local
   desktop wiretap archive; no Discord bot token is configured or needed.
3. Skip every datasource that probes infeasible (for example Notion or
   Telegram when their CLIs or native stores are not present) and always
   report the skipped list to the user, with what is missing for each.
4. Set up a skipped datasource only when the user explicitly asks for it:
   install or authenticate the backing CLI first, then configure it.
5. E-mail datasources (`mail-export`, `mailcrawl`) matter to most
   users — always probe them and report their status, even when they end up
   skipped.

Datasource skills belong in trusted config. Builtin
template names are `kakao`, `whatsapp`, `telegram`, `slack`, `discord`,
`clawgallery`, `notion`, `github`, `github-gist`, `cloud-drive`, `mail-export`,
`mailcrawl`, `obsidian`, `rss`, `spotlight`, `lark`, and `hister`. Config keys may be connection
aliases with `"type": "<template>"`. Unknown names are skipped with an
`unknown-datasource-skill` warning; they do not fail config resolution.
`scope` narrows a query to a sub-path as ordinary filtering. Tags are descriptive metadata only, not search filters. MCP `datasourceIds` selects configured connections before retrieval; discover their IDs with `autorag.datasources.list`.

```jsonc
{
  "datasources": {
    "github": { "connector": { "repos": ["owner/repo"], "tokenEnv": "GITHUB_TOKEN" } },
    "github-gist": { "connector": { "tokenEnv": "GITHUB_TOKEN" } },
    "google-drive": { "type": "cloud-drive", "connector": { "provider": "google-drive", "remote": "gdrive:" } },
    "archive-drive": { "type": "cloud-drive", "connector": { "remote": "archive:" } },
    "mailcrawl": { "instanceId": "personal", "connector": { "account": "personal", "mailbox": "INBOX", "binaryPath": "mailcrawl" } },
    "obsidian": { "connector": { "vaultPath": "/path/to/vault" } },
    "rss": { "connector": { "feeds": [{ "url": "https://example.com/feed.xml" }] } }
  }
}
```

Tokens are environment-variable names, not raw secrets. CLI-backed connectors
keep authentication in their external tool configuration.

Mailcrawl must be installed separately (`@nomadamas/mailcrawl@0.2.2` or newer)
and configured through its own Himalaya account. AutoRAG runs its local `sync`
and `index` lifecycle, then uses the mailcrawl CLI for BM25, semantic, or
hybrid search. Do not use 0.1.3 or earlier: a no-op sync followed by `index`
fails with `text array must be non-empty`. 0.2.2 embeds through the same
backend as the rest of AutoRAG (a remote `minSync.embedder`, else the
`minsync serve` daemon) instead of its own in-process model, and keeps vectors
in LanceDB. Switching from the old native model rebuilds the semantic index on
the next `index`. kakao needs `lazykatok` 0.3.3 or newer for the same reason.
Use mailcrawl for Gmail, IMAP, and Maildir retrieval. The former Gmail REST
datasource is removed.

### Browser history (Hister) and the shared embedding model

`hister` is the one datasource whose backing program AutoRAG does not install,
start, or bundle (it is AGPL-3.0-or-later and stays a separate program). Setting
it up is an agent task with these steps. Run each command; do not just describe it.

1. **Probe first.** Add the datasource entry, then run `autorag setup --format json`
   and read the `hister` object: `installed`, `running`, and `embedding`. Every
   problem has a `remediation` string with the exact next command.
2. **Install Hister** when `installed` is false: `brew install hister` (other
   platforms: the release binaries at https://github.com/asciimoo/hister/releases).
3. **Run it as a login item** when `running` is false: `brew services start hister`.
   Hister must stay running: its browser extension only captures while it is up,
   and a page visited while it is down is not re-sent. AutoRAG never starts it.
4. **Install the browser extension** (Chrome Web Store "Hister", or Firefox
   Add-ons). It must be done in the user's browser; tell the user if you cannot.
5. **Share the embedding model** (below).
6. **Before importing past history, install a skip list** in Hister's `rules.json`
   (login, mail, documents, social and console domains, private hosts, token-bearing
   URLs). `hister import browser history` re-downloads each URL *without* the
   browser's cookies. See docs/datasource-skills.md, "Browser history via Hister".
7. Add `"browser": { "type": "hister", "connector": { "semantic": true } }` under
   `datasources`, then `autorag refresh` (only probes Hister) and
   `autorag lite retrieve "<a question>" --scope '/browser/**' --json`.

**Shared embedding model.** Hister keeps its own vector store and embeds with
whatever its `config.yml` `semantic_search` block says. Pointing it at a different
model does not fail: vectors simply stop lining up and results get worse. So Hister
must use the same model as AutoRAG, which is the `minsync serve` daemon (model
`Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions, with Qwen3's query instruction).

```bash
autorag setup --format json            # reports hister.embedding.differs: the keys that disagree
autorag setup --sync-hister            # writes them, restarts Hister, re-embeds pages if needed
```

- `--sync-hister` rewrites only `enable`, `embedding_endpoint`, `embedding_model`,
  `dimensions`, `query_prefix`, `document_prefix` in Hister's config (a
  `config.yml.bak.<time>` is kept next to it), then runs `brew services restart hister`,
  waits for it to answer, and runs `hister reindex` when a change makes stored
  vectors stale. Without the flag setup never writes Hister's config.
- The daemon port is chosen when `minsync serve` starts. After restarting the daemon,
  run `autorag setup --sync-hister` again. It is idempotent: when nothing differs
  it changes nothing and restarts nothing.
- It refuses, and says why, when the shared backend is a remote endpoint (browsing
  text would leave the machine) or MinSync's in-process engine (Hister cannot reach
  it; run `minsync serve --idle-timeout 0`, docs/minsync-setup.md).
- If `restartRequired` or `reindexRequired` is true in the report, the step was
  not done (no Homebrew service, or it failed): run `brew services restart hister`
  and `hister reindex` yourself. Check with `hister doctor`: the line
  `server.index.embeddings` must be `ok`.
- A changed `query_prefix` alone needs a restart but no reindex. Never hand-edit the
  `query_prefix` value across two lines: Hister's YAML parser folds the newline into
  a space and the model's instruction is silently corrupted. `--sync-hister` writes
  it correctly.

## Verify and build indexes

Configuration alone is not a successful setup:

```bash
autorag setup --format json
autorag status --json
autorag health --json
autorag refresh --json
autorag search "summarize the collection" --top-k 3 --json --debug
```

- `setup` probes the embedding backend (`embedding.backend`), MinSync, and
  datasource readiness.
- `status` is model-free and path-opaque.
- `health` resolves the single model, checks credential presence, and normally
  performs one live completion probe.
- `health --skip-probes` is only for intentionally offline validation and does
  not prove live provider access.
- `refresh` syncs parsed mirrors, MinSync, Jikji, configured datasources, and
  on Windows the bundled Everything file-name index. `--method <csv>` may
  deliberately narrow it (`parsed,minsync,datasources,jikji,everything,all`).
- Use `refresh --force` for a full resync only when incremental refresh is not
  enough. Keep destructive reset/rebuild operations scoped to workspace
  `.autorag` indexes, never source documents.
- After a search, use `autorag evidence <sessionId> --json` to inspect the
  exact source chunks behind numbered results, including source, method,
  stable evidence ID, excerpt/content, chunk index, and line number.

## Connect external agents through MCP

When setting up AutoRAG for a coding agent, register the package's
`autorag-mcp` stdio executable with the same absolute `AUTORAG_CONFIG` path.
Follow `autorag-lite-setup`'s MCP registration and verification procedure:
inspect existing host registration, use an absolute executable path,
reload/reconnect, discover schemas with `tools/list`, and exercise
`autorag.status`, `autorag.datasources.list`, and a known-phrase
`autorag.search` through MCP. Restart the MCP server after config changes.
The MCP server returns model-free source chunks; the calling agent curates
them. It does not invoke the configured librarian model. The same server also
exposes `autorag.report` and `autorag.evidence` for the
curation lifecycle; the matching CLI commands remain a maintenance path.
Discover the exact schemas with MCP `tools/list`. Keep the full `autorag` skill
only when model-backed curated search is also wanted. For MCP-only setup, use
`autorag-lite-setup` instead of requiring live model health.

## Keep indexes fresh

For continuous freshness, create or verify an OS-appropriate scheduled
`autorag watch --once` job, hourly by default (every 1 hour; shorten only when
the user asks for fresher indexes). Prefer cron or launchd on macOS, cron or a
user systemd timer on Linux, and Task Scheduler on Windows. Use the same
config as search, avoid overlapping runs, and keep logs outside source trees.
Once the schedule is installed, tell the user right away that hourly freshness
is set up.

## Environment overrides

- `AUTORAG_HOME`
- `AUTORAG_CONFIG`
- `AUTORAG_SEARCH_PATHS`
- `AUTORAG_WORKSPACE`
- `AUTORAG_MEMORY_PATH`
- `AUTORAG_MODEL_PROVIDER`
- `AUTORAG_MODEL_ID`
- `AUTORAG_PERMISSION_MODE` (`auto` default, `ask`, `full`; config key `permissionMode`; with `auto` Jev checks the agent's
  file-changing calls and asks the user only when it will not approve, `ask` asks about every one, `full` never asks)

## Completion condition

Setup is complete only when the CLI is installed, roots are approved, a
non-secret single-model config is written, every datasource has been probed
and the auto-configured and skipped lists reported to the user, dupey is
installed or its absence reported, `status` is acceptable, live `health`
passes, `refresh` builds the requested indexes, one real structured search
succeeds, and any requested ongoing schedule is installed or verified with the
user told it is active.
For external-agent integration, also require successful MCP discovery and a
known-source MCP search; CLI success alone does not prove the host connection.
