---
name: autorag-setup
description: Install and configure AutoRAG, register its Lite MCP server for external agents, or repair its single search model, approved roots, indexes, datasources, and health checks without exposing credentials. Use when AutoRAG or MCP is missing, init/refresh/health fails, indexes are stale, or the user wants to add folders or datasources.
license: MIT
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
- Never move, rename, edit, or delete source documents.
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
changed parsed mirrors, and `jikji prepare` reuses unchanged documents. Roots
prepare in parallel.

Jikji stores its prepared corpus metadata in a hidden `.jikji` directory
inside each indexed source root — that is Jikji's native index layout and
intended behavior, not a misplaced artifact. Tell the user before the first
`refresh` that `<root>/.jikji` will be created inside every approved document
root (one per root, alongside the documents), and never delete or edit its
contents; removing it only forces a full Jikji re-prepare on the next refresh.

Exact duplicate exclusion is enabled by default. AutoRAG invokes the external
`dupey` CLI before parsed-mirror indexing, keeps the newest filesystem copy for
each exact canonical-text hash, and excludes older copies from the mirror.
Install dupey during setup when it is missing (`command -v dupey || cargo
install dupey --locked`) and tell the user the feature is available; when installation
is impossible, refresh continues without this optimization and the user is
told duplicate exclusion is off. Set `"excludeExactDuplicates": false` to
index every copy.

MinSync's default embedder is in-process native Qwen3 embeddings
(`native:Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions, MinSync 0.4.5+). The
default needs no embedder flags, no API key, and no external daemon (such as
Ollama) — all embeddings run in-process locally and privately. For workspaces
using the loopback llama-server gateway, prefetch the verified model with
`autorag models prefetch --profile qwen3-embedding-0.6b` (or verify with
`autorag models verify --profile qwen3-embedding-0.6b`).

The legacy Ollama/TEI adapter path (EmbeddingGemma, 768 dimensions) is
supported only for existing legacy workspaces and manual QA; do not use it as a
fresh install default.

Override the embedder only when intentionally using a different, for example
remote, provider:

```bash
autorag init \
  --embedder-id "voyageai/voyage-4-lite" \
  --embedder-base-url "https://openrouter.ai/api/v1" \
  --embedder-api-key-env "OPENROUTER_API_KEY" \
  --embedder-dimension 1024 \
  --embedder-batch-size 64
```

Only store the environment-variable name, never its value. Dimension and batch
size must be positive integers, and the dimension must match the embedder
(default Qwen3 is 1024; legacy EmbeddingGemma is 768; `voyageai/voyage-4-lite` via OpenRouter is 1024).

### Jev routing and question decomposition (on by default)

Leave both enabled. They are the recommended setup: they make simple questions
fast and multi-part questions thorough.

- **Jev** (`jev`, default `{ "backend": "openrouter" }`, model
  `typesafe/jev-1.13`) runs before the fast answer. It routes each question to
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
  `openrouter/qwen/qwen3.7-flash`) splits a multi-part question into at most
  five search queries that run in parallel.

Both use the user's `OPENROUTER_API_KEY`; confirm it is set (`test -n
"$OPENROUTER_API_KEY"`, never print it) and tell the user Jev routing is on.
Without the key, routing falls back to a single local search and the run
always verifies, so searches still work. A `query-route-fallback` diagnostic
(`autorag search --debug`) shows that state.

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
change as old → new through `emit_autorag_results`. It never prints a
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

## Probe and configure datasource skills (setup wizard)

Use `autorag setup` to probe the local runtime, model profile, and known
datasources automatically:

```bash
autorag setup --format json
```

A datasource setup UI is not shipped in this build — do not recommend it for datasource
setup. Configure datasources directly in trusted config, wizard-style:

1. Probe every datasource for setup feasibility before asking the user
   anything: the backing CLI exists (`lazykatok`, `discrawl`, `slacrawl`,
   `wacrawl`, `telecrawl`, `notcrawl`, `qmd`, `mailcrawl`, `rclone`, `lark-cli`) and its
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
`mailcrawl`, `obsidian`, `rss`, `spotlight`, and `lark`. Config keys may be connection
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

Mailcrawl must be installed separately (`@nomadamas/mailcrawl@0.2.0` or newer)
and configured through its own Himalaya account. AutoRAG runs its local `sync`
and `index` lifecycle, then uses the mailcrawl CLI for BM25, semantic, or
hybrid search. Do not use 0.1.3 or earlier: a no-op sync followed by `index`
fails with `text array must be non-empty`. 0.2.0 defaults to the in-process
native `Qwen/Qwen3-Embedding-0.6B` embedder and keeps vectors in LanceDB, so a
cold cache makes the first `index` download ONNX weights and run for minutes.
Use mailcrawl for Gmail, IMAP, and Maildir retrieval. The former Gmail REST
datasource is removed.

## Verify and build indexes

Configuration alone is not a successful setup:

```bash
autorag setup --format json
autorag status --json
autorag health --json
autorag refresh --json
autorag search "summarize the collection" --top-k 3 --json --debug
```

- `setup` probes runtime health, model profile, and datasource readiness.
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
exposes `autorag.report`, `autorag.evidence`, and `autorag.feedback` for the
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
