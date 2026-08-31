---
name: autorag-setup
description: Configure AutoRAG for first use or repair its single-agent model, approved document roots, retrieval indexes, datasource skills, and health checks without exposing credentials.
---

# AutoRAG setup

Use this skill when AutoRAG is unconfigured, model resolution fails, indexes are
missing or stale, or the user wants to change the document collection.

## Safety

- Inspect only non-secret provider/model metadata and credential availability.
- Never print, copy, migrate, compare, or persist credential values. Store only
  environment-variable names such as `apiKeyEnv`.
- Do not scan the whole filesystem or home directory without explicit approval.
- Never move, rename, edit, or delete source documents.
- Do not index system trees, app bundles, caches, credential stores,
  `node_modules`, `.git`, `dist`, `build`, `target`, `.cache`, `.autorag`, or
  `.jikji`.

## Inspect existing configuration

Check `~/.autorag/config.json`, an explicit `--config` path, or
`AUTORAG_CONFIG`. Relevant fields are:

- `searchPaths`, `workspacePath`, and `memoryPath`
- `model.provider`, `model.id`, `model.api`, `model.baseUrl`, `model.apiKeyEnv`
- `bm25`, `minSync`, and `jikji`
- `datasources` and `datasourceAccess`

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

Supported index formats include `md`, `markdown`, `txt`, `text`, `pdf`, `docx`,
`pptx`, `xlsx`, `hwpx`, and `eml`; OCR image formats are optional. Do not
present legacy `doc`, `xls`, or `hwp` as fully supported parsed formats.

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

If the authenticated local runtime already supplies the intended model, model
flags may be omitted. For a custom endpoint, add `api`, `baseUrl`, and
`apiKeyEnv` to the single `model` object in the trusted config:

```json
{
  "searchPaths": ["/path/to/documents"],
  "model": {
    "provider": "openrouter",
    "id": "anthropic/claude-sonnet-4",
    "api": "openai-completions",
    "baseUrl": "https://openrouter.ai/api/v1",
    "apiKeyEnv": "OPENROUTER_API_KEY"
  }
}
```

Use `--force` only when intentionally replacing an existing config. Legacy cwd
`autorag.config.json` is a migration source only and is never deleted by init.

### Retrieval defaults

BM25, MinSync, and Jikji are enabled by default. Leave them enabled unless the
user explicitly asks otherwise. MinSync auto-install is off by default and its
binary must already be installed or on `PATH`; Jikji can install `jikji-cli`
through cargo when enabled and allowed.

Exact duplicate exclusion is enabled by default. AutoRAG invokes the external
`dupey` CLI before parsed-mirror indexing, keeps the newest filesystem copy for
each exact canonical-text hash, and excludes older copies from the mirror. Set
`"excludeExactDuplicates": false` to index every copy. Missing dupey is
non-fatal; refresh continues without this optimization.

Optional MinSync embedder settings:

```bash
autorag init \
  --embedder-id "text-embedding-3-small" \
  --embedder-base-url "https://api.openai.com/v1" \
  --embedder-api-key-env "OPENAI_API_KEY" \
  --embedder-dimension 1536 \
  --embedder-batch-size 64
```

Only store the environment-variable name, never its value. Dimension and batch
size must be positive integers.

## Configure datasource skills when requested

Datasource skills belong in trusted config and remain default-deny. Common
entries include WhatsApp, Telegram, Slack, Discord, Notion, GitHub, Google
Drive, a generic cloud-drive/rclone source, Gmail, local mail exports,
Mailcrawl, Obsidian, RSS/news, and Spotlight.

```jsonc
{
  "datasources": {
    "github": { "connector": { "repos": ["owner/repo"], "tokenEnv": "GITHUB_TOKEN" } },
    "google-drive": { "type": "cloud-drive", "connector": { "provider": "google-drive", "remote": "gdrive:" } },
    "archive-drive": { "type": "cloud-drive", "connector": { "remote": "archive:" } },
    "gmail": { "connector": { "backend": "himalaya", "account": "gmail", "folder": "INBOX" } },
    "mailcrawl": { "instanceId": "personal", "connector": { "account": "personal", "mailbox": "INBOX", "binaryPath": "mailcrawl" } },
    "obsidian": { "connector": { "vaultPath": "/path/to/vault" } },
    "rss": { "connector": { "feeds": [{ "url": "https://example.com/feed.xml" }] } }
  },
  "datasourceAccess": {
    "allowedTags": ["github", "cloud-drive", "gmail", "mailcrawl", "obsidian", "rss"],
    "allowedScopes": ["/github/**", "/google-drive/**", "/archive-drive/**", "/gmail/**", "/mailcrawl/**", "/obsidian/**", "/rss/**"]
  }
}
```

Tokens are environment-variable names, not raw secrets. CLI-backed connectors
keep authentication in their external tool configuration. Unknown skill names
fail config resolution. `scope` and tags can narrow trusted access but cannot
grant it.

Mailcrawl must be installed separately (`@nomadamas/mailcrawl@0.1.4` or newer)
and configured through its own Himalaya account. AutoRAG runs its local `sync`
and `index` lifecycle, then uses the mailcrawl CLI for BM25, semantic, or
hybrid search. Do not use 0.1.3 or earlier: a no-op sync followed by `index`
fails with `text array must be non-empty`.

## Verify and build indexes

Configuration alone is not a successful setup:

```bash
autorag status
autorag health
autorag refresh
autorag search "summarize the collection" --top-k 3 --json
```

- `status` is model-free and path-opaque.
- `health` resolves the single model, checks credential presence, and normally
  performs one live completion probe.
- `health --skip-probes` is only for intentionally offline validation and does
  not prove live provider access.
- `refresh` syncs parsed mirrors, BM25, MinSync, Jikji, and authorized
  datasources. `--method <csv>` may deliberately narrow it.
- Use `refresh --force` for a full resync only when incremental refresh is not
  enough. Keep destructive reset/rebuild operations scoped to workspace
  `.autorag` indexes, never source documents.

## Keep indexes fresh

For continuous freshness, create or verify an OS-appropriate scheduled
`autorag watch --once` job, typically every 30 minutes. Prefer cron or launchd
on macOS, cron or a user systemd timer on Linux, and Task Scheduler on Windows.
Use the same config as search, avoid overlapping runs, and keep logs outside
source trees.

## Environment overrides

- `AUTORAG_CONFIG`
- `AUTORAG_SEARCH_PATHS`
- `AUTORAG_WORKSPACE`
- `AUTORAG_MEMORY_PATH`
- `AUTORAG_MODEL_PROVIDER`
- `AUTORAG_MODEL_ID`

## Completion condition

Setup is complete only when roots are approved, a non-secret single-model
config is written, `status` is acceptable, live `health` passes, `refresh`
builds the requested indexes, one real structured search succeeds, and any
requested ongoing schedule is installed or verified.
