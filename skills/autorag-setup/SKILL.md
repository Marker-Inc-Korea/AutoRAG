---
name: autorag-setup
description: Install and configure AutoRAG, or repair its single search model, approved document roots, retrieval indexes, datasource skills, and health checks without exposing credentials. Use when autorag is missing, init/refresh/health fails, indexes are stale, or the user wants to add folders or datasources.
license: MIT
---

# AutoRAG setup

Use this skill when AutoRAG is unconfigured, the `autorag` CLI is missing, model
resolution fails, indexes are missing or stale, or the user wants to change the
document collection or datasources.

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
- `datasources`, `datasourceAccess`, and `ui`

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
user explicitly asks otherwise. MinSync auto-installs a verified GitHub release
into `<workspace>/.autorag/bin` on first use (`minSync.autoInstall` defaults to
true). Set `"autoInstall": false` only when managing the binary yourself. Jikji
auto-installs `jikji-cli` through cargo when enabled (`jikji.autoInstall`
defaults to true; requires the Rust toolchain).

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

Prefer `autorag ui --no-open` to connect datasources. It writes the same trusted
`datasources` / `datasourceAccess` fields as a hand-edited config, stores
env-var names rather than secrets, and prints a loopback URL (`127.0.0.1`).
Do not bind non-loopback hosts unless the user explicitly set `ui.allowRemote`.

Datasource skills belong in trusted config and remain default-deny. Builtin
template names are `kakao`, `whatsapp`, `telegram`, `slack`, `discord`,
`clawgallery`, `notion`, `github`, `cloud-drive`, `gmail`, `mail-export`,
`mailcrawl`, `obsidian`, `rss`, and `spotlight`. Config keys may be connection
aliases with `"type": "<template>"`. Unknown names are skipped with an
`unknown-datasource-skill` warning; they do not fail config resolution.
`scope` and tags can narrow trusted access but cannot grant it.

```jsonc
{
  "datasources": {
    "github": { "connector": { "repos": ["owner/repo"], "tokenEnv": "GITHUB_TOKEN" } },
    "google-drive": { "type": "cloud-drive", "connector": { "provider": "google-drive", "remote": "gdrive:" } },
    "archive-drive": { "type": "cloud-drive", "connector": { "remote": "archive:" } },
    "gmail": { "connector": { "tokenEnv": "GMAIL_ACCESS_TOKEN", "labelIds": ["INBOX"] } },
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
keep authentication in their external tool configuration.

Mailcrawl must be installed separately (`@nomadamas/mailcrawl@0.1.4` or newer)
and configured through its own Himalaya account. AutoRAG runs its local `sync`
and `index` lifecycle, then uses the mailcrawl CLI for BM25, semantic, or
hybrid search. Do not use 0.1.3 or earlier: a no-op sync followed by `index`
fails with `text array must be non-empty`.
Use mailcrawl for Himalaya-backed IMAP/Maildir retrieval. The legacy
`gmail` connector option `backend: "himalaya"` is no longer registered; migrate
that configuration to an explicit `mailcrawl` datasource.

## Verify and build indexes

Configuration alone is not a successful setup:

```bash
autorag status --json
autorag health --json
autorag refresh --json
autorag search "summarize the collection" --top-k 3 --json --debug
```

- `status` is model-free and path-opaque.
- `health` resolves the single model, checks credential presence, and normally
  performs one live completion probe.
- `health --skip-probes` is only for intentionally offline validation and does
  not prove live provider access.
- `refresh` syncs parsed mirrors, BM25, MinSync, Jikji, and authorized
  datasources. `--method <csv>` may deliberately narrow it
  (`parsed,bm25,minsync,datasources,jikji,all`).
- Use `refresh --force` for a full resync only when incremental refresh is not
  enough. Keep destructive reset/rebuild operations scoped to workspace
  `.autorag` indexes, never source documents.

## Keep indexes fresh

For continuous freshness, create or verify an OS-appropriate scheduled
`autorag watch --once` job, typically every 15–30 minutes. Prefer cron or
launchd on macOS, cron or a user systemd timer on Linux, and Task Scheduler on
Windows. Use the same config as search, avoid overlapping runs, and keep logs
outside source trees.

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
non-secret single-model config is written, `status` is acceptable, live `health`
passes, `refresh` builds the requested indexes, one real structured search
succeeds, and any requested ongoing schedule is installed or verified.
