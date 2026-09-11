---
name: autorag-lite-setup
description: Initialize, index, refresh, and maintain the model-free AutoRAG Lite lifecycle (config, roots, datasources, refresh, watch, status, index reset/rebuild) without configuring any model. Use when autorag lite init/refresh/status is needed, indexes are missing or stale, or the user wants local document indexing without a search model.
license: MIT
---

# AutoRAG Lite setup

Use this skill when the user wants AutoRAG's indexing lifecycle without a
model: initialize a config, build and refresh indexes, watch roots, and inspect
index health. Everything here is model-free. Use `autorag-setup` instead when a
search model must be configured or repaired. Use `autorag-lite-search` for
retrieval, reports, evidence, and feedback.

## Safety

- Inspect only non-secret config metadata: `searchPaths`, `workspacePath`,
  `memoryPath`, `minSync`, `jikji`, `datasources`, `datasourceAccess`.
- Never print, copy, migrate, compare, or persist credential values. Store only
  environment-variable names such as `tokenEnv` or `apiKeyEnv`.
- Never move, rename, edit, or delete source documents. Lite commands write
  indexes only under the configured workspace `.autorag/` directory and
  Jikji's per-source `.jikji/` caches.
- Do not index system trees, app bundles, caches, credential stores,
  `node_modules`, `.git`, `dist`, `build`, `target`, `.cache`, `.autorag`, or
  `.jikji`.

## Install the CLI if needed

The CLI is `@autorag/librarian` (`autorag`). Runtime is Node.js >= 24 or Bun.

```bash
command -v autorag >/dev/null || bun install -g @autorag/librarian
autorag lite --help
```

If Bun is unavailable, `npm install -g @autorag/librarian` is acceptable.

## Initialize a model-free config

```bash
autorag lite init \
  --search-paths "/path/to/documents,/path/to/notes" \
  --workspace "/path/to/workspace" \
  --memory-path "/path/to/memory.json"
```

`lite init` writes the configured model-free lifecycle config. Use `--force`
only when intentionally replacing an existing config. Explicit user paths win;
otherwise propose one to three document-dense roots and get approval before
indexing. Supported parsed formats are `md`, `markdown`, `txt`, `text`, `pdf`,
`docx`, `pptx`, `xlsx`, `xls`, `hwp`, `hwpx`, and `eml`. Legacy `.doc` is not
a supported parsed format.

Config resolution follows the usual order: `--config`, `AUTORAG_CONFIG`,
`$AUTORAG_HOME/config.json`, or `~/.autorag/config.json`. Environment overrides
include `AUTORAG_HOME`, `AUTORAG_CONFIG`, `AUTORAG_SEARCH_PATHS`,
`AUTORAG_WORKSPACE`, and `AUTORAG_MEMORY_PATH`.

## Configure datasources when requested

Datasource skills belong in trusted config and remain default-deny. Prefer
`autorag lite ui --no-open` to connect datasources; it writes the same trusted
`datasources` / `datasourceAccess` fields as a hand-edited config, stores
env-var names rather than secrets, and binds a loopback address by default.
Do not bind non-loopback hosts unless the user explicitly allowed remote
access.

Config keys may be builtin template names (`kakao`, `whatsapp`, `telegram`,
`slack`, `discord`, `clawgallery`, `notion`, `github`, `cloud-drive`, `gmail`,
`mail-export`, `mailcrawl`, `obsidian`, `rss`, `spotlight`) or connection
aliases with `"type": "<template>"`. Unknown names are skipped with an
`unknown-datasource-skill` warning; they do not fail config resolution.
`datasourceAccess.allowedTags` and `allowedScopes` narrow trusted access and
can never grant it.

## Build and refresh indexes

```bash
autorag lite refresh --json
autorag lite refresh --method parsed,minsync --json
autorag lite refresh --full --json
autorag lite refresh --force --json
```

- A plain `refresh` is incremental: it syncs parsed mirrors, MinSync, Jikji,
  and configured datasources against what changed.
- `--full` and `--force` both request a full resync. Use them only when
  incremental refresh is not enough, such as after config changes to roots or
  index settings.
- `--method <csv>` deliberately narrows the refresh. Valid values are
  `parsed`, `minsync`, `datasources`, `jikji`, and `all`. Omit the flag to run
  all methods. Unknown values are rejected.
- MinSync and Jikji auto-install on first use by default. If they are missing
  or broken, run a full refresh or return to setup rather than silently
  degrading to lexical-only search.
- Exact duplicate exclusion during refresh is enabled by default via the
  external `dupey` CLI. Missing dupey is non-fatal; refresh continues without
  it. Set `"excludeExactDuplicates": false` to index every copy.

Retrieval requires a completed refresh. `autorag lite retrieve` before any
refresh exits with code 2 and an `index-not-ready` diagnostic; a successful
refresh is recorded even when the corpus is empty or only a non-parsed method
was selected. Always refresh first, and refresh again when roots change.
Jikji is a discovery/indexing preparer, not a lite retrieval method.

## Watch and scheduled freshness

```bash
autorag lite watch --once --json
autorag lite watch
```

Prefer non-daemon `autorag lite watch --once` from cron, launchd, a systemd
user timer, or Task Scheduler, typically every 15 to 30 minutes. Use the same
config as retrieval, avoid overlapping runs, and keep logs outside source
trees. `--immediate` triggers a first refresh on start and `--debounce-ms N`
tunes change coalescing.

## Status and index maintenance

```bash
autorag lite status --json
autorag lite health --json
autorag lite index rebuild --yes --json
autorag lite index reset --method parsed --yes --json
autorag lite duplicates --json
```

- `status` shows path-opaque corpus freshness and index health. `health` is an
  alias of `status`; neither resolves a model.
- `index reset` and `index rebuild` remove or rebuild only workspace
  `.autorag` indexes selected by `--method`. They never target source
  documents.
- `duplicates` scans duplicate document families read-only and never deletes
  or moves files.

## Unavailable components and failure handling

Missing optional components degrade gracefully: refresh and retrieval continue
with path-opaque diagnostics such as `minsync-unavailable` or
`retrieval-method-failed` instead of failing the whole run. Exit codes are 0
on success, 2 for config or usage errors, and 1 for runtime errors. When a
component stays unavailable after a full refresh, return to setup rather than
accepting silently degraded search.

## Completion condition

Setup is complete only when the CLI is installed, roots are approved, a
non-secret model-free config is written, `refresh` has built the requested
indexes, `status` reports healthy indexes, and any requested watch schedule is
installed or verified.
