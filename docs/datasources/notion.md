---
title: "Notion"
description: "Index a Notion workspace through the notcrawl CLI and search archived pages, databases, and block text."
---

# Notion

The `notion` datasource searches an archive of Notion pages, databases, titles, properties, and block text through the external `notcrawl` CLI. Use it for questions about Notion docs and wikis.

## Prerequisites

- The `notcrawl` CLI installed and its credentials configured in notcrawl itself. AutoRAG never reads the Notion API directly and never opens notcrawl's database.
- A notcrawl archive present. `autorag setup` reports the connection as `skipped: missing notcrawl binary` when the CLI is absent and `skipped: native store is not present` when no archive exists.

## Connect

Add a `datasources` entry to `~/.autorag/config.json`:

```json
{
  "datasources": {
    "notion": {
      "connector": { "configPath": "~/.notcrawl/config.toml" }
    }
  }
}
```

All connector options are optional; notcrawl falls back to its own default config and store.

| Connector option | Purpose |
| --- | --- |
| `configPath` | Passed to notcrawl as `--config`; points at notcrawl's own config. |
| `binaryPath` | The `notcrawl` executable, when it is not on `PATH`. |
| `databasePath` | notcrawl's archive path, when it is not in its default location. |
| `sourcePath` / `syncSource` | notcrawl source selections, passed through unchanged. |
| `timeoutMs` | Interactive `search` timeout (default 60s). |
| `indexTimeoutMs` | `sync` timeout (default one hour); see [Connector sync/index timeouts](connecting.md#connector-sync-index-timeouts). |

Then index it:

```bash
autorag refresh --method datasources
```

## What is indexed

- notcrawl owns the archive; refresh runs `notcrawl [--config <path>] sync` and maps the reported page count.
- Chunk identity is the opaque source `/notion/<instance>/chunks/<chunk-id>`. Source descriptions list hierarchy paths such as `/notion/<instance>/pages/<page-id>`.
- Each hit carries `title`, an optional `parentTitle`, a `url`, and `lastEditedTime`; the time is notcrawl's `last_edited_time`.

## Search behavior

- Retrieval is lexical FTS5 (`notion-fts`), scoped to the instance path (see [Scope narrowing](search-semantics.md#scope-narrowing)). There is no semantic method for Notion.
- A `since`/`until` window is applied after the search over the hit's `last_edited_time` (see [Time windows](search-semantics.md#time-windows)).
- The agent may also call notcrawl directly through `bash`, for example `notcrawl search "<query>" --limit 20`.

## Migrating

Notion is reached through `notcrawl` now. The datasource no longer speaks the Notion API itself, so any Notion API token you previously configured is unused; set up notcrawl's own authentication instead.

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| `notcrawl binary-missing`, `notcrawl spawn-error`, or `notcrawl timeout` | `datasource-unavailable` | Install notcrawl, or point `connector.binaryPath` at it. |
| A sync that timed out | `datasource-unavailable` with the connector-timeout hint | Raise `connector.indexTimeoutMs`; a timed-out sync restarts from scratch next refresh. |
| Non-zero exit or unparseable output | `datasource-index-failed` | Run `notcrawl doctor` and `notcrawl status`, then `notcrawl sync --source desktop`. |
| `autorag setup` reports `skipped: native store is not present` | — | Run notcrawl's sync once so an archive exists, or set `connector.databasePath`. |
