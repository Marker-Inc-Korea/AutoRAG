---
title: "Telegram"
description: "Connect the Telegram desktop archive through telecrawl — prerequisites, channel selection, indexing, and search behavior."
---

# Telegram

This page shows how to connect Telegram through the `telecrawl` CLI, which chats to expose, and what AutoRAG indexes and searches. AutoRAG never reads Telegram's private databases directly.

## Prerequisites

- The upstream `telecrawl` CLI installed and available on `PATH`, or pointed at by `connector.binaryPath`.
- A local Telegram Desktop archive that telecrawl can import.
- On macOS, the Full Disk Access telecrawl documents, so it can read the desktop cache.

## Connect

Refresh runs telecrawl's `import` (the sync step), then embeds new or edited messages into the AutoRAG sidecar through the shared embedder.

```json
{
  "datasources": {
    "telegram": {
      "type": "telegram",
      "instanceId": "default",
      "description": "Personal Telegram chats and a few groups.",
      "connector": { "databasePath": "~/.telecrawl/telecrawl.db" }
    }
  }
}
```

| Connector key | Meaning |
|---|---|
| `binaryPath` | Explicit telecrawl path; default is a bare `telecrawl` lookup. |
| `databasePath` | Archive path passed as `--db`. |
| `sourcePath` | Import source passed as `--source`. |
| `timeoutMs` / `indexTimeoutMs` | Interactive (`search`) and sync (`import`) budgets. |
| `channels.ids` / `channels.names` | Chat allowlist; absent means every chat. |

The native CLI owns the archive; a manual search looks like:

```bash
telecrawl search --limit 20 "<query>"
```

## What is indexed

telecrawl owns the desktop archive plus FTS5. AutoRAG keeps a read-only sidecar at `<workspace>/.autorag/datasources/<name>/<instance>/sidecar.sqlite` that stores only `message key -> vector`; message text is always read from the crawler archive at query time. Results map to opaque `/telegram/<instance>/chunks/<event_id>` sources. Media-only messages carry no text to match or cite and are skipped.

## Search behavior

- One method per connection: `telegram-hybrid` by default (BM25 over the read-only sidecar plus semantic, fused with reciprocal rank fusion), falling back to the sidecar's BM25 and only then to the CLI-backed `telegram-fts`. Set `connector.sidecar` to `false` for FTS only, or `semantic: false` for BM25 only. See [Hybrid search for chat archives](./chat-hybrid-search.md).
- `channels.ids` / `channels.names` filter in the sidecar's SQL, so excluded chats are never searched, embedded, or stored.
- A time window is applied natively in the sidecar's SQL before the `topK` cut. See [Search semantics](./search-semantics.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `missing telecrawl binary` from setup | Install telecrawl or set `connector.binaryPath`. |
| Empty store | A legitimate zero-hit result; telecrawl prints JSON `null` for no hits, not `[]`. |
| `semantic-unavailable` warning | The embedder is unavailable; BM25 stays available and the CLI sync result is not failed. |
| `datasource-index-partial` "catching up" | The first sync is still embedding the backlog; run another refresh to finish. |
| First refresh times out | Raise `connector.indexTimeoutMs`; a timed-out import restarts from scratch. |

Diagnostics use the datasource codes in [Search semantics](./search-semantics.md); telecrawl's stderr reaches them verbatim.
