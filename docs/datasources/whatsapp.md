---
title: "WhatsApp"
description: "Connect the local WhatsApp archive through wacrawl — prerequisites, chat selection, indexing, and search behavior."
---

# WhatsApp

This page shows how to connect WhatsApp through the `wacrawl` CLI, which chats to expose, and what AutoRAG indexes and searches. AutoRAG never reads WhatsApp's private databases directly.

## Prerequisites

- The upstream `wacrawl` CLI installed and available on `PATH`, or pointed at by `connector.binaryPath`.
- A local WhatsApp archive that wacrawl maintains.
- On macOS, the Full Disk Access wacrawl documents, so it can read the desktop database.

## Connect

Refresh runs `wacrawl sync`, then embeds new or edited messages into the AutoRAG sidecar through the shared embedder.

```json
{
  "datasources": {
    "whatsapp": {
      "type": "whatsapp",
      "instanceId": "default",
      "description": "Personal chats and two family groups.",
      "connector": { "databasePath": "~/.wacrawl/wacrawl.db" }
    }
  }
}
```

| Connector key | Meaning |
|---|---|
| `binaryPath` | Explicit wacrawl path; default is a bare `wacrawl` lookup. |
| `databasePath` | Archive path passed as `--db`. |
| `sourcePath` | Import source passed as `--source`. |
| `timeoutMs` / `indexTimeoutMs` | Interactive (`search`) and sync budgets. |
| `channels.ids` / `channels.names` | Chat allowlist; absent means every chat. |

The native CLI owns the archive; a manual search looks like:

```bash
wacrawl --sync never search --limit 20 "<query>"
```

## What is indexed

wacrawl owns the local-first incremental archive plus FTS5. AutoRAG keeps a read-only sidecar at `<workspace>/.autorag/datasources/<name>/<instance>/sidecar.sqlite` that stores only `message key -> vector`; message text is always read from the crawler archive at query time. Results map to opaque `/whatsapp/<instance>/chunks/<event_id>` sources. Group events and media-only messages carry no text to match or cite and are skipped. A WhatsApp archive of 16,591 messages was measured at about 224 messages/s; at 16,384 embeds per refresh it catches up in two refreshes.

## Search behavior

- One method per connection: `whatsapp-hybrid` by default (BM25 over the read-only sidecar plus semantic, fused with reciprocal rank fusion), falling back to the sidecar's BM25 and only then to the CLI-backed `whatsapp-fts`. Set `connector.sidecar` to `false` for FTS only, or `semantic: false` for BM25 only. See [Hybrid search for chat archives](./chat-hybrid-search.md).
- `channels.ids` / `channels.names` filter in the sidecar's SQL, so excluded chats are never searched, embedded, or stored.
- A time window is applied natively in the sidecar's SQL before the `topK` cut. See [Search semantics](./search-semantics.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `missing wacrawl binary` from setup | Install wacrawl or set `connector.binaryPath`. |
| Empty store | A legitimate zero-hit result; a reachable-but-empty archive reports `datasource-empty`. |
| `semantic-unavailable` warning | The embedder is unavailable; BM25 stays available and the CLI sync result is not failed. |
| `datasource-index-partial` "catching up" | The first sync is still embedding the backlog; run another refresh to finish. |
| First refresh times out | Raise `connector.indexTimeoutMs`; a timed-out sync restarts from scratch. |

Diagnostics use the datasource codes in [Search semantics](./search-semantics.md); wacrawl's stderr reaches them verbatim.
