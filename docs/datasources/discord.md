---
title: "Discord"
description: "Connect the Discord archive through discrawl — token-free wiretap import, attachment saving, BM25 ranking, and channel selection."
---

# Discord

This page shows how to connect Discord through the `discrawl` CLI, how attachments are saved before their signed links expire, and how lexical ranking works. AutoRAG never calls the Discord API itself and never reads the Discord Desktop cache directly.

## Prerequisites

- The upstream `discrawl` CLI installed and available on `PATH`, or pointed at by `connector.binaryPath`.
- A local Discord archive (imported from the Discord Desktop cache). No Discord bot or user token is configured, required, or forwarded: AutoRAG drives discrawl's token-free `wiretap` path only.

## Connect

Refresh runs `discrawl doctor`, then the incremental cursor-based `wiretap` import, then `discrawl embed` to drain the embedding queue, so a poll costs work proportional to what changed.

```json
{
  "datasources": {
    "discord": {
      "type": "discord",
      "instanceId": "default",
      "description": "Team server: release engineering and general channels.",
      "connector": {
        "defaultMode": "hybrid",
        "downloadAttachments": true
      }
    }
  }
}
```

| Connector key | Meaning |
|---|---|
| `binaryPath` | Explicit discrawl path; default is a bare `discrawl` lookup. |
| `configPath` | Explicit operator-owned config passed as `--config`; AutoRAG never rewrites it. |
| `defaultMode` | `hybrid` (default), `fts`, or `semantic`. |
| `downloadAttachments` | `false` disables automatic attachment downloads on refresh. |
| `embedLimit` | Max messages embedded per index pass; undefined drains the whole backlog. |
| `guildId` | Restrict the connection to one guild. |
| `timeoutMs` / `indexTimeoutMs` | Interactive and sync budgets. |
| `channels.ids` / `channels.names` | Channel allowlist applied in retrieval and the sidecar. |

Without an explicit `configPath`, AutoRAG can write a managed `[search.embeddings]` section (preceded by the marker `# AutoRAG managed discrawl embeddings v1`) that points discrawl at the shared embedder, when that endpoint speaks the OpenAI protocol. An explicit `configPath` is authoritative and is never rewritten.

## What is indexed

discrawl owns the SQLite archive, its FTS5 index, and the message vectors. Messages map to opaque `/discord/<instance>/chunks/<message-id>` sources; Discord message ids are stable snowflakes. Refresh reports a `datasource-index-partial` diagnostic with attachment counts when it saved files, and an info `datasource-empty` diagnostic when the sync found no new messages. An English-only embedding model is reported as a warning because it degrades non-English semantic search to noise.

### Attachments

Discord signs every attachment link for 24 hours; after that the file can no longer be downloaded. Each refresh of a Discord connection saves attachments by default, right after the import. Files go to `~/.autorag/attachments/discord/<connection>/<server>/<channel>/<original name>` on every OS (under `$AUTORAG_HOME` when set). discrawl does the downloading into a private `.discrawl/` config inside the folder; AutoRAG never talks to Discord and uses no token.

```bash
autorag attachments status --connection discord
autorag attachments download --connection discord
```

`download-status.json` at the top of the folder lists every attachment:

| State | Meaning |
|---|---|
| `downloaded` | The saved file still has the exact bytes Discord served (SHA-256). |
| `not-downloaded` | Not saved yet. |
| `missing` | The saved file was deleted, moved, or renamed. |
| `changed` | The saved file was edited; it stays as your file. |
| `failed` | The last attempt failed; `error` says why. |
| `skipped` | discrawl refused it (for example `too_large`, over 100 MB); not retried. |

`missing` and `changed` files are downloaded again while their link works; an edited file is never overwritten, so the original lands beside it as `name (2).ext`. Nothing in the folder is deleted. Attachments are saved only, not indexed for search.

## Search behavior

- With the BM25 sidecar enabled (the default when an AutoRAG workspace exists), the connection registers one method matching `connector.defaultMode`: `discord-hybrid` by default, `discord-fts` for lexical ranking, or `discord-semantic`. This prevents overlapping native lexicals from voting twice in the outer merger. With the sidecar off, the single native method for `defaultMode` is registered.
- BM25 ranks the base `message_fts` index and any tokenizer-specific language indexes discrawl built (`message_fts_ko`, `message_fts_ja`, `message_fts_zh`, `message_fts_ar`), fusing their rankings, so an inflected Korean, Japanese, Chinese, or Arabic message is still ranked lexically. Native FTS supplements the lexical candidate list after BM25 hits.
- The FTS index merges words across line breaks into one token, so lexical-only search can miss a term right after a newline; semantic recall covers that gap, which is why hybrid is the default.
- `scope` narrows results and the channel allowlist filters in SQL; a time window is applied over the over-fetch. See [Search semantics](./search-semantics.md) and [Hybrid search for chat archives](./chat-hybrid-search.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `datasource-permission-denied` | discrawl rejected a Discord user token; AutoRAG uses the token-free wiretap path. |
| `datasource-unavailable` | discrawl missing, or `doctor` reports the archive database is not ready. |
| `datasource-index-failed` with "embeddings are not configured" | Semantic and hybrid retrieval fall back to FTS only; configure discrawl embeddings. |
| Non-English semantic search unreliable | The embedding model is English-only; the warning names it and the preferred model. |
| For a "latest messages" question, add a filter | `discrawl messages` needs at least one filter (for example `--since`, `--hours`, or `--last`). |

Diagnostics use the datasource codes in [Search semantics](./search-semantics.md); discrawl's stderr reaches them verbatim.
