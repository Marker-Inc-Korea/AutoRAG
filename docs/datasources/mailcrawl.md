---
title: "Mail (mailcrawl)"
description: "Connect local email through mailcrawl — Himalaya or file sources, sync and index, hybrid search, and embedding limits."
---

# Mail (mailcrawl)

This page shows how to connect Gmail, IMAP, Maildir, or static `.eml`/`.mbox` email through the `mailcrawl` CLI. AutoRAG never opens `archive.sqlite` directly; the CLI owns the archive and the index.

## Prerequisites

- `@nomadamas/mailcrawl@0.3.0` or newer (Node.js 24+). Do not use 0.1.3 or earlier: a no-op sync followed by `index` fails with `text array must be non-empty`.
- A Himalaya account configured separately, or `connector.source: "files"` with `connector.paths` to index static `.eml` files, `.mbox`/`.mbx` archives, Apple Mail bundles, or directories without Himalaya.
- Email search is the sole mail path; the former `mail-export` connector is retired.

## Connect

Refresh runs `mailcrawl sync`, then `mailcrawl index`.

```json
{
  "datasources": {
    "mailcrawl": {
      "type": "mailcrawl",
      "instanceId": "personal",
      "description": "Personal Gmail inbox, last few years.",
      "connector": {
        "binaryPath": "mailcrawl",
        "account": "personal",
        "mailbox": "INBOX"
      }
    }
  }
}
```

| Connector key | Meaning |
|---|---|
| `binaryPath` | Explicit mailcrawl path; default is a bare `mailcrawl` lookup. |
| `account` / `mailbox` | Himalaya account and mailbox; they also narrow retrieval results. |
| `source` | `files` indexes `.eml`/`.mbox` on disk; otherwise Himalaya sync. |
| `paths` | Paths used with `source: "files"`. |
| `dataDir` | Explicit mailcrawl data directory; set it only to move the native archive. |
| `pageSize` / `concurrency` / `readTimeoutMs` | IMAP window size, parallel reads, and per-message read deadline. |
| `includeCategories` / `excludeCategories` | Categories re-included or excluded from sync. |
| `rebuildIndex` | Forces a vector-store rebuild on `index`. |
| `timeoutMs` / `indexTimeoutMs` | Interactive and sync/index budgets; `indexTimeoutMs` defaults to one hour. |

## What is indexed

`mailcrawl sync` fetches envelopes and messages; `mailcrawl index` builds the search index. Semantic vectors live in a LanceDB table (`<data-dir>/semantic.lance`) with the embedder identity in `semantic.identity.json`; a mismatch rebuilds the table and a search against a different embedder fails loudly rather than returning wrong neighbours. With a shared endpoint (`inl serve` daemon or a remote `minSync.embedder`), mailcrawl embeds through it; a cold cache otherwise downloads ONNX weights on the first `index`. Results map to opaque `/mailcrawl/<instance>/chunks/<chunk-id>` sources.

```bash
mailcrawl doctor
mailcrawl status
autorag refresh --method datasources
```

## Search behavior

- Retrieval is hybrid-first: one `mailcrawl search --mode hybrid` per query, with BM25 only when hybrid cannot run (for example no usable semantic index). `account`/`mailbox` narrow results after the CLI returns.
- The native CLI exposes `bm25`, `semantic`, and `hybrid` modes: `mailcrawl search --mode hybrid --limit 20 --json "<query>"`.
- `scope` narrows results; a time window is applied over the over-fetch (`publishedAt` from the message `date`). See [Search semantics](./search-semantics.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `datasource-embedding-egress-rejected` | A non-loopback embedding URL was set; AutoRAG forwards only loopback endpoints and refuses a remote before spawning. |
| `mailcrawl semantic index unavailable` (warning) | `index` failed but sync succeeded; BM25 stays available. |
| `text array must be non-empty` | mailcrawl 0.1.3 or earlier; upgrade to 0.3.0+. |
| First `index` slow or downloading models | Cold cache building the semantic index; raise `connector.indexTimeoutMs`. |
| Stale vectors after switching embedder | Identity mismatch rebuilds on the next `index`, or force it with `mailcrawl repair --semantic`. |

Diagnostics use the datasource codes in [Search semantics](./search-semantics.md); mailcrawl's stderr reaches them verbatim.
