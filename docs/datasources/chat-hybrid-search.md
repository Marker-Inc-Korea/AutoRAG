---
title: "Hybrid search for chat archives"
description: "How the read-only sidecar adds BM25, semantic, and hybrid retrieval to the chat crawlers, and how Discord ranks lexically."
---

# Hybrid search for chat archives

This page explains the read-only sidecar that gives the chat crawlers ranked lexical, semantic, and hybrid retrieval, and how Discord reuses its native FTS5 index for BM25. It applies to `slack`, `telegram`, `whatsapp`, and `kakao`; `discord` has a lexical-only variant.

## Why the sidecar exists

The crawler CLIs expose only a plain FTS5 query, and `slacrawl` returns the newest matches and ANDs every term, so a natural-language question often matches nothing. The sidecar adds BM25 ranking, semantic search, and hybrid fusion on top, without writing to the crawler's archive.

## Configuring it

```json
{ "datasources": { "slack-work": { "type": "slack", "connector": { "workspace": "T0123456789" } } } }
```

Tune or opt out per datasource with `connector.sidecar`:

| `connector.sidecar` | Meaning |
|---|---|
| absent / `true` / `{}` | One hybrid method per connection; BM25 then the CLI search only if it throws (default). |
| `false` or `{ "enabled": false }` | CLI-backed FTS only. |
| `semantic: false` | BM25 only; no embedding calls and no `<name>-hybrid` method (the CLI search is the fallback). |
| `crawlerDbPath` | Archive to read. Otherwise: `connector.databasePath`; for `slack`, the `db_path` in the slacrawl config that `connector.configPath` names (relative paths resolve against that file); otherwise the crawler default (`~/.slacrawl/slacrawl.db`, `~/.telecrawl/telecrawl.db`, `~/.wacrawl/wacrawl.db`). |
| `batchSize`, `maxEmbedPerSync` | Embedding request size (256) and the most messages one refresh embeds (16,384). |

## How it works

- The crawler archive is opened read-only (`ATTACH ... mode=ro`); AutoRAG never writes to it, so a crawler that migrates, purges, vacuums, or rebuilds its file cannot lose sidecar state. SQLite may leave `-wal`/`-shm` files beside a WAL-mode archive after a read-only open; the archive file itself is unchanged.
- The sidecar owns its own file, `<workspace>/.autorag/datasources/<name>/<instance>/sidecar.sqlite`. It stores only `message key -> vector`; message text is always read from the crawler archive at query time.
- It is created lazily: building a skill creates no file, directory, or handle. Nothing is written until the crawler archive exists and a search or refresh runs. Without an AutoRAG workspace there is nowhere to keep it, so the datasource behaves as before.
- One method per connection, `<name>-hybrid` (reciprocal rank fusion, k=60): it runs the sidecar's hybrid search and never also runs the CLI or the sidecar's BM25 and semantic components separately. Only if the hybrid search throws does the sidecar's BM25 answer (`<name>-sidecar-bm25`), and only if that throws the CLI-backed `<name>-fts`. A search that finds nothing is an answer and starts no fallback. A message found by the CLI and by the sidecar carries the same id (`event_id` for telegram and whatsapp), so a fallback keeps stable evidence ids.
- `channels.ids` / `channels.names` are applied in the sidecar's SQL: excluded chats are never searched, embedded, or stored as vectors, and vectors of a chat removed from the allowlist are pruned on the next refresh.
- `autorag refresh --method datasources` runs the CLI sync first, then embeds new or edited messages with the shared embedder, prunes vectors of purged messages, and re-embeds everything if the embedding identity changes.
- Embedding is bounded per refresh (16,384 messages) so the first sync of a large archive does not hold the whole refresh. While a backlog remains, the datasource reports an info `datasource-index-partial` diagnostic (`semantic index catching up`), and hybrid search uses BM25 for the messages not yet embedded.
- Each message's embedding input is its first 8192 characters (BM25 still indexes all of it). A message the embedder rejects is skipped with a warning and retried only after it is edited.
- An embedding failure is a `semantic-unavailable` warning; BM25 stays available and the CLI sync result is never turned into a failure. Two AutoRAG processes sharing the sidecar file: a writer waits up to 5 s for the other's lock.
- Semantic search is a linear scan over stored vectors (about 280 ms at 16.6k messages). Fine for personal archives; measure before pointing it at a multi-million-message archive.
- Message text is sent only to the shared embedder. See [Embedding backends](../indexing/embedding-backends.md).

`notion` (notcrawl) keeps its page/comment FTS and has no sidecar.

## Discord BM25 ranking

When an AutoRAG workspace is available, `discord` ranks lexical matches in discrawl's existing FTS5 index by BM25 relevance, so an older relevant message can outrank newer mentions before the result limit. It builds no second vector index; discrawl keeps its own embeddings.

- With the sidecar enabled, the skill registers one method matching `connector.defaultMode`: `discord-hybrid` by default, `discord-fts` for lexical ranking, or `discord-semantic`. This prevents the outer merger from treating overlapping native FTS, native hybrid, and BM25 results as independent votes. Set `connector.sidecar` to `false` to restore the separate native `discord-hybrid`, `discord-fts`, and `discord-semantic` methods.
- Native FTS-only hits supplement the lexical candidate list after BM25 hits; duplicate evidence contributes only once. Hybrid retrieval fuses the one lexical ranking with native semantic results using reciprocal rank fusion.
- BM25 ranks the base `message_fts` index and any tokenizer-specific indexes discrawl built (`message_fts_ko`, `message_fts_ja`, `message_fts_zh`, `message_fts_ar`), fusing their rankings. Tables that are not configured are ignored.
- The archive path comes from the read-only `discrawl --json status` command; an explicit `connector.sidecar.crawlerDbPath` or `connector.databasePath` takes precedence for the sidecar only. The native CLI still uses its own config.
- Soft-deleted and empty normalized messages are excluded from BM25, and `channels.ids` / `channels.names` filter its results in SQL. Native candidates are checked against the configured guild and channel allowlists before combination.
- If a backend fails and no candidates remain, retrieval surfaces the failure instead of reporting a successful empty search.

## Relationship to time windows

The sidecar's SQL applies a window before its own `topK` cut, so an in-window message that ranks low is still found. Discord's native methods and the CLI-backed `*-fts` methods apply the window over the over-fetch instead. See [Search semantics](./search-semantics.md) for the per-method table.
