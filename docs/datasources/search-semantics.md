---
title: "Search semantics"
description: "How datasource search narrows by time window and scope, names results, exposes diagnostics, and how the old access controls were removed."
---

# Search semantics

This page explains how a datasource query is narrowed, how hits are named, and how failures are reported. Use it to write `since`/`until`, `scope`, and to read diagnostics.

## Time windows

`search_all_documents`, every `search_datasource_<id>`, MCP `autorag.search` and `autorag.search_datasource_<id>`, and `autorag lite retrieve --since`/`--until` accept a time window. A window only narrows: it composes with `scope` (a result must match both) and never widens what a datasource's trusted config reaches, and it cannot change connector configuration.

- `since` is inclusive, `until` is exclusive: `[since, until)`. "All of September" is `since 2026-09-01`, `until 2026-10-01`.
- Values are ISO-8601. A date-only value, or a datetime without an offset, is read as local midnight / local time in the host's time zone; `Z` and `±HH:MM` are honored exactly. Calendar-invalid values (`2026-02-31`) are rejected, not rolled over.
- A malformed value or an empty window (`since >= until`) is returned verbatim as the tool error (MCP error code `invalid-time-window`; `autorag lite retrieve` exits 2). Nothing is searched.

The agent's system prompt carries the current date and time zone, and the values the model chose are visible in the run's tool calls.

**How a result is dated.** `metadata.publishedAt` (epoch ms, the document's own timestamp) first; otherwise `metadata.sourceMtimeMs`, the local source file's last-modified time from the mirror index. Local mtime is "last modified", not "authored".

```bash
autorag lite retrieve "quarterly planning" --since 2026-09-01 --until 2026-10-01
```

No store filters by time on its own, so the pipeline over-fetches when a window is set — the same shape as the `scope` over-fetch (`max(5 × topK, topK + 20)`, bounded at 300) — and post-filters each method's candidates. Stores that can filter natively (a connector chunk store, the chat sidecars) do so before their own `topK` cut, so an in-window item that ranks low is still found.

**A method that cannot date a result excludes it.** Such results are dropped, never passed through as in range, and the method is reported in `unsearched` with a `time-window-untimed` diagnostic whose reason is verbatim.

**An empty range is explicit.** Every windowed search returns a `window` report: `{ since, until, examined, matched, untimed, ceilingHit, fetchDepth }`. The agent tools print "Nothing in this time range …" with those counts. When `ceilingHit` is true (some method returned as many candidates as it was asked for), a `time-window-ceiling` diagnostic says a thin or empty list is not proof that nothing exists in the range.

### Per-method support

| Surface | Time source | Window applied |
| --- | --- | --- |
| Local files (`minsync`, `hybrid`) | `sourceMtimeMs` (last modified) | post-filter over over-fetch |
| Slack, Telegram, WhatsApp, Discord, KakaoTalk sidecar (`*-sidecar-bm25`, `*-hybrid`) | `publishedAt` from the crawler's message time | native SQL, before `topK` |
| RSS, GitHub, GitHub Gist lexical (connector chunk store) | `publishedAt` | native, before `topK`; chunks with no date rank last and are excluded and reported |
| GitHub Gist semantic | `publishedAt` | post-filter over over-fetch |
| Slack / Telegram / WhatsApp / Notion CLI-backed `*-fts` | `publishedAt` from the hit timestamp | post-filter over over-fetch |
| Discord (`discrawl` methods) | `publishedAt` from `timestamp` | post-filter over over-fetch |
| KakaoTalk (`lazykatok` methods) | `publishedAt` from `startedAt` of the chat chunk | post-filter over over-fetch |
| Mailcrawl | `publishedAt` from the message `date` | post-filter over over-fetch |
| Wombat | `publishedAt` from the session's `startedAt` | post-filter over over-fetch |
| Hister (browser history) | `publishedAt` from the page's last visit | post-filter over over-fetch |
| Lark / Feishu | `publishedAt` from `create_time` / `update_time` | post-filter over over-fetch |
| Obsidian | `sourceMtimeMs` of the staged note | post-filter over over-fetch |
| ClawGallery | none (no timestamp in results) | unsupported: every result is excluded and the method is reported |

A hit whose store reported no usable time falls into the same rule as an unsupported method: excluded and reported, per result count. The `mail-export` and `cloud-drive` connectors no longer exist; a static mailbox is covered by the mailcrawl rows, and a mounted cloud folder is covered by the local-files row.

Out of scope: a query-less "newest N" listing, date flags on the external MinSync binary, recency-weighted ranking, and file-name search.

## Scope narrowing

`scope` is an ordinary per-query narrowing filter for datasource methods that advertise the `scoped` capability. A result is kept only when its source matches the requested scope; a scope can never widen what a connection reaches. Glob and leading-slash normalization behave like the local retrieval scope matcher.

Datasource sources always reject a `#` fragment. Datasources without the `scoped` capability (for example KakaoTalk's chat-identity results) own any narrower filtering themselves. Folder filtering for vault-backed datasources is the ordinary scope: a scope that already contains `*` is matched literally as written, so a nested-folder glob needs an explicit trailing `/**`.

## Progressive disclosure

Each configured connection gets its own model-visible `search_datasource_<id>` tool whose schema is exactly `{ query, topK?, scope?, since?, until? }`. There is no datasource fan-out tool: a question that spans every datasource uses `search_all_documents`, which registers every configured connection's retrieval methods alongside the local ones. The model can also load a connection's full instructions with `load_datasource_skill`.

Over MCP, `autorag.search` takes `datasourceIds` (and `methods`/`local`) to narrow execution before any backend is invoked; every registered fixed tool is listed by `autorag.datasources.list`. When configured, each integrated datasource also registers `autorag.search_datasource_<id>`, which searches only that connection.

Every search-tool call counts toward the per-search tool budget. Memory is written only from Jev-judged evidence cited in the final answer, never from how a single search call turned out.

## Result identity

Datasource results carry a slash-hierarchical opaque source such as `/kakao/<instance>/chunks/<chunk-id>` or `/discord/<instance>/chunks/<message-id>`. These are not OS paths: never pass one to `cd`, `cat`, or another filesystem tool, and never send one to `web_fetch`. Search them through the connection's dedicated tool or its native CLI. Provenance references the datasource ids of configured connections and is not an authorization decision.

Local-file sources, by contrast, are absolute filesystem paths and may be read with `bash` after verifying the returned path.

## Diagnostics

Datasource operations surface diagnostics with datasource codes. The retrieval pipeline collapses them onto the generic `retrieval-method-failed` code and preserves the original code in the message as a `[datasource:<code>]` prefix; the source is the instance or `datasource`.

| Code | Meaning |
|---|---|
| `datasource-unavailable` | The service or binary is unreachable. |
| `datasource-cli-error` | The external CLI failed. |
| `datasource-empty` | The source is reachable but holds nothing. |
| `datasource-rate-limited` | The upstream returned a rate limit. |
| `datasource-auth-error` | Credentials are wrong or missing. |
| `datasource-permission-denied` | The identity lacks access to the resource. |
| `datasource-index-failed` | Indexing failed. |
| `datasource-index-partial` | Indexing succeeded but is incomplete (for example a semantic backlog still draining); not a failure. |
| `datasource-embedding-egress-rejected` | Remote embedding egress was refused before the CLI ran. |
| `datasource-remote-embedding-rejected` | A CLI refused a remote embedding endpoint. |

A retrieval method that cannot answer throws a `DatasourceCliError` instead of returning an empty set, so the pipeline reports the surface as unsearched and hands the operator the CLI's own words (failure kind, exit status, stderr). Diagnostics never contain tokens, keys, or secret-looking values.

## Migrating from datasource access controls

Datasource authorization was removed. A connection is searchable as soon as it is configured under `datasources` and authenticated through its native CLI. If a config still carries the old block:

- Remove the `datasourceAccess` block and its `allowedTags` / `allowedScopes` retrieval fields.
- Remove any CLI `--tags` or MCP `tags` filters that were used only for datasource visibility.
- Keep query `scope`: it narrows a scope-capable datasource's results and can never widen visibility.
- Keep skill/descriptor `tags`: they remain descriptive metadata only.
