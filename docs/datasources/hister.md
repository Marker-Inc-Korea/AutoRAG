---
title: "Browser history (Hister)"
description: "Search the web pages you visited through a running Hister server, and share AutoRAG's embedding model with it."
---

# Browser history (Hister)

The `hister` datasource searches the web pages your browser visited, through a running [Hister](https://github.com/asciimoo/hister) server. Hister is a separate program under AGPL-3.0-or-later: AutoRAG ships none of its code, does not modify it, and only sends search requests to its public HTTP API. Hister owns the index, the browser extension, incremental capture, and (optionally) the embeddings.

## Prerequisites

- A running Hister server. Start it with `hister listen`; it serves `http://127.0.0.1:4433` by default and must stay running for the extension to capture pages and for AutoRAG to search. AutoRAG never starts or stops it.

```bash
brew install hister                      # or a release binary, Docker, Nix
hister listen                            # http://127.0.0.1:4433, keep it running
hister search 'domain:example.com'       # must print rows before wiring it up
```

- The Hister browser extension installed in Chrome/Chromium or Firefox, so pages are captured as you browse.
- Optional: an access token, if the Hister server requires one.

## Connect

Add a `datasources` entry to `~/.autorag/config.json`. The connection template is `hister`; give it an alias to make the identity readable:

```json
{
  "datasources": {
    "browser": {
      "type": "hister",
      "connector": {
        "baseUrl": "http://127.0.0.1:4433",
        "tokenEnv": "HISTER_TOKEN",
        "semantic": false
      }
    }
  }
}
```

| Connector option | Default | Purpose |
| --- | --- | --- |
| `baseUrl` | `http://127.0.0.1:4433` | The Hister server. Must be `http` or `https`. |
| `tokenEnv` | unset | Name of the environment variable holding Hister's `app.access_token`. The token itself is never configured. |
| `timeoutMs` | `30000` | Per-request timeout. |
| `semantic` | `false` | Also register Hister's semantic search. |

::: warning **Security:** a non-loopback `http://` URL is refused when `tokenEnv` is set, because the token would travel in clear text. Use `https` for a remote server.
:::

## What is indexed

- Web pages only. Every query is sent confined to `type:web`, and any document Hister returns with another type (watched local files, file snapshots) is dropped. Hister's local-directory indexing is never searched here.
- Nothing is copied: Hister indexes each page when the extension sends it, so `autorag refresh` only probes the server for reachability, authentication, page count, and whether it also holds non-web documents. The probe is read-only.
- Sources are `/<id>/<instance>/pages/<domain>/<hash of the URL>`, where `<id>` is the connection id (default `hister`). The URL itself is in the hit content and in `metadata.url`; read the page as it is now with `web_fetch`.

## Search behavior

- Lexical full-text search through Hister (method `<id>-lexical`). Plain words in a query are sent as one alternation group, `(a|b|c)` — Hister's only OR; the word `OR` is searched literally. Hister syntax (`"phrases"`, `title:`, `text:`, `domain:`, `url:`, `language:`, `-term`, `secur*`, `added:<7d`, `sort:date`) passes through as written.
- A scope of `/<id>/<instance>/pages/<domain>/**` is pushed down to Hister as `domain:<domain>`. Other scopes narrow locally (see [Scope narrowing](search-semantics.md#scope-narrowing)).
- With `connector.semantic: true`, Hister's semantic search is also registered (`<id>-semantic`). It requires `semantic_search.enable` plus an embedding endpoint in the Hister config; the page text is sent to that endpoint, so use a loopback one for privacy. With it disabled on the server, the semantic method fails loudly instead of returning nothing.
- A `since`/`until` window uses the page's last visit (see [Time windows](search-semantics.md#time-windows)).
- The agent may also call `hister` directly through `bash`, for example `hister search 'domain:example.com sort:date' --format json --limit 20` or `hister doctor`.

## Sharing the embedding model

Hister keeps its own vector store and embeds with whatever its `config.yml` `semantic_search` block says. If that model differs from AutoRAG's, nothing fails — the vectors simply stop lining up and results get worse. The two must share one model, and `autorag setup` handles it:

```bash
autorag setup                # reports hister.embedding.differs: the keys that disagree; writes nothing
autorag setup --sync-hister  # writes them, restarts Hister, waits, re-embeds pages when needed
hister doctor                # the line server.index.embeddings must be ok
```

- The source of truth is the resolved shared embedder (the `minsync serve` daemon by default), so Hister embeds queries exactly as AutoRAG does.
- `--sync-hister` edits only the six keys `enable`, `embedding_endpoint`, `embedding_model`, `dimensions`, `query_prefix`, and `document_prefix` in Hister's own `config.yml`, keeping a `config.yml.bak.<time>` next to it. Every other byte is preserved.
- It restarts the service with `brew services restart hister`, waits until the server answers, and runs `hister reindex` only when a changed key makes stored vectors stale (the query prefix alone does not). It is idempotent.
- It refuses, and says why, for a remote shared backend (browsing text would leave the machine and Hister's config would have to hold the key) and for MinSync's in-process engine (Hister cannot reach it): start `minsync serve --idle-timeout 0` first (see [Embedding backends](../indexing/embedding-backends.md)).
- The daemon's port is chosen when it starts, so run `autorag setup --sync-hister` again after restarting the daemon. If the report says `restartRequired` or `reindexRequired`, run `brew services restart hister` and `hister reindex` yourself.

## Running it day to day

- On macOS, `brew services start hister` makes Hister a login item and `brew services stop hister` removes it. A page visited while Hister is down is not sent again later.
- Backfill a gap with `hister import browser history --start-date YYYY-MM-DD`.

::: danger **Backfilling is not private by default:** `hister import browser history` re-downloads each URL *without* your logged-in cookies. A login-gated URL is saved as its login page, and tokens in a URL are sent to that site again. Before importing, put a skip list in Hister's `rules.json` (one regular expression per entry, matched against the full URL and also applied to future captures). This is a deny list: a new, unlisted login page is still captured — use an `allow` list instead if you want the reverse. Check the result with `hister list-urls`.
:::

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| `hister server at … is not reachable` | `datasource-unavailable` | Start it with `hister listen` and check `connector.baseUrl`. |
| `hister requires an access token` | `datasource-auth-error` | Set `connector.tokenEnv` to the variable holding Hister's `app.access_token`. |
| `hister holds no web pages yet` | info `datasource-empty` | Install the extension and browse, or run `hister import browser history`. |
| `hister also holds N non-web document(s)` | warning `datasource-index-partial` | Harmless: those documents are excluded and never searched. |
| `connector.semantic` is true but semantic search is off | warning `datasource-unavailable` | Set `semantic_search.enable` in Hister's config and restart it, or set `connector.semantic: false`. |
| Semantic search returns `semantic-disabled` | — | The server did not run a semantic search; enable it or use `autorag setup --sync-hister`. |
