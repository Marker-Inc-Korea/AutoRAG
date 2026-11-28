---
title: "Browser history (Nogon)"
description: "Search the web pages you visited through a running Nogon server, and share AutoRAG's embedding model with it."
---

# Browser history (Nogon)

::: tip **Formerly Hister**
Nogon was called Hister until v0.22.0 (asciimoo renamed it after a trademark complaint; the data, config, and HTTP API are unchanged). An AutoRAG config written for Hister keeps working: `type: "hister"` is accepted as the `nogon` template, the connection id you chose (for example `hister`) is kept so existing `/<id>/...` source identities stay valid, `autorag setup` accepts a still-installed `hister` program, and the old `HISTER_CONFIG`, `~/.config/hister/`, and `--sync-hister` spellings are read as before. New configs should use `nogon`.
:::

The `nogon` datasource searches the web pages your browser visited, through a running [Nogon](https://github.com/asciimoo/nogon) server. Nogon is a separate program under AGPL-3.0-or-later: AutoRAG ships none of its code, does not modify it, and only sends search requests to its public HTTP API. Nogon owns the index, the browser extension, incremental capture, and (optionally) the embeddings.

## Prerequisites

- A running Nogon server. Start it with `nogon listen`; it serves `http://127.0.0.1:4433` by default and must stay running for the extension to capture pages and for AutoRAG to search. AutoRAG never starts or stops it.

```bash
brew install nogon                      # or a release binary, Docker, Nix
nogon listen                            # http://127.0.0.1:4433, keep it running
nogon search 'domain:example.com'       # must print rows before wiring it up
```

- The Nogon browser extension installed in Chrome/Chromium or Firefox, so pages are captured as you browse.
- Optional: an access token, if the Nogon server requires one.

## Connect

Add a `datasources` entry to `~/.autorag/config.json`. The connection template is `nogon`; give it an alias to make the identity readable:

```json
{
  "datasources": {
    "browser": {
      "type": "nogon",
      "connector": {
        "baseUrl": "http://127.0.0.1:4433",
        "tokenEnv": "NOGON_TOKEN",
        "semantic": false
      }
    }
  }
}
```

| Connector option | Default | Purpose |
| --- | --- | --- |
| `baseUrl` | `http://127.0.0.1:4433` | The Nogon server. Must be `http` or `https`. |
| `tokenEnv` | unset | Name of the environment variable holding Nogon's `app.access_token`. The token itself is never configured. |
| `timeoutMs` | `30000` | Per-request timeout. |
| `semantic` | `false` | Also register Nogon's semantic search. |

::: warning **Security:** a non-loopback `http://` URL is refused when `tokenEnv` is set, because the token would travel in clear text. Use `https` for a remote server.
:::

## What is indexed

- Web pages only. Every query is sent confined to `type:web`, and any document Nogon returns with another type (watched local files, file snapshots) is dropped. Nogon's local-directory indexing is never searched here.
- Nothing is copied: Nogon indexes each page when the extension sends it, so `autorag refresh` only probes the server for reachability, authentication, page count, and whether it also holds non-web documents. The probe is read-only.
- Sources are `/<id>/<instance>/pages/<domain>/<hash of the URL>`, where `<id>` is the connection id (default `nogon`). The URL itself is in the hit content and in `metadata.url`; read the page as it is now with `web_fetch`.

## Search behavior

- Lexical full-text search through Nogon (method `<id>-lexical`). Plain words in a query are sent as one alternation group, `(a|b|c)`; a typed `AND`/`OR`/`NOT` is dropped, not searched. **Stopwords are removed first** (see below). Nogon syntax (`"phrases"`, `title:`, `text:`, `domain:`, `url:`, `language:`, `-term`, `secur*`, `added:<7d`, `sort:date`) passes through as written.
- A scope of `/<id>/<instance>/pages/<domain>/**` is pushed down to Nogon as `domain:<domain>`. Other scopes narrow locally (see [Scope narrowing](search-semantics.md#scope-narrowing)).
- With `connector.semantic: true`, Nogon's semantic search is also registered (`<id>-semantic`). It requires `semantic_search.enable` plus an embedding endpoint in the Nogon config; the page text is sent to that endpoint, so use a loopback one for privacy. With it disabled on the server, the semantic method fails loudly instead of returning nothing.
- A `since`/`until` window uses the page's last visit (see [Time windows](search-semantics.md#time-windows)).
- The agent may also call `nogon` directly through `bash`, for example `nogon search 'domain:example.com sort:date' --format json --limit 20` or `nogon doctor`.

## Sharing the embedding model

Nogon keeps its own vector store and embeds with whatever its `config.yml` `semantic_search` block says. If that model differs from AutoRAG's, nothing fails — the vectors simply stop lining up and results get worse. The two must share one model, and `autorag setup` handles it:

```bash
autorag setup                # reports nogon.embedding.differs: the keys that disagree; writes nothing
autorag setup --sync-nogon  # writes them, restarts Nogon, waits, re-embeds pages when needed
nogon doctor                # the line server.index.embeddings must be ok
```

- The source of truth is the resolved shared embedder (the `inl serve` daemon by default), so Nogon embeds queries exactly as AutoRAG does.
- `--sync-nogon` edits only the six keys `enable`, `embedding_endpoint`, `embedding_model`, `dimensions`, `query_prefix`, and `document_prefix` in Nogon's own `config.yml`, keeping a `config.yml.bak.<time>` next to it. Every other byte is preserved.
- It restarts the service with `brew services restart nogon`, waits until the server answers, and runs `nogon reindex` only when a changed key makes stored vectors stale (the query prefix alone does not). It is idempotent.
- It refuses, and says why, for a remote shared backend (browsing text would leave the machine and Nogon's config would have to hold the key) and for MinSync's in-process engine (Nogon cannot reach it): start `inl serve --idle-timeout 0` first (see [Embedding backends](../indexing/embedding-backends.md)).
- Nogon's config holds the daemon's URL. inl-server 0.3.4 and later bring a restarted daemon back on the same port, so the URL stays valid; an older daemon picks a new port on every start, so run `autorag setup --sync-nogon` again after restarting it. `autorag setup` reports `embedding_endpoint` as out of sync when the URL no longer matches. If the report says `restartRequired` or `reindexRequired`, run `brew services restart nogon` and `nogon reindex` yourself.

## Running it day to day

- On macOS, `brew services start nogon` makes Nogon a login item and `brew services stop nogon` removes it. A page visited while Nogon is down is not sent again later.
- Backfill a gap with `nogon import browser history --start-date YYYY-MM-DD`.

::: danger **Backfilling is not private by default:** `nogon import browser history` re-downloads each URL *without* your logged-in cookies. A login-gated URL is saved as its login page, and tokens in a URL are sent to that site again. Before importing, put a skip list in Nogon's `rules.json` (one regular expression per entry, matched against the full URL and also applied to future captures). This is a deny list: a new, unlisted login page is still captured — use an `allow` list instead if you want the reverse. Check the result with `nogon list-urls`.
:::

## How keyword search ranks, and the stopwords AutoRAG removes

Nogon's keyword search is an analyzed full-text index, not exact matching: text is tokenized and lower-cased, and a document in a language with an analyzer (English, German, French, Spanish, ... and CJK bigrams for Korean, Japanese and Chinese) is stemmed and loses that language's stopwords at index time. A page whose language is not recognized keeps every word. Results are scored with TF-IDF (Bleve's default; Nogon sets no BM25 model), and fields are weighted: a match in the title counts 12×, the domain 8×, the URL 4×, the body 1×.

That weighting is made for a short query typed into a search box. A natural-language question sent as alternatives lets a function word in any page title outrank the one rare word in a page body: on a 1,600-page index the correct page ranked 24th, and the same corpus under Hister v0.20 and v0.21 ranked it the same, so this is not a rename regression.

So before building the alternation AutoRAG drops stopwords (`src/retrieval/stopwords.ts`), for the languages it supports: English, Korean, Japanese, Chinese, French, German, Spanish, Italian, Portuguese, Russian, Vietnamese, Thai, Arabic and Hindi. They are function words (articles, pronouns, prepositions, auxiliaries, question words) and the request forms "찾아줘" / "教えて", not a full linguistic set. Rules:

- A script that identifies its language (Hangul, kana, Han, Cyrillic, Arabic, Devanagari, Thai) is filtered word by word.
- Latin script is filtered only after the query shows its language: at least two unambiguous stopwords of one language. German `die`/`war` and French `son` are ordinary English words, so `war in ukraine` is left alone.
- A query that is all stopwords, or a single word, is left as it is.
- A query written in Nogon syntax (`"phrases"`, `field:`, `-term`) is sent as written, stopwords included. Semantic search always gets the whole question.

Measured on a real 1,608-page index with a word that occurs in exactly one page: Korean questions went to rank 1 every time; English questions went from ranks between 2 and beyond 100 to ranks between 2 and 19, so they are much better but not always first, because a content word such as `read` that appears in a title still outweighs a rare word in a body. Use a short keyword query, or semantic search, for the rest.

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| `nogon server at … is not reachable` | `datasource-unavailable` | Start it with `nogon listen` and check `connector.baseUrl`. |
| `nogon requires an access token` | `datasource-auth-error` | Set `connector.tokenEnv` to the variable holding Nogon's `app.access_token`. |
| `nogon holds no web pages yet` | info `datasource-empty` | Install the extension and browse, or run `nogon import browser history`. |
| `nogon also holds N non-web document(s)` | warning `datasource-index-partial` | Harmless: those documents are excluded and never searched. |
| `connector.semantic` is true but semantic search is off | warning `datasource-unavailable` | Set `semantic_search.enable` in Nogon's config and restart it, or set `connector.semantic: false`. |
| Semantic search returns `semantic-disabled` | — | The server did not run a semantic search; enable it or use `autorag setup --sync-nogon`. |
