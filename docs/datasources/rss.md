---
title: "RSS"
description: "Poll RSS 2.0 and Atom feeds into a lexical index and search recent articles with a 24-hour dedupe window."
---

# RSS

The `rss` datasource polls a trusted list of RSS 2.0 and Atom feeds and indexes their items for lexical search. Use it for questions about recent news items or blog posts the server subscribes to.

## Prerequisites

- One or more feed URLs.
- Nothing to install: the connector fetches the feeds over HTTP with AutoRAG's own XML parser.

## Connect

Add a `datasources` entry to `~/.autorag/config.json`:

```json
{
  "datasources": {
    "rss": {
      "connector": {
        "feeds": [
          { "url": "https://example.com/feed.xml", "category": "tech" }
        ]
      }
    }
  }
}
```

| Connector option | Default | Purpose |
| --- | --- | --- |
| `feeds` | none (required) | Array of `{ "url": "…" }`, each with an optional `category`. |
| `maxDocuments` | `500` | Cap on items indexed across all feeds. |
| `maxItemsPerFeed` | `100` | Cap on items per feed. |
| `maxContentChars` | `20000` | Characters kept from each item body. |
| `timeoutMs` | `30000` | Per-feed fetch timeout (feeds are often slow static hosts). |

The feed list is trusted configuration: it is never settable from model or tool arguments. Then index it:

```bash
autorag refresh --method datasources
```

## What is indexed

- RSS 2.0 and Atom items. Item HTML is stripped to plain text; content is the title plus the body, truncated to `maxContentChars`.
- The item id is its `guid`/`link` (RSS) or `id` (Atom); it becomes the opaque source `/rss/<instance>/chunks/<chunk-id>`.
- Hierarchy paths are `feeds/<feedTitle>` or `feeds/<category>/<feedTitle>`; metadata carries `feedIndex`, `feedTitle`, and `categories`.
- The item time is `pubDate` (RSS) or `published`/`updated` (Atom).
- Feeds are polled with a 24-hour dedupe window, so a re-delivered item is indexed once per window.
- The chunk store persists to `<workspace>/.autorag/datasources/rss/<instance>/chunks.json`.

## Search behavior

- Retrieval is lexical BM25 (`rss-lexical`), narrowed by `scope` (see [Scope narrowing](search-semantics.md#scope-narrowing)).
- Time windows are applied natively over the item's date before the result cut; an item with no date ranks last and is excluded and reported (see [Time windows](search-semantics.md#time-windows)).
- Default poll cadence is 15 minutes while auto-refresh runs.

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| `not-configured: no feeds configured` | `datasource-unavailable` | Set `connector.feeds`. |
| One feed failed or is unparseable | warning `datasource-index-failed` | The other feeds still index; check the feed URL and its XML. |
| Every feed failed | `datasource-unavailable` (`all configured feeds failed`) | Check connectivity and the feed hosts. |
| A windowed search drops undated items | `time-window-untimed` | The feed items carry no date; widen or remove the window. |
