---
title: "ClawGallery"
description: "Search local screenshots and photos by caption and visual meaning through the external ClawGallery CLI."
---

# ClawGallery

The `clawgallery` datasource searches local screenshots and photos through the external `clawgallery` CLI. Captions, index state, and vector search all live in clawgallery; AutoRAG never reads `images.jsonl` or `vdr.sqlite3`, and never triggers captioning or renaming.

## Prerequisites

- The `clawgallery` CLI installed with the `clawgallery` binary on `PATH` (or set `connector.binaryPath`).
- A folder of images. `connector.path` scopes the bootstrap to it; when set, a missing folder fails the index.

Unverified: the upstream crate is installed with `cargo install clawgallery`; the install command is not pinned in this repository.

## Connect

Add a `datasources` entry to `~/.autorag/config.json`:

```json
{
  "datasources": {
    "screenshots": {
      "type": "clawgallery",
      "instanceId": "personal",
      "connector": {
        "binaryPath": "clawgallery",
        "syncVisual": true,
        "vdrBackend": "vsplade"
      }
    }
  }
}
```

| Connector option | Default | Purpose |
| --- | --- | --- |
| `binaryPath` | `clawgallery` | The executable, when it is not on `PATH`. |
| `path` | unset | Local image folder to bootstrap; passed as `--path`. |
| `syncVisual` | `true` | Run `clawgallery vdr sync` after bootstrap to build the vector index. |
| `vdrBackend` | clawgallery default | `mlx`, `jina-mlx`, `vsplade`, or `inl`; passed as `--backend` to `vdr sync`. `inl` uses the im-not-llm multimodal embedder (`google/embeddinggemma-2`, 768-d) served by `inl serve --multimodal`. |
| `defaultMode` | `hybrid` | The retrieval mode registered for search: `keyword`, `lexical`, `embedding`, or `hybrid`. |
| `configDir` | AutoRAG-managed under the workspace | clawgallery's `CLAWGALLERY_CONFIG_DIR`. |
| `timeoutMs` | `60000` | CLI spawn timeout. |

Then index it:

```bash
autorag refresh --method datasources
```

## What is indexed

- Refresh runs `clawgallery bootstrap` (incremental: it reports `indexed`, `skipped`, and `pruned`), then, when `syncVisual` is true, `clawgallery vdr sync --backend <backend>`. A failed visual sync is a warning, not a failure: the existing search index stays available.
- AutoRAG stores no image text itself; the CLI owns captions, V-SPLADE sparse vectors, and VDR dense vectors.
- Sources are `/screenshots/<instance>/images/<image-id>` (the source kind is `screenshots`, not the connection id). Result metadata carries `imageId`, `path`, `title`, and `caption` when the CLI reports them.

## Search behavior

- Modes map to clawgallery: `keyword` searches image paths and captions and works without any vector index; `lexical` needs a sparse V-SPLADE index (`vdr sync --backend vsplade`); `embedding` needs a dense VDR index (`vdr sync --backend mlx`, `vdr sync --backend inl`, or another dense backend); `hybrid` asks clawgallery to RRF-fuse keyword results with every available dense and sparse channel.
- The default mode is `hybrid`, which runs alone with `keyword` as a fallback only when hybrid fails. A mode pinned with `connector.defaultMode` runs by itself with no fallback.
- A scope narrows within the connection, for example `/screenshots/**` (see [Scope narrowing](search-semantics.md#scope-narrowing)).
- clawgallery results carry no timestamp, so a `since`/`until` window excludes every result and reports the method (see [Time windows](search-semantics.md#time-windows)).

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| `clawgallery binary not found` / `could not start` | `datasource-unavailable` | Install clawgallery or set `connector.binaryPath`. |
| `configured ClawGallery folder does not exist` | `datasource-index-failed` | Check `connector.path`. |
| `clawgallery bootstrap failed` | `datasource-index-failed` (or `datasource-unavailable` for a spawn/timeout failure) | Run `clawgallery bootstrap` directly to see the CLI's error. |
| `clawgallery visual sync failed` | warning `datasource-index-failed` | The keyword index remains searchable; inspect `clawgallery vdr status --json`. |
| A vector mode returns nothing | — | The mode's index does not exist. Prefer `hybrid` or `keyword`, or build the needed index with `vdr sync`. |
| `clawgallery bootstrap found no changed images` | info `datasource-empty` | The folder has no new images. |
