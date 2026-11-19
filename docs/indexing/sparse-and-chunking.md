---
title: "Sparse index and chunking"
description: "SPLADE sparse retrieval, sparse-only workspaces, upgrading a BM25 index, measured quality, and the MinSync chunk size."
---

# Sparse index and chunking

This page covers MinSync's SPLADE sparse retrieval, how to run a sparse-only workspace, how a BM25-era index is upgraded, measured quality, and how to set the chunk size.

## SPLADE sparse retrieval

MinSync replaced its BM25 full-text index with learned-sparse retrieval: `nlpai-lab/KURE-Sparse-nano` (Apache-2.0) encodes every chunk and every query into weighted vocabulary terms. The keyword half of local search (the `sparse` mode, and the sparse side of `hybrid`) runs through it. It needs no tokenizer library: Korean inflected forms (`인증서를` → `인증서`) match without Kiwi, which AutoRAG no longer probes for, and MinSync no longer takes `--language` at init.

MinSync downloads the 299 MB sparse model into `~/.cache/minsync/models` on first use (sha256-verified) and runs it in-process on Metal or CPU. AutoRAG does not start a `minsync serve --sparse` daemon.

The known limit, from MinSync's release notes: exact strings such as error codes, hashes, and version strings among many look-alike documents match far less reliably than with BM25. For exact lookups, use a bounded `grep`.

## Sparse-only index

Set `"minSync": { "embedder": { "id": "none" } }` to build a SPLADE-only index: MinSync encodes sparse vectors and no dense ones, so no dense model is downloaded or run (indexing a small corpus peaked at about 1.5 GB). Every MinSync search is then a sparse search: the `hybrid` and `minsync` methods query `--mode sparse` and describe themselves as `sparse`, an Obsidian vault offers only `obsidian-sparse`, the shared embedder reports no backend (memory, chat sidecars, and gist keep their keyword paths), and neither `autorag status` nor `autorag setup` warns about a missing embedder.

A dense index cannot become sparse-only in place, or back: the vector column's dimension differs. When the configured embedder switches between `none` and a dense one, the next refresh runs `minsync init --force` with the new embedder and rebuilds the index with a full sync.

## Upgrading a BM25 workspace

AutoRAG requires MinSync 0.7.4 or newer and installs 0.7.4. A `minsync` on `PATH` or in the workspace cache that reports an older version is never run: one later on `PATH` is used instead, a cached one is replaced by the next install (`autorag refresh` with `minSync.autoInstall`), and a query with no usable binary fails with a `MinSyncRequiredError` naming the version and path it found. A binary whose `--version` does not print a MinSync version line (a custom build) is not gated. MinSync 0.7.4 needs rustc 1.94 or newer for `cargo install`; with an older toolchain the install fails and AutoRAG falls back to the GitHub release binary.

On the first `autorag refresh` after the upgrade, AutoRAG removes the `[lexical]` table from MinSync's config file and MinSync migrates the index in place: the BM25 index and `lexical_text` column are dropped and sparse vectors are encoded for every chunk; dense vectors are kept. This is a one-time cost that grows with the corpus. On an Apple M4 Max, a 5,394-file workspace with 52,851 chunks took 38 minutes (45,361 chunks encoded, 7,490 reused) with a 2.7 GB peak memory footprint, and the store grew from 0.65 GB to 2.3 GB.

## Measured quality

On that workspace (the maintainer's own Downloads/Documents/Desktop, mostly Korean), 594 queries were generated from 297 sampled passages (one natural question and one keyword query each), and recall counted a hit when the passage's file, or a byte-identical copy, ranked in the top k:

| Mode | R@1 | R@5 | R@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| BM25 (`multilingual`, MinSync 0.5.1) | 0.473 | 0.732 | 0.811 | 0.583 |
| SPLADE sparse (MinSync 0.6.0) | 0.564 | 0.825 | 0.887 | 0.676 |
| Hybrid, BM25 + dense (0.5.1) | 0.481 | 0.779 | 0.848 | 0.607 |
| Hybrid, SPLADE + dense (0.6.0) | 0.492 | 0.779 | 0.847 | 0.614 |

For the natural-language questions, a model judged whether the top 5 chunks contained the reference answer: BM25 0.657, SPLADE 0.838, hybrid 0.717 → 0.768.

## Chunk size

MinSync defaults to `max_chunk_size = 4096`. Smaller-context local models may need a smaller value. Set `minSync.maxChunkSize` in `config.json`, or pass `--minsync-max-chunk-size` to `autorag init`:

```bash
autorag init \
  --search-paths /path/to/docs \
  --workspace /path/to/workspace \
  --minsync-max-chunk-size 1000 \
  --force
```

AutoRAG writes the value to MinSync's `[chunker.options].max_chunk_size` and forces a full reindex when the configured chunk size changes.

## Related pages

- [Embedding backends](embedding-backends.md) — the embedding model, backends, and daemon.
- [Overview](overview.md) — the pipeline and workspace layout.
- [Retrieval overview](../retrieval/overview.md) — how sparse, vector, and hybrid modes are used.
