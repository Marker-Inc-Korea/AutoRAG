# MinSync setup

AutoRAG uses MinSync for local lexical BM25, semantic vector, and hybrid
retrieval over parsed document mirrors. MinSync remains the owner of its
workspace, chunks, vectors, cursor, and embedding identity.

## Default path: the AutoRAG gateway

The product default is the AutoRAG-owned `autorag-gateway` path. On a new
workspace, MinSync starts the local gateway on demand and uses the
`qwen3-embedding-0.6b` profile (1024 dimensions). No Ollama installation, TEI
adapter, API key, endpoint, or model selection is required.

MinSync still auto-installs its own binary when no usable executable is found
on `PATH` or in the workspace cache. If the MinSync binary is missing or its
auto-install fails, AutoRAG reports a degraded result; it does not claim the
index is ready.

A normal setup is:

```bash
autorag init --search-paths /path/to/docs --workspace /path/to/workspace
autorag refresh --method parsed,minsync --json
autorag search --json "semantic question about the documents"
```

The gateway is loopback-only and is started by the semantic MinSync path. To
inspect or stop it:

```bash
autorag gateway status --format json
autorag gateway stop
```

`gateway status` reports the supervisor state, profile, model, backend, port
when available, health, and sanitized failure details. `gateway stop` stops
the owned child process and removes its pid/lock state.

## Selecting a profile

The default profile is Qwen3-Embedding-0.6B at 1024 dimensions. To keep a
768-dimensional workspace, set a profile in trusted `config.json` rather than
pointing the product path at an implicit Ollama model:

```json
{
  "minSync": {
    "embedder": {
      "profile": "embeddinggemma-300m"
    },
    "maxChunkSize": 1000
  }
}
```

EmbeddingGemma uses the prefixes `task: search result | query: ` for queries
and `title: none | text: ` for passages. Its 2048-token context makes a
smaller `maxChunkSize` useful. AutoRAG writes only the supported MinSync
embedder fields and does not write secrets.

## Model cache and offline operation

The runtime cache root is `~/.autorag` on macOS and `%USERPROFILE%\\.autorag` on
Windows; `AUTORAG_HOME` overrides it. The cache downloader stores verified
assets under the cache root's `models/` directory. Downloads use a `.part` file
and an atomic rename. A failed or corrupt download is removed rather than
treated as a usable model.

Use the CLI to fetch, import, or verify a pinned model:

```bash
autorag models prefetch --profile qwen3-embedding-0.6b
autorag models import /path/to/Qwen3-Embedding-0.6B-Q8_0.gguf --profile qwen3-embedding-0.6b
autorag models verify --profile qwen3-embedding-0.6b
```

`import` verifies the source file against the selected profile hash before
placing it in the cache. `verify` never downloads a missing asset. The runtime
API also accepts `offline: true`; in that mode a missing or corrupt model or
runtime asset returns an `offline-missing` diagnostic and makes no network
request. Offline import works when the source file is already available.

The exact model and runtime URLs, revisions, SHA-256 values, archive members,
and notices are in [`licenses/embedding-assets.json`](../licenses/embedding-assets.json).
Mutable model weights are not in the npm package.

## Chunk size

MinSync defaults to `max_chunk_size = 4096`. Smaller-context local models may
need a smaller value. Set `minSync.maxChunkSize` in `config.json`, or pass
`--minsync-max-chunk-size` to `autorag init`:

```bash
autorag init \
  --search-paths /path/to/docs \
  --workspace /path/to/workspace \
  --minsync-max-chunk-size 1000 \
  --force
```

AutoRAG writes the value to MinSync's `[chunker.options].max_chunk_size` and
forces a full reindex when the configured chunk size changes.

## Existing Ollama/TEI adapter path: legacy/manual QA only

The repository's Python adapter is no longer the default product path. Keep it
only for a legacy workspace or manual compatibility QA. It translates Ollama's
`/api/embeddings` response to MinSync's TEI `/embed` response and must remain
bound to loopback:

```bash
ollama pull embeddinggemma:latest
ollama serve
OLLAMA_EMBEDDINGS_URL=http://127.0.0.1:11434/api/embeddings \
  python3 scripts/manual-qa/ollama-tei-adapter.py
```

A legacy/manual QA workspace may be initialized explicitly as follows:

```bash
autorag init \
  --search-paths /path/to/docs \
  --workspace /path/to/workspace \
  --embedder-id tei:embeddinggemma:latest \
  --embedder-base-url http://127.0.0.1:18080 \
  --embedder-dimension 768 \
  --minsync-max-chunk-size 1000 \
  --force
autorag refresh --method parsed,minsync --json
```

Do not use this adapter path as a fresh-install requirement or as an implicit
fallback from the gateway. AutoRAG never silently switches an existing vector
store between Ollama/TEI and the shared runtime.

Direct Ollama OpenAI-compatible embedding endpoints are also operator-managed
legacy configuration, not the AutoRAG default. An explicit external endpoint
remains authoritative and is not overwritten by the shared runtime.

## Migration and failure behavior

A MinSync workspace records the embedding identity in:

```text
<workspace>/.autorag/minsync/.minsync/autorag-embedding-identity.json
```

MinSync's own store, cursor (`cursor.json`), and config (`config.toml`) live
under that same `.minsync` subdirectory.

The identity includes provider, model, immutable model revision, dimension,
query prefix, passage prefix, and runtime build. A mismatch requires a full
reindex. Existing 768-dimensional Ollama/TEI stores must therefore be handled
by either:

- an explicit `autorag index reset --method minsync --yes` followed by
  `autorag refresh --method minsync`; or
- an explicit `minSync.embedder.profile` of `embeddinggemma-300m`, followed by
  refresh and identity validation.

A dimension mismatch is detected before vector reuse. MinSync reports a typed
semantic failure rather than mixing vector dimensions. When semantic
infrastructure is unavailable, the lexical/BM25 lane remains usable; hybrid
retrieval can fall back to BM25 and reports the degraded semantic diagnostic.

The legacy migration diagnostic is:

```text
This workspace uses the legacy 768-dimensional Ollama/TEI embedding path. Reindex explicitly, or pin an explicit profile config before using the new default runtime.
```
