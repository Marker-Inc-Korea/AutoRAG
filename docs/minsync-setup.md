# MinSync setup

AutoRAG uses MinSync for local lexical BM25, semantic vector, and hybrid
retrieval over parsed document mirrors.

## Automatic installation

MinSync is enabled by default. When no usable `minsync` executable is found
in the configured `binaryPath`, on `PATH`, or in the workspace cache, AutoRAG
downloads the verified release asset for the current platform into:

```text
<workspace>/.autorag/bin/minsync
```

Release assets are selected by platform and architecture and verified against
their SHA-256 digest before installation. If a compatible release asset is not
available, AutoRAG falls back to:

```bash
cargo install minsync --version 0.3.0 --locked
```

The fallback requires a Rust toolchain. Installation failures are reported as
a degraded MinSync status; AutoRAG does not claim that the index is ready.

To manage MinSync yourself, set an explicit path and disable installation:

```json
{
  "minSync": {
    "binaryPath": "/absolute/path/to/minsync",
    "autoInstall": false
  }
}
```

An explicit `binaryPath` is authoritative. If it is missing, AutoRAG reports a
missing binary instead of silently installing another executable.

## Local EmbeddingGemma

For a local-only semantic index, run Ollama and expose its embeddings through
the repository TEI adapter:

```bash
ollama pull embeddinggemma:latest
ollama serve
OLLAMA_EMBEDDINGS_URL=http://127.0.0.1:11434/api/embeddings \
  python3 scripts/manual-qa/ollama-tei-adapter.py
```

Then initialize a workspace with:

```bash
autorag init \
  --search-paths /path/to/docs \
  --workspace /path/to/workspace \
  --embedder-id tei:embeddinggemma:latest \
  --embedder-base-url http://127.0.0.1:18080 \
  --embedder-dimension 768 \
  --force
autorag refresh --method parsed,bm25,minsync --json
```

Verify both the index and the local semantic query:

```bash
autorag status --json
autorag search --json "semantic question about the documents"
```

The MinSync workspace is local to the configured AutoRAG workspace. The
embedding adapter must remain bound to loopback; do not use a remote endpoint
for private corpus text in this QA flow.
