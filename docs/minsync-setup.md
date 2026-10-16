# MinSync setup

AutoRAG uses MinSync for local lexical BM25, semantic vector, and hybrid
retrieval over parsed document mirrors. MinSync remains the owner of its
workspace, chunks, vectors, cursor, and embedding identity.

## Default path: the AutoRAG gateway

The product default is the AutoRAG-owned `autorag-gateway` path. On a new
workspace, MinSync starts the local gateway on demand and uses the
`qwen3-embedding-0.6b` profile (1024 dimensions). No Ollama installation, TEI
adapter, API key, endpoint, or model selection is required.

MinSync is required. AutoRAG auto-installs its binary when no usable executable
is found on `PATH` or in the workspace cache. If the binary is missing or its
auto-install fails, `autorag search`, `autorag refresh`, `autorag lite retrieve`,
and the MCP tools fail with a `MinSyncRequiredError` message and a non-zero exit;
AutoRAG never answers from the other retrieval paths instead. `minSync: false`
and `minSync.enabled: false` are rejected as config errors. Install the binary
yourself with `cargo install minsync` when auto-install cannot run.

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

The default profile is Qwen3-Embedding-0.6B at 1024 dimensions; it is the only
bundled runtime profile. A 768-dimensional workspace cannot be kept by profile
selection anymore: reindex for the default profile, or pin an explicit legacy
Ollama/TEI embedder endpoint (`minSync.embedder.id` and `baseUrl`, or the
`--embedder-id`/`--embedder-base-url` init flags documented below). AutoRAG
writes only the supported MinSync embedder fields and does not write secrets.

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

## Refresh diagnostics

Parsed refresh reads each scanned file before MinSync indexes its mirror. A
source deleted or replaced after the scan (`ENOENT`, `ENOTDIR`, or `EISDIR`)
records a `source-vanished` informational diagnostic. A source that cannot be
read (`EACCES` or `EPERM`) records a `source-unreadable` warning. The
diagnostic `source` is the virtual path, and the message is the underlying
filesystem error. Refresh removes any previous mirror for that path and
continues with the remaining files. These transient reads are not stored as
skip records, so a later refresh tries the source again once it can be read.
Any other read error, a parser that throws something other than `ParseError`, and
a failed mirror write are also isolated to the one file. Each is skipped and the
refresh continues:

| Code | Severity | Cause |
| --- | --- | --- |
| `source-read-failed` | warning | Read error not listed above (`EMFILE`, `EIO`, `EBUSY`, `ETIMEDOUT`, …); `reason` carries the errno code. |
| `parser-crashed` | warning | The parser threw a plain `Error` or non-`ParseError` value; `reason` carries the error name. The message is withheld because it may contain a path. |
| `mirror-write-failed` | warning | Writing the parsed mirror for this file failed; `reason` carries the errno code. |

`EMFILE`, `EBUSY`, `EIO`, and `ETIMEDOUT` are retried with doubling backoff
(50 ms, 100 ms, 200 ms) before the file is skipped. The retry count is the
`refresh.transientRetries` config key (default `3`, `0` disables retries):

```json
{ "refresh": { "transientRetries": 3 } }
```

Like `source-unreadable`, these skips are not persisted, so the next refresh
tries the file again. Mirror writes go through a temp file that is removed on
any failure, so a failed write never leaves a `*.tmp` orphan.

A write failure that makes the workspace itself unusable (`ENOSPC`, `EDQUOT`,
`EACCES`, `EPERM`, `EROFS`) is not a per-file problem and aborts the refresh with
a `MirrorWorkspaceError` naming the workspace. Files already mirrored in that run
stay in the index, so the next refresh resumes incrementally.

`AutoRAGAgent.refresh()` and `getRefreshStatus()` expose all of these codes
through the public `SearchDocumentDiagnosticCode` type, alongside parser
diagnostics.

### Unavailable search roots and mass deletions

Parsed refresh never treats a search root it cannot see as proof that its
documents are gone. Before walking, each root is probed. A root that no longer
exists (`gone`), is no longer a directory (`not-a-directory`), cannot be listed
(`unreadable`), or is empty although it previously held indexed documents
(`empty`) is skipped: its previous mirrors and skip records stay untouched and
refresh records one `search-root-unavailable` warning whose `source` is the
opaque root prefix (for example `/docs`) and whose `reason` is one of the four
values. `deleted-mirror` is never emitted for these roots. Healthy roots in the
same refresh are still indexed and still delete their genuinely stale mirrors.
A forced refresh (`force`) treats an empty root as intentionally emptied.

A refresh that would delete more than `refresh.maxDeleteRatio` (default `0.3`)
of the indexed documents, and more than 100 of them, is refused: nothing is
deleted and one `mass-delete-suppressed` warning is recorded. The next refresh
repeats the check; a forced refresh (`autorag refresh --force`) deletes them.

The directory walk is bounded by `refresh.maxScannedEntries` (default
`2000000` entries per root) and `refresh.maxDepth` (default `64`). Hitting
either records a `walk-truncated` warning (`reason` `max-entries` or
`max-depth`) and keeps mirrors for anything the walk did not reach. A single
unreadable subdirectory records `directory-unreadable` with the opaque virtual
path of that directory, keeps the mirrors beneath it, and does not stop its
siblings from indexing.

```json
{ "refresh": { "maxDeleteRatio": 0.3, "maxScannedEntries": 2000000, "maxDepth": 64 } }
```

## Cancelling a refresh

`AutoRAGAgent.refresh(force, { signal })` forwards the signal to the MinSync
`init`, `check`, and `sync` processes. Aborting stops the running `minsync` and
every descendant it started (SIGTERM to the process group, SIGKILL after 5
seconds), restores the `config.toml` this run rewrote, and rejects `refresh`
with the signal's reason. On POSIX each MinSync process leads its own process
group, so the host also reaps it on SIGINT, SIGTERM, SIGHUP, and exit; Windows
uses `taskkill /t`. A run without a signal behaves as before. Only the MinSync
phase observes the signal; parse, datasource, and Jikji phases do not yet.

A `minsync` run that outlives its host (for example after `SIGKILL` of the
host, which no handler can catch) keeps `.minsync/lock`. The next `check` or
`sync` exits 3 with `another sync is in progress`, which AutoRAG reports as the
retryable `minsync-lock-held` diagnostic and a `lock-held` result reason instead
of a generic `minsync-sync-failed`.

## Skipped files (partial sync)

MinSync 0.4.8 skips a file it cannot read, chunk, or embed instead of aborting
the sync, and exits 6 with `partial: true` and a `files_failed` list. AutoRAG
treats that as a completed sync (`minsync.ok: true`) and reports each skipped
file as a `minsync-file-skipped` warning diagnostic whose `source` is the path
inside the MinSync workspace. MinSync keeps the file's previous chunks and
retries it on the next sync. Only an exit 6 without a `partial` report is a
`minsync-sync-failed`.

## Migration and failure behavior

A MinSync workspace keeps its state in the `.minsync` subdirectory of
`<workspace>/.autorag/minsync`, and records the embedding identity alongside the
sync cursor and `config.toml`:

```text
<workspace>/.autorag/minsync/.minsync/autorag-embedding-identity.json
<workspace>/.autorag/minsync/.minsync/cursor.json
<workspace>/.autorag/minsync/.minsync/config.toml
```

An explicit `minSync.workspacePath` setting replaces the default
`<workspace>/.autorag/minsync` location; MinSync then keeps its `.minsync` state
under that configured path instead.

The identity includes provider, model, immutable model revision, dimension,
query prefix, passage prefix, and runtime build. A mismatch requires a full
reindex. Existing 768-dimensional Ollama/TEI stores must therefore be handled
by either:

- an explicit `autorag index reset --method minsync --yes` followed by
  `autorag refresh --method minsync`; or
- an explicit Ollama/TEI `minSync.embedder.id` and `baseUrl`, followed by
  refresh and identity validation.

A dimension mismatch is detected before vector reuse. MinSync reports a typed
semantic failure rather than mixing vector dimensions. When semantic
infrastructure is unavailable, the lexical/BM25 lane remains usable; hybrid
retrieval can fall back to BM25 and reports the degraded semantic diagnostic.

The legacy migration diagnostic is:

```text
This workspace uses the legacy 768-dimensional Ollama/TEI embedding path. Reindex explicitly, or pin an explicit Ollama/TEI embedder config before using the new default runtime.
```
