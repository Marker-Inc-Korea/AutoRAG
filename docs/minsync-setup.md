# MinSync setup

AutoRAG uses MinSync for local lexical BM25, semantic vector, and hybrid
retrieval over parsed document mirrors. MinSync remains the owner of its
workspace, chunks, vectors, cursor, and embedding identity.

## MinSync owns the embedding model

AutoRAG owns no embedding model, no llama.cpp, and no GGUF assets, and it does
not download, cache, import, or verify model weights. MinSync provisions
Qwen3-Embedding-0.6B (1024 dimensions) in its own cache at
`~/.cache/minsync/models` and runs the engine. No Ollama installation, TEI
adapter, API key, or endpoint is required for the default local path.

MinSync is required. AutoRAG auto-installs its binary when no usable executable
is found on `PATH` or in the workspace cache. If the binary is missing or its
auto-install fails, `autorag search`, `autorag refresh`, `autorag lite retrieve`,
and the MCP tools fail with a `MinSyncRequiredError` message and a non-zero exit;
AutoRAG never answers from the other retrieval paths instead. `minSync: false`
and `minSync.enabled: false` are rejected as config errors. Install the binary
yourself with `cargo install minsync` when auto-install cannot run. MinSync
0.4.11 on Apple Silicon and Linux needs rustc 1.96 or newer, because its
dependency `candle-core` requires NEON f16 support; AutoRAG's installer pins
MinSync 0.4.11.

A normal setup is:

```bash
autorag init --search-paths /path/to/docs --workspace /path/to/workspace
autorag refresh --method parsed,minsync --json
autorag search --json "semantic question about the documents"
```

`autorag refresh`, `autorag watch`, and `autorag setup` start a loopback
`minsync serve` daemon when none answers. A search or query never starts one:
it reads the prebuilt indexes and starts no indexing infrastructure.
`autorag setup` reports the resolved backend (`daemon`, `in-process`, or
`remote`), whether it started a daemon, and the provider, model, and dimension.

## Embedding backends

One shared embedder serves everything that embeds inside AutoRAG: MinSync
indexing and query, memory similar-question search, the chat-crawler sidecars
(Slack, Telegram, WhatsApp, Kakao), the github-gist vector index, and the
endpoint external CLIs such as discrawl are pointed at. AutoRAG resolves it in
this order:

1. **Remote endpoint.** If `minSync.embedder` names a remote endpoint, that
   endpoint is used for everything. No daemon is started and no local model is
   loaded. `id` names the protocol and model (`openai:<model>`, `tei:<model>`,
   or any id together with `baseUrl`). OpenAI uses
   `POST {baseUrl}/v1/embeddings`; TEI uses `POST {baseUrl}/embed`. `openai:`
   without a `baseUrl` means `https://api.openai.com`. Optional fields:
   `apiKeyEnv` (the name of an environment variable holding the bearer token,
   never the token), `dimension`, `batchSize`, and `timeoutMs`.
2. **`minsync serve` daemon.** Otherwise a running daemon is preferred. It is
   loopback only and requires MinSync 0.4.11 or newer. It serves TEI
   `POST /embed`, OpenAI `POST /v1/embeddings`, and `GET /health`, and exits
   after 300 s idle by default. Running instances are discovered through
   `~/.cache/minsync/serve/<instance>/` (`host`, `port`, `pid`, `lock`).
   `autorag refresh`, `autorag watch`, and `autorag setup` start it detached
   when none answers; a search never does.
3. **In-process engine.** When no daemon answers and none could be started,
   AutoRAG-owned consumers use MinSync's in-process engine through the optional
   Node-API addon `@nomadamas/minsync-native`. It runs the same engine and
   weights, so vectors are bit-for-bit identical to the daemon's. The addon is
   loaded lazily and is not bundled; when it is absent, the error says so. The
   `minsync` CLI independently adopts a running daemon for `native:`
   workspaces and otherwise falls back in-process (`MINSYNC_SERVER=off`
   disables adoption).

The reported identity is provider `minsync`, model
`Qwen/Qwen3-Embedding-0.6B`, and dimension `1024`.

### Rebuilding after the old embedding gateway

Daemon and in-process vectors are identical, so stores keyed on that identity
do not re-embed when the daemon appears or disappears. The vectors from the
removed llama-server runtime are not compatible.

- Memory vector stores (`memory.json.vectors.db`), chat-sidecar semantic
  stores, and the github-gist `vectors.json` re-embed automatically on the next
  refresh because their recorded identity changed.
- A MinSync workspace created through the old gateway profile
  (`minSync.embedder.id: "tei:Qwen3-Embedding-0.6B-Q8_0.gguf"`, or an old
  `profile` field) must be rebuilt explicitly:

  ```bash
  autorag index rebuild --method minsync
  ```

  `autorag index reset --method minsync --yes` followed by
  `autorag refresh --method minsync` has the same effect. Legacy `profile` and
  `.gguf` ids in `config.json` are ignored on read, without an error.

## Excluding files and folders

Exclude generated workspaces, archives, or individual files before the first
refresh by repeating `--exclude`:

```bash
autorag init --search-paths /path/to/docs \
  --exclude "/path/to/docs/Build output" \
  --exclude "/path/to/docs/Archive, old"
autorag refresh
```

Each `--exclude` argument is one complete path; spaces and commas are preserved.
Relative CLI paths resolve from the current directory, even when `--workspace`
points elsewhere. The generated config stores absolute paths.

For an existing config, add or edit `excludePaths`, preserving the other settings:

```json
{
  "searchPaths": ["/path/to/docs"],
  "excludePaths": ["/path/to/docs/Build output", "/path/to/docs/private-notes.txt"]
}
```

Each entry excludes that exact file or directory and all descendants of an
excluded directory. Entries are literal paths, not glob patterns. Relative paths
in JSON resolve against the configured `workspacePath`, or the config file's
directory when `workspacePath` is absent.

Run `autorag refresh` after editing the config to remove excluded documents from
the parsed mirror and update the indexes. Native fsearch and Everything results
are filtered before pagination, including exact file exclusions and stale index
entries. Native filename indexes may retain metadata for individually excluded
files; the result filter prevents those entries from appearing in searches.
The fsearch filesystem fallback applies exclusions before walking a subtree, so
excluded contents do not consume its traversal budget.

## MinSync model cache

MinSync provisions Qwen3-Embedding-0.6B at 1024 dimensions in
`~/.cache/minsync/models`. The daemon and the in-process addon read that cache;
AutoRAG makes no model download of its own. There is no `AUTORAG_HOME` model or
runtime cache and no AutoRAG prefetch, import, or verify command for embedding
weights: those commands were removed with the old runtime. `autorag models
list` lists chat models only.

A 768-dimensional workspace cannot be kept by choosing a profile: reindex for
the default MinSync identity, or configure an explicit Ollama/TEI endpoint
(`minSync.embedder.id` and `baseUrl`, or the
`--embedder-id`/`--embedder-base-url` init flags documented below). AutoRAG
writes only the supported MinSync embedder fields and does not write secrets.
The weights are MinSync's dependency and are not part of the AutoRAG npm
package.

## BM25 tokenizer

MinSync tokenizes text before indexing it and before every BM25 query. Its
default, `simple`, splits on whitespace only, so a Korean word with a particle
(`인증서를`) never matches a query for its stem (`인증서`), and `환불정책과` never
matches `환불`.

AutoRAG always sets MinSync's `[lexical].language` to `multilingual`, whatever
the global `languages` setting says; `languages` only drives OCR. `multilingual`
indexes the original whitespace-separated words together with the morphemes
from Kiwi (Korean) and the other analyzers' tokens, so an exact term such as a
compound (`연구목표`) and an inflected form (`인증서를`) both match, and Latin
words stay case-insensitive. MinSync's `ko` preset indexes the morphemes only:
a compound then exists only as its pieces, and exact-term queries lose to
documents that merely contain both pieces. AutoRAG never selects `ko`.

Measured on a sample of real Korean documents (410 documents, BM25 only, queries
sampled from the documents themselves, top-5 precision / top-50 recall):

| Queries | `simple` | `ko` | `multilingual` |
| --- | --- | --- | --- |
| 60 stems that also occur with particles or endings | 0.18 / 0.10 | 0.45 / 0.67 | 0.54 / 0.77 |
| 40 exact words and compounds | 0.97 / 0.93 | 0.59 / 0.79 | 0.89 / 0.98 |

Tokenizing is slower than whitespace splitting: indexing 5,456 chunks took
about 36 s with `multilingual` against 2 s with `simple`.

### Kiwi is required

`multilingual` loads the [Kiwi](https://github.com/bab2min/Kiwi) native library
and its base model at run time. Neither ships with MinSync or AutoRAG, and
without them every sync fails. AutoRAG therefore uses `multilingual` only when
it finds both, by the same rules as MinSync's `kiwi-rs` loader:
`KIWI_LIBRARY_PATH` and `KIWI_MODEL_PATH` when set (an explicit path is never
second-guessed), otherwise the per-user install `~/.local/kiwi`
(`%LOCALAPPDATA%\kiwi` on Windows) and the usual system directories
(`/usr/local`, `/usr`, `/opt/homebrew`). A library that only the OS loader's own
search path can find is not probed; set `KIWI_LIBRARY_PATH` for it.

When Kiwi is missing, a new workspace is created with `simple` and `autorag
refresh` reports a `lexical-tokenizer-unavailable` warning that names what to
install, instead of failing. This applies to every corpus, including an
English-only one. The next refresh after Kiwi is installed switches the
workspace to `multilingual`. A workspace that is already on `multilingual` is
never switched back on a missing Kiwi, since that would re-embed every chunk;
its sync reports MinSync's own error instead.

### Changing the tokenizer

A new workspace is initialized with `multilingual`. For an existing workspace,
`autorag refresh` writes it to `[lexical].language` in `.minsync/config.toml`.
MinSync records the tokenizer an index was built with in its cursor; when the
two differ, its next `sync` deletes and re-indexes every chunk, re-embedding
them as well, not only rebuilding the BM25 lane. A refresh that fails or is
cancelled restores the previous `config.toml`, so the rebuild is retried by the
next refresh. Every workspace created before this change used `simple`, so
upgrading re-indexes it once (when Kiwi is present).

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
fallback from the MinSync embedder. AutoRAG never silently switches an existing
vector store between Ollama/TEI and MinSync's local embedder.

Direct Ollama OpenAI-compatible embedding endpoints are also operator-managed
legacy configuration, not the AutoRAG default. An explicit external
`minSync.embedder` endpoint remains authoritative and is not overwritten by the
daemon or the in-process engine.

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

MinSync 0.4.11 skips a file it cannot read, chunk, or embed instead of aborting
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
query prefix, and passage prefix. A mismatch requires a full reindex. Existing
768-dimensional Ollama/TEI stores, and MinSync workspaces created through the
old embedding gateway, must therefore be handled by either:

- an explicit `autorag index reset --method minsync --yes` followed by
  `autorag refresh --method minsync`; or
- an explicit Ollama/TEI `minSync.embedder.id` and `baseUrl`, followed by
  refresh and identity validation; or
- `autorag index rebuild --method minsync` for a workspace whose identity came
  from the old embedding gateway (see [Embedding backends](#embedding-backends)).

A dimension mismatch is detected before vector reuse. MinSync reports a typed
semantic failure rather than mixing vector dimensions. When semantic
infrastructure is unavailable, the lexical/BM25 lane remains usable; hybrid
retrieval can fall back to BM25 and reports the degraded semantic diagnostic.

An identity mismatch is reported as the `embedding-identity-mismatch`
diagnostic; the fix is a full reindex of the affected store
(`autorag index rebuild --method minsync`).
