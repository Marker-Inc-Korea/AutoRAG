# MinSync setup

AutoRAG uses MinSync for local SPLADE sparse, semantic vector, and hybrid
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
yourself with `cargo install minsync --version 0.7.3 --locked` when
auto-install cannot run. MinSync on Apple Silicon and Linux needs rustc
1.96 or newer, because its dependency candle-core requires NEON f16 support;
with an older toolchain the cargo install fails and AutoRAG falls back to the
GitHub release binary. AutoRAG's installer pins MinSync 0.7.3 from crates.io.
A `minsync` older than 0.7.1 is never run; see
[Upgrading a BM25 workspace](#upgrading-a-bm25-workspace).

Existing executables are not silently replaced. New installations, including
GitHub release fallbacks, must pass the read-only protocol check before they
are accepted; an incompatible newly downloaded binary is removed from cache.

A normal setup is:

```bash
autorag init --search-paths /path/to/docs --workspace /path/to/workspace
autorag refresh --method parsed,minsync --json
autorag search --json "semantic question about the documents"
```

`autorag setup` and `autorag status` report which embedding backend is alive
(`daemon`, `in-process`, or `remote`) and the provider, model, and dimension,
or why none is and how to start one. Nothing here starts a daemon: see
[Running the embedding daemon](#running-the-embedding-daemon). A search or
query never starts indexing infrastructure either.

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
   loopback only and requires MinSync 0.4.12 or newer. It serves TEI
   `POST /embed`, OpenAI `POST /v1/embeddings`, and `GET /health`. Running
   instances are discovered through `~/.cache/minsync/serve/<instance>/`
   (`host`, `port`, `pid`, `lock`). **AutoRAG never starts or stops it for you**
   (see [Running the embedding daemon](#running-the-embedding-daemon)); it is
   looked up again on every call, so one you start or stop while AutoRAG runs
   is noticed.
3. **In-process engine** (the optional addon, not the `minsync` CLI). When no
   daemon answers, AutoRAG's own consumers (memory, the chat sidecars, gist,
   discrawl) can load MinSync's embedding engine inside the AutoRAG process
   through the Node-API addon `@nomadamas/minsync-native`. It runs the same
   engine and weights as `minsync`, so vectors are bit-for-bit identical to
   the daemon's. It is not bundled and not on npm yet, so today this step is
   effectively empty and the error says so. This is separate from the `minsync`
   CLI, which embeds in-process by itself for the MinSync datasource (it adopts
   a running daemon for `native:` workspaces, and `MINSYNC_SERVER=off` disables
   that).

### Query instruction and passage prefix

Qwen3-Embedding is instruction-aware: a search query is embedded as
`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:<query>`
and a document is embedded raw. AutoRAG applies this on every semantic path:

- MinSync's own index and search get it through `[embedder].query_prefix` in
  `.minsync/config.toml`, which MinSync prepends to the query only.
- AutoRAG-owned stores (chat sidecars, GitHub Gist) embed through the shared
  embedder with `role: "query"` for a search query and `role: "document"` for
  stored text. The shared embedder prepends the prefix itself; neither the
  daemon nor a remote server is asked to.
- mailcrawl receives the prefix through `MAILCRAWL_QUERY_PREFIX`.
- Hister (the browser-history datasource) keeps its own vector store, so it is
  pointed at the shared daemon through its own `config.yml` `semantic_search`
  block (`embedding_endpoint`, `embedding_model`, `dimensions`, `query_prefix`,
  `document_prefix`). `autorag setup --sync-hister` writes those keys from the
  resolved shared endpoint, byte for byte, so the query instruction above reaches
  Hister too. The daemon's port is chosen when it starts, so re-run
  `autorag setup --sync-hister` after restarting it. See
  [Browser history via Hister](datasource-skills.md#browser-history-via-hister).
- Similar-past-question lookup compares a question with questions, so both
  sides are embedded the same way, without the instruction.

The default applies only to MinSync's local Qwen3 (no `id`, or
`native:Qwen/Qwen3-Embedding-0.6B`, and no `baseUrl`). An operator-set
`minSync.embedder.queryPrefix` or `passagePrefix`, including `""`, always wins,
and a remote endpoint or another model gets no default prefix.

### Running the embedding daemon

Semantic search over everything except the MinSync datasource needs an
embedding backend that is alive: a remote endpoint, a running `minsync serve`,
or the optional in-process addon. Without one, memory, the chat-sidecar
semantic search, gist, discrawl, mailcrawl, lazykatok, and Hister fall back to keyword
search or report `embedding-unavailable`; the MinSync datasource keeps working
because `minsync` embeds in-process by itself. AutoRAG leaves the daemon's
lifecycle to you, so a remote endpoint is the way to avoid managing one:
it needs no local model and no process.

Without a remote endpoint, keep the daemon running. `--idle-timeout 0` serves
until stopped; the default of 300 s frees the ~2.8 GB model after five idle
minutes, and the next semantic call then fails until the daemon is started
again. Do not pass `--state-dir`: only the default state directory is
discovered.

**macOS (launchd).** Save as `~/Library/LaunchAgents/dev.autorag.minsync-serve.plist`
(replace `YOU`, and the path from `which minsync`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.autorag.minsync-serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/.cargo/bin/minsync</string>
    <string>serve</string>
    <string>--idle-timeout</string>
    <string>0</string>
    <string>--format</string>
    <string>json</string>
    <string>--quiet</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/YOU/Library/Logs/minsync-serve.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOU/Library/Logs/minsync-serve.log</string>
</dict>
</plist>
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.autorag.minsync-serve.plist
launchctl kickstart -k gui/$(id -u)/dev.autorag.minsync-serve   # restart
launchctl bootout gui/$(id -u)/dev.autorag.minsync-serve        # stop and unload
```

**Linux (systemd user unit).** `~/.config/systemd/user/minsync-serve.service`
with `ExecStart=%h/.cargo/bin/minsync serve --idle-timeout 0 --quiet` and
`Restart=always`, then `systemctl --user enable --now minsync-serve`.

**Without a service manager**, run `minsync serve --idle-timeout 0` in a
terminal, or press **Ctrl+Shift+E** or type `/embedding on` in `autorag tui` (below).

`autorag setup` and `autorag status` report whether a backend is alive and say
how to start one; the MCP status tool returns the same `embedding-unavailable`
diagnostic. They only look: nothing is started.

### The TUI footer, Ctrl+Shift+E, and /embedding

`autorag tui` shows the embedding state in the footer and re-checks it every
15 s and after each toggle:

| Footer | Meaning |
|---|---|
| `embedding: daemon ● up` | A `minsync serve` daemon answers `GET /health`. |
| `embedding: in-process ●` | No daemon, but the in-process addon is loadable. |
| `embedding: remote ● up` / `remote ○ down (…)` | The configured remote's `GET /health` (a 404 still counts as up, since hosted APIs have no such route; a network error or 5xx is down). |
| `embedding: off ○ (/embedding on, or ctrl+shift+e, starts minsync serve)` | Nothing can embed. |

**Ctrl+Shift+E** and **`/embedding`** do the same thing. The key toggles; the
command takes `on`, `off`, `toggle` (the default with no argument), or `status`
(a one-line report of the backend and why it is down, without changing
anything). The command works in terminals that do not deliver Ctrl+Shift
combinations and autocompletes its arguments. `on` starts the daemon
(`minsync serve --idle-timeout 0`, so it stays on until you stop it) and `off`
stops it with `SIGTERM`; each says so when there is nothing to do. With a remote
endpoint configured they only explain that there is no local daemon to manage,
and `status` still reports the remote's `/health`. A daemon started by a
launchd service is stopped too, and launchd then restarts it (`KeepAlive`); use
`launchctl bootout` to stop it for good.

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

## Sparse retrieval (SPLADE)

MinSync 0.6 replaced its BM25 full-text index with learned-sparse retrieval:
[nlpai-lab/KURE-Sparse-nano](https://huggingface.co/nlpai-lab/KURE-Sparse-nano)
(Apache-2.0) encodes every chunk and every query into weighted vocabulary terms.
The keyword half of AutoRAG's local search (the `sparse` mode, and the sparse
side of `hybrid`) runs through it. It needs no tokenizer library: Korean
inflected forms (`인증서를` → `인증서`) match without Kiwi, which AutoRAG no
longer probes for, and `minsync init` no longer takes `--language`.

MinSync downloads the 299 MB model into `~/.cache/minsync/models` on first use
(sha256-verified) and runs it in-process on Metal or CPU. AutoRAG does not
start a `minsync serve --sparse` daemon.

### Sparse-only index (no dense embedder)

Set `"minSync": { "embedder": { "id": "none" } }` to build a SPLADE-only index:
MinSync encodes sparse vectors and no dense ones, so no dense model is
downloaded or run (indexing a small corpus peaked at about 1.5 GB). Every
MinSync search is then a sparse search: the `hybrid` and `minsync` methods
query `--mode sparse` and describe themselves as `sparse`, an Obsidian vault
offers only `obsidian-sparse`, the shared embedder reports no backend (memory,
chat sidecars and gist keep their keyword paths), and neither `autorag status`
nor `autorag setup` warns about a missing embedder.

A dense index cannot become sparse-only in place, or back: the vector column's
dimension differs. When the configured embedder switches between `none` and a
dense one, the next refresh runs `minsync init --force` with the new embedder
and rebuilds the index with a full sync.

### Upgrading a BM25 workspace

AutoRAG requires MinSync 0.7.1 or newer. A `minsync` on `PATH` or in the
workspace cache that reports an older version is never run: one later on
`PATH` is used instead, a cached one is replaced by the next install (`autorag
refresh` with `minSync.autoInstall`), and a query with no usable binary fails
with a `MinSyncRequiredError` naming the version and path it found. A binary
whose `--version` does not print a MinSync version line (a custom build) is not
gated.

The first `autorag refresh` after the upgrade removes the `[lexical]` table
from `.minsync/config.toml` and MinSync migrates the index in place: the BM25
index and `lexical_text` column are dropped and sparse vectors are encoded for
every chunk; dense vectors are kept. This is a one-time cost that grows with
the corpus. On an Apple M4 Max, a 5,394-file workspace with 52,851 chunks took
38 minutes (45,361 chunks encoded, 7,490 reused) with a 2.7 GB peak memory
footprint, and the store grew from 0.65 GB to 2.3 GB.

### Measured quality

On that workspace (the maintainer's own Downloads/Documents/Desktop, mostly
Korean), 594 queries were generated from 297 sampled passages (one natural
question and one keyword query each), and recall counted a hit when the
passage's file, or a byte-identical copy, ranked in the top k:

| Mode | R@1 | R@5 | R@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| BM25 (`multilingual`, MinSync 0.5.1) | 0.473 | 0.732 | 0.811 | 0.583 |
| SPLADE sparse (MinSync 0.6.0) | 0.564 | 0.825 | 0.887 | 0.676 |
| Hybrid, BM25 + dense (0.5.1) | 0.481 | 0.779 | 0.848 | 0.607 |
| Hybrid, SPLADE + dense (0.6.0) | 0.492 | 0.779 | 0.847 | 0.614 |

For the natural-language questions, a model judged whether the top 5 chunks
contained the reference answer: BM25 0.657, SPLADE 0.838, hybrid 0.717 → 0.768.

Known limit (from MinSync's release notes, not measured here): exact strings
such as error codes, hashes, and version strings among many look-alike
documents match far less reliably than with BM25. Use a bounded `grep` for
exact lookups.

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

## Durable incremental parsing

Parsed mirrors require the **extended MinSync stage protocol (protocol 2)**,
including a refresh that selects only `parsed`. MinSync 0.5.1 and later — AutoRAG installs 0.7.3 — ship it; published binaries up to 0.5.0 have the original
durable stage, but not `capabilities`, `plan`, `inspect`, deliberate skips or
deferred-failure policy. AutoRAG probes capabilities before opening a run and
refuses an older build before mutating stage state. Parsing itself uses no
embedding model or endpoint.

`inspect` also exposes all held source identities, independently of output
verification. AutoRAG uses that inventory to preserve unknown/offline sources,
including damaged outputs and ordinary failures; only verified `items` become
searchable mirrors. Status remains available when a previously used binary is
missing or incompatible and reports the component unavailable and corpus stale.

AutoRAG inventories supported sources and supplies their identities and parser
configuration to one `minsync stage` session. MinSync owns the stage lock,
verified completions, deletion decisions, interrupted-run journal, and snapshot
under `.autorag/parsed/stage/`. AutoRAG owns parsing and atomic Markdown writes
under `.autorag/parsed/files/`. Completion is recorded only after the final
Markdown file exists and MinSync has measured and verified it.

- Unchanged, verified mirrors are not read or parsed again. Their document-level
  timestamps stay stable, so warm refreshes do not copy the entire corpus into
  the retrieval workspace again. The durable source revision uses nanosecond
  mtime and size; chmod, xattr, inode and device changes alone do not reparse it.
  A stronger stat identity is checked before committing within a run. As with
  other stat-based scans, a content edit preserving both source mtime and size
  between runs is not detected.
- Parser name/version and output-affecting options invalidate completed work.
  Execution deadlines, worker heap limits and OCR cache location only invalidate
  recorded failure skips, not healthy outputs. Language and OCR enablement changes
  still invalidate output. Stage keys are fixed-length SHA-256 IDs, independent
  of the length or encoding of the virtual source path.
- A source changed during parsing is not committed. Its last good mirror stays
  available, remains reported as stale, and is retried on the next refresh,
  including when the interrupted run's metadata must be recovered from MinSync.
- Missing or damaged mirrors are rebuilt. MinSync's fast verification hashes an
  output when its modification time changes; it does not detect deliberate edits
  that preserve both size and modification time.
- An interruption leaves committed items in MinSync's journal. The next matching
  refresh resumes them, including an interrupted forced refresh. There is no
  AutoRAG-specific periodic full-index checkpoint.
- AutoRAG no longer writes `.autorag/parsed/index.json` or reads MinSync's
  private snapshot/journal formats. Downstream source mappings come from the
  engine's verified `inspect` response, including committed work left by a hard
  host crash. `plan` supplies read-only freshness decisions without a stage lock.
- AutoRAG supplies source revisions, parser settings, omission reasons and retry
  policy identities. MinSync stores omissions and deferred failures, decides
  whether they are reusable, and releases failed work when its source, parser
  settings or retry policy changes, or an explicit retry is requested. Healthy
  outputs are unaffected by a retry-policy change. Failed work is never a
  successful completion. Unobservable roots and mass-deletion candidates are
  explicitly preserved rather than deleted.
- Existing workspaces without a parse-stage baseline are parsed once to establish
  verified completions. Subsequent refreshes use the stage baseline; the old
  checkpoint-based resume path is not retained.

The inventory and stage plan still hold the source listing in memory. This
cutover eliminates repeated whole-index writes during parsing; it does not make
the inventory or MinSync's stage-open request a streaming API.

## Refresh diagnostics

Parsed refresh reads only sources that need processing before MinSync indexes their mirrors.
A source deleted or replaced after the scan (`ENOENT`, `ENOTDIR`, or `EISDIR`)
records a `source-vanished` informational diagnostic. A source that cannot be
read (`EACCES` or `EPERM`) records a `source-unreadable` warning. The
diagnostic `source` is the virtual path, and the message is the underlying
filesystem error. Refresh removes any previous mirror for that path and
continues with the remaining files. Permission failures are paced using a skip
record until the source/parser changes or `--retry-skipped` is requested.
Any other read error, a parser that throws something other than `ParseError`, and
a failed mirror write are also isolated to the one file. Each is skipped and the
refresh continues:

| Code | Severity | Cause |
| --- | --- | --- |
| `source-read-failed` | warning | Read error not listed above (`EMFILE`, `EIO`, `EBUSY`, `ETIMEDOUT`, …); `reason` carries the errno code. |
| `source-changed` | warning | The source identity changed during parsing; no completion is committed and the next refresh retries it. |
| `parser-crashed` | warning | The parser threw a plain `Error` or non-`ParseError` value; `reason` carries the error name. The message is withheld because it may contain a path. |
| `mirror-write-failed` | warning | Writing the parsed mirror for this file failed; `reason` carries the errno code. |
| `cloud-placeholder-skipped` | info | A File Provider placeholder whose provider has no verified release path was not read; `reason` carries the provider. |
| `cloud-evict-failed` | warning | The file was indexed but its local copy could not be released after the retries; the message is the release helper's own error text and the attempt count. |
| `cloud-fingerprint-failed` | warning | A cloud or network file was indexed but its duplicate fingerprint could not be read or saved; it stays indexed and does not take part in exact-duplicate exclusion. |

`EMFILE`, `EBUSY`, `EIO`, and `ETIMEDOUT` are retried with doubling backoff
(50 ms, 100 ms, 200 ms) before the file is skipped. The retry count is the
`refresh.transientRetries` config key (default `3`, `0` disables retries):

```json
{ "refresh": { "transientRetries": 3 } }
```

Unlike permission-denied failures, these failures are recorded without a retry
deferral, so MinSync schedules them again on the next refresh. Mirror writes go
through a temp file removed on failure, so a failed write leaves no `*.tmp` orphan.

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

### Cloud and network folders

AutoRAG treats every location that is not a local, physical volume as a folder it must not open on its own: a
cloud sync folder, a network share, or a virtual filesystem. A refresh reads each such file once — a cloud
placeholder is downloaded within the budget below, a network file is read over the wire — releases the cloud
placeholder it downloaded, and takes the file's dupey fingerprint on that same read. The Jikji `prepare` pass
and the dupey exact-duplicate scan open file contents themselves, so both are kept away from these folders: a
released placeholder is never downloaded twice, and a network file is never read twice.

Every refresh detects this machine's non-physical roots from directory metadata only, never from file
contents. A helper that fails (no `mount`, no `xattr`, an unmounted share) contributes no roots rather than
failing the refresh.

| Root | Platform | Detected by |
| --- | --- | --- |
| File Provider mount (Google Drive, OneDrive, Dropbox, Box, …) | macOS | One directory per provider under `~/Library/CloudStorage` |
| iCloud Drive | macOS | The presence of `~/Library/Mobile Documents` |
| A File Provider–managed directory outside those two (the iCloud-synced `~/Desktop` and `~/Documents`) | macOS | `com.apple.file-provider-domain-id` on a search root, on one of its ancestors (the root is inside it) or on one of its direct children (it is nested under the root), read in batches with `xattr` |
| Network share (SMB/NFS/AFP), WebDAV mount, FUSE volume | macOS | A mount point from `/sbin/mount` whose options lack `local` and whose type is not `autofs` (the automounter is not a share), canonicalized to its real path |
| Cloud Files sync root (OneDrive, Dropbox, iCloud for Windows, …) | Windows | `HKCU\…\Explorer\SyncRootManager`, as before |
| Mapped network drive (`DriveType 4`) | Windows | `Win32_LogicalDisk`, one PowerShell process per refresh |
| FUSE-compatible volume (WinFsp, Dokan, rclone) | Windows | The disk's `FileSystem` starting with `FUSE`, `WinFsp`, or `Dokan` |
| UNC search root (`\\server\share\…`) | Windows | The search root's own share prefix; `\\?\…` and `\\.\…` device paths are not shares |

Both sides of every comparison are canonicalized, so a mount that macOS reports at its real path
(`/private/var/…` for a mount the caller made at `/var/…`) still matches the search root the user configured.

**Unverified.** No Windows host with *Google Drive for desktop* installed has been measured; its virtual drive
is recognized by a `GoogleDriveFS` filesystem or a `Google Drive` volume name, which is a documented guess.
*Synology Drive Client* modes have not been measured on either platform: its macOS File Provider mount is found
by the `~/Library/CloudStorage` listing, while a Synology virtual drive on Windows is not recognized at all.

**Behaviour.**

- Jikji is never run over a search root that is a cloud or network folder (or lies inside one): the root is
  skipped whole and reported once as the informational `jikji-cloud-root-skipped` diagnostic, and its
  documents are already indexed through the parsed mirror. A folder nested under a search root is excluded
  from the Jikji run by its relative path (`--exclude`), measured with jikji 0.2.1 to drop the whole subtree.
- dupey never scans a search root inside such a folder and excludes a nested one by its name
  (`--exclude-dir`, measured to skip reads under it entirely). Files in these folders join the exact-duplicate
  exclusion through the stored fingerprints below.
- A network file is read normally: no download budget and no release, because no local copy remains to give
  back.
- A cloud or network file is read once per change and fingerprinted on that same read; an unchanged file is
  never read again, and an unchanged refresh issues no content read at all.
- A root that is currently offline (an unmounted share, a signed-out provider) is never deleted: the refresh
  reports `search-root-unavailable` and leaves its previously indexed documents untouched, so the mirror
  survives until the root is back.

### Connecting cloud storage and migrating from the `cloud-drive` datasource

Cloud storage is connected by adding the provider's sync folder to
`searchPaths`; AutoRAG keeps no separate cloud datasource and no `rclone`
credential boundary. `autorag setup` reports every cloud or network folder the
machine has that no `searchPaths` entry covers, with the exact command to add
it.

The removed rclone-backed `cloud-drive` datasource is not migrated or shimmed: a
config entry that still names it is skipped with the `unknown-datasource-skill`
warning. Remove it from `datasources`, add the sync folder above, then delete
the old `.autorag/datasources/<name>/` mirror data (`mirror/`, `manifest.json`,
`chunks.json`) — AutoRAG no longer reads it.

Storage that has no desktop sync app can be mounted instead: `rclone mount
<remote>: <mountpoint>` exposes a remote as a local folder, or `rclone serve
webdav <remote>:` plus an OS WebDAV mount (`mount_webdav` on macOS, a mapped
drive on Windows) does the same. Mounted this way the folder is a network
folder: each file is read over the wire, with no download budget and nothing to
release.

Google Docs, Sheets, and Slides are a known gap: in a sync folder they appear
only as small `.gdoc`/`.gsheet`/`.gslides` pointer files with no document
content, so they are not indexed beyond their names. Whether `rclone mount` /
`rclone serve webdav` exports them as readable files is unverified.

### Cloud folders (macOS)

A folder synced by a File Provider app (iCloud Drive, Google Drive, OneDrive under
`~/Library/CloudStorage` or `~/Library/Mobile Documents`) lists every file, but a
file you have not opened is a placeholder: its data is only in the cloud, and
reading it downloads it. A refresh looks at the file's `st_flags` (`SF_DATALESS`)
before reading and treats the file according to whether AutoRAG can give the local
copy back.

For **iCloud Drive, Google Drive, OneDrive, and Dropbox** — all measured on macOS 26 — a
refresh downloads a placeholder within the budget below, parses and indexes it, then
releases the local copy again with one Foundation call
(`NSFileManager.evictUbiquitousItem`, run through `osascript -l JavaScript`; no helper
binary, app, or entitlement is needed). The file stays in the cloud and keeps its size
and modification time, so an unchanged file is never downloaded again; a changed one
is downloaded once and released again. Files already on disk are read as before and
never released.

Any **other File Provider** has no verified release path, so a placeholder is not read
(`cloud-placeholder-skipped`, `reason` is the provider): AutoRAG cannot give the local
copy back and would otherwise leave every document on disk. A previously indexed copy is
kept. Make the file or folder available offline in the provider's app to index it.

Downloads are bounded per batch: a placeholder that would exceed the batch's budget ends
the batch, which is handed to MinSync (searchable once that sync reaches it), and is read
first in the next batch with a fresh budget, so every placeholder is indexed in the same
refresh. Each downloaded file is released right after it is parsed, before its batch is
handed to MinSync. A refused release is
retried with doubling waits (500 ms, 1 s, 2 s); the provider briefly holding a
just-downloaded file is the usual cause. A file still uploading to the cloud cannot be
released yet — the refusal names that state. When the retries are spent the file is
left on disk, the document stays indexed, and one `cloud-evict-failed` records the
helper's own error text and the attempt count. Giving up is final: an unchanged file is
not processed again, so it stays on disk until its content changes. A read that fails
per file (a OneDrive placeholder has timed out) is a `source-read-failed` skip, never a
failed refresh.

The agent's own tools open originals too. When the indexed text is not enough, the model
may read the file with `bash` or `read`, which hydrates one placeholder. AutoRAG watches
the absolute paths a run's tool calls name (including the direct children of a directory a
call names, capped at 500) and the absolute sources its retrieval results print, and
releases every one that became local when the run ends, with the same retry-and-backoff
as a refresh release. A file that was already on disk before the run is never released,
and a release that keeps failing is logged once and never fails the answer. The watch is
deliberately narrow: it does not see a file the agent opens through a path it never
printed, such as a relative name after `cd` into a subfolder deeper than the direct
children of a directory the call named.

A file written on this Mac is downloaded one extra time. Its first hydrate rounds the
modification time to the precision the cloud keeps (measured on macOS 26: Google Drive
`+101 ns`, OneDrive `+77 ns`, iCloud `+110 ns`, Dropbox `-24 ns`), so the next refresh sees the placeholder
as changed and downloads it once more. After that first hydrate the value is stable and
later refreshes skip the file. A file that already exists in the cloud has the cloud's
rounded time from the start and is not re-downloaded.

```json
{ "refresh": { "cloudMaxFiles": 200, "cloudMaxBytes": 536870912, "cloudEvictAfterParse": true, "cloudEvictRetries": 3 } }
```

`cloudMaxFiles` (default `200`) and `cloudMaxBytes` (default 512 MiB) bound what one
batch downloads; a batch's first placeholder is always allowed, so one large file cannot starve.
`cloudEvictAfterParse: false` keeps downloaded cloud files on disk. `cloudEvictRetries` (default `3`, `0` tries once) is the number of retries after the first failed release. These
keys apply on Windows too (below); Linux reads every file as before.

Two other readers open file contents themselves, so each one used to download a released
cloud file again on every refresh: the Jikji `prepare` pass (`jikji`) and the
exact-duplicate scan (`dupey`, `excludeExactDuplicates`). Both are now aware of cloud
and network folders (see *Cloud and network folders* above) and keep their defaults on:

- **Jikji** is never run over a search root that is a cloud or network folder (or lies inside one).
  Such a root is skipped whole and reported once as the informational
  `jikji-cloud-root-skipped` diagnostic; its documents are already indexed through the
  parsed mirror. A folder nested under a search root is excluded from the Jikji run
  by its relative path (`--exclude`), measured to drop the whole subtree with jikji 0.2.1.
- **dupey** never scans a search root inside such a folder, and excludes a nested folder
  under one by its name (`--exclude-dir`, measured to skip reads under it entirely).
  A cloud or network file instead joins the exact-duplicate exclusion through a **stored fingerprint**:
  while the file is on disk during its single read (before a downloaded placeholder is released), a refresh
  runs `dupey fingerprint` on it once and keeps `{sourceVersion, exact, modified, revision}`
  under `.autorag/cloud-fingerprints.json` in the workspace (an AutoRAG-owned store; the
  MinSync stage schema is unchanged). An unchanged file keeps that fingerprint without being
  read; a changed file gets a new one; a removed file drops it. A file whose provider has no
  verified release path is never read, so it is never fingerprinted. A missing dupey or a
  fingerprint failure never fails the refresh: the document stays indexed and one
  `cloud-fingerprint-failed` warning is recorded.

Exact-duplicate exclusion then groups the cloud fingerprints with the scanned, non-cloud
files by exact hash and keeps one keeper — the newest by dupey's own ranking for scanned
files, and the fingerprint's recorded `modified` time for a cloud file (falling back to a
filesystem `stat`, which reports metadata without downloading the placeholder). The other
copies are excluded through the normal `duplicate-excluded` path. A file dupey extracts no
text from (an image-only PDF, for one) is grouped by dupey's hash of its bytes, so only
byte-identical copies match. Because the fingerprints
are written during a refresh, a cloud file first fingerprinted this refresh can be excluded
on the *next* one; that is expected and still never reads the released original. The parsed
mirror, MinSync, and fsearch passes never re-download.

AutoRAG requires **dupey 0.2.0 or later** (`cargo install dupey`) and refuses an older binary
with a message naming its version: dupey 0.1.2 hashed incomplete text for HWP documents and
empty text in `fingerprint` for files without text, so its hashes disagree with 0.2.0's. A
store written by 0.1.2 (version 1) keeps every fingerprint except its HWP files and its files
without text; those take part in exact-duplicate exclusion again once they change and are
fingerprinted by 0.2.0.

### Cloud Files folders (Windows)

Windows sync apps (OneDrive, Dropbox, iCloud for Windows, and others built on the Cloud Files API) keep
online-only files as placeholders that carry `FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS` (`0x400000`);
`FILE_ATTRIBUTE_OFFLINE` is not set on them. A refresh reads the attributes of the files it is about to
parse in one PowerShell process per batch, downloads placeholders within the same budget as on macOS,
parses and indexes them, then releases the local copy with `CfDehydratePlaceholder` (one PowerShell
process per file). The file's pin state is not changed.

Release is enabled for every Cloud Files sync root because Windows itself refuses the two unsafe cases:
a file with local changes the provider has not synced (`0x80070179`) and a file pinned to stay on the
device (`0x80070188`); either is reported in `cloud-evict-failed` and the file stays as it is. This was
measured against a synthetic Cloud Files provider (`scripts/manual-qa/windows-cloud-files-provider.ps1`)
on Windows 11; OneDrive, Dropbox, and other commercial providers have not been measured.

Windows lets a third-party process download an online-only file only where it can show its "app is
downloading files" notification. An AutoRAG refresh running without an interactive desktop (a service,
or an SSH session in session 0) gets `ERROR_CLOUD_FILE_ACCESS_DENIED` (395) for every placeholder; those
files are skipped with `source-read-failed` and nothing is released. Run the refresh in the desktop
session, and if the notification offers to block the app, allow it (Settings, Privacy & security,
Automatic file downloads).

Beyond the Cloud Files registry, a Windows refresh also treats a mapped network drive (`DriveType 4`), a
FUSE-compatible volume (a `FileSystem` starting with `FUSE`, `WinFsp`, or `Dokan`), and a search root that is
itself a UNC path (`\\server\share\…`) as a cloud or network folder, so Jikji and dupey never read them. The
disk list comes from one `Win32_LogicalDisk` PowerShell process per refresh; a failure contributes no roots
instead of failing the refresh. See *Cloud and network folders* above for what is unverified — in particular
Google Drive for desktop's virtual drive (recognized by a `GoogleDriveFS` filesystem or a `Google Drive`
volume name, never measured) and Synology Drive Client's Windows mode (not recognized).

Jikji and dupey keep their placeholders released on Windows the same way as on macOS: Jikji skips a
search root inside a registered sync root, dupey excludes nested sync roots by name and judges cloud
files from their stored fingerprints. This part is verified by unit tests only; no commercial Windows
sync provider has been measured.

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

## Indexing progress and liveness

AutoRAG runs `minsync sync --progress jsonl`. Each MinSync `start`, `batch`,
`heartbeat`, and `done` event updates `.autorag/refresh-progress.json` with
`minsync: { synced, total }`, advances `updatedAt`, and emits the same counts to
`AutoRAGAgent.refresh(..., { onProgress })` as a `minsync` event. MinSync's
heartbeat interval defaults to five seconds.

A stale `updatedAt` between events means that the phase is still running; it is
not proof of a hang. Use the owner pid through
`AutoRAGAgent.getRefreshStatus().progress.ownerAlive` (or
`isRefreshOwnerAlive`) for liveness, and use advancing progress to distinguish
active work from a stalled batch.

## Skipped files (partial sync)

MinSync 0.4.11 and newer skip a file they cannot read, chunk, or embed instead of
aborting the sync, and exit 6 with `partial: true` and a `files_failed` list. AutoRAG
treats that as a completed sync (`minsync.ok: true`) and reports each skipped
file as a `minsync-file-skipped` warning diagnostic whose `source` is the path
inside the MinSync workspace. `AutoRAGRefreshResult.minsync.filesSkippedCount`
and the CLI JSON envelope expose the count, while the completed
`refresh-progress.json` records it as `minsync.filesSkipped`; human refresh and
status output show `skipped=N` and `filesSkipped=N` respectively. MinSync keeps
the file's previous chunks and retries it on the next sync. A later successful
retry omits the skipped count again. Only an exit 6 without a `partial` report
is a `minsync-sync-failed`.

## Migration and failure behavior

During a MinSync-backed refresh, three consecutive unsuccessful sync runs with
the same structured reason and diagnostic code stop further MinSync work. Missing
fields compare consistently; changing only the error message does not reset the
count. A different reason or code starts a new streak, and any successful sync
(including one with per-file skips) clears it. A held index lock still stops
immediately.

Parsing overlaps indexing, so in-flight parsing may finish its current batch
before observing the stop at the next applicable committed-batch callback. There
is no final retry after the threshold. Committed mirrors and the durable stage
journal remain available: fix the reported cause and run `autorag refresh --method
minsync` again to resume incrementally without reparsing unchanged committed
sources. Deletions remain deferred until the final parse batch.

A systemic stop reports one MinSync component failure in each diagnostic list,
preserving the terminal cause and original error text, including stderr, exit
status, and paths. Other failed runs retain at most 100 per-file sync warnings
plus one summary with the omitted count and the normal component diagnostic.
A later successful sync clears these warnings and the omitted count. Parser,
datasource, and partial-sync per-file diagnostics are reported separately.

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
query prefix, and passage prefix. A mismatch in any field except the query
prefix requires a full reindex; the query prefix only shapes query vectors, so
changing it is recorded without re-embedding documents. Existing
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
