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
yourself with `cargo install minsync --version 0.5.1 --locked` when
auto-install cannot run. MinSync 0.5.1 on Apple Silicon and Linux needs rustc
1.96 or newer, because its dependency candle-core requires NEON f16 support;
with an older toolchain the cargo install fails and AutoRAG falls back to the
GitHub release binary. AutoRAG's installer pins MinSync 0.5.1 from crates.io.

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

## Durable incremental parsing

Parsed mirrors require the **extended MinSync stage protocol (protocol 2)**,
including a refresh that selects only `parsed`. MinSync 0.5.1 — the version
AutoRAG installs — ships it; published binaries up to 0.5.0 have the original
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
| `cloud-placeholder-skipped` | info | A Google Drive, OneDrive, or other File Provider placeholder was not read; `reason` carries the provider. |
| `cloud-download-deferred` | info | A cloud placeholder is over this refresh's download budget and waits for the next refresh. |
| `cloud-evict-failed` | warning | The file was indexed but its local copy could not be released after the retries; the message is the last provider error (`brctl` on macOS, the Cloud Files API on Windows) and the attempt count. |

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

### iCloud Drive and other cloud folders (macOS)

A folder synced by a File Provider app (iCloud Drive, Google Drive, OneDrive under
`~/Library/CloudStorage` or `~/Library/Mobile Documents`) lists every file, but a
file you have not opened is a placeholder: its data is only in the cloud, and
reading it downloads it. A refresh looks at the file's `st_flags` (`SF_DATALESS`)
before reading and treats the three cases differently:

| Provider | What a refresh does with a placeholder |
| --- | --- |
| iCloud Drive | Downloads it within the budget below, parses and indexes it, then releases the local copy with `brctl evict`. The file stays in iCloud and keeps its size and modification time, so an unchanged file is never downloaded again; a changed one is downloaded once and released again. |
| Google Drive, OneDrive, other File Providers | Does not read it (`cloud-placeholder-skipped`, `reason` is the provider), because AutoRAG has no way to release the copy again and would otherwise leave every document on disk. A previously indexed copy is kept. Make the file or folder available offline in the provider's app to index it. |

Files already on disk are read as before and never evicted. Downloads per refresh
are bounded; the rest wait for the next refresh (`cloud-download-deferred`, one
diagnostic per deferred file). A failed `brctl evict` is retried with doubling
waits (500 ms, 1 s, 2 s; `Resource busy ... currently in use by an application` is
common right after a download and usually clears). When the retries are spent the
file is left on disk, the document stays indexed, and one `cloud-evict-failed`
records the command's own error text and the attempt count. Giving up is final:
an unchanged file is not processed again, so it stays on disk until its content changes.

```json
{ "refresh": { "cloudMaxFiles": 200, "cloudMaxBytes": 536870912, "cloudEvictAfterParse": true, "cloudEvictRetries": 3 } }
```

`cloudMaxFiles` (default `200`) and `cloudMaxBytes` (default 512 MiB) bound what one
refresh downloads; the first file is always allowed, so one large file cannot starve.
`cloudEvictAfterParse: false` keeps downloaded cloud files on disk. `cloudEvictRetries` (default `3`, `0` tries once) is the number of retries after the first failed release.
Linux hosts read every file as before; only macOS and Windows have a cloud probe.

Two other readers open file contents themselves and download a released iCloud file
again, so with their defaults the local copy comes back after the refresh: the Jikji
`prepare` pass (`jikji`) and the exact-duplicate scan (`dupey`, `excludeExactDuplicates`).
Measured on macOS 26 with iCloud Drive, a refresh with both switched off
(`"jikji": false`, `"excludeExactDuplicates": false`) left every released file released;
with either one on, they were downloaded again. The parsed mirror, MinSync, and fsearch
passes never re-download. Making Jikji and dupey placeholder-aware is tracked separately.

### OneDrive and other Cloud Files providers (Windows)

A folder synced by a Cloud Files provider (OneDrive, and any other provider that
registers a sync root under `HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\SyncRootManager`)
lists every file, but a file you have not opened is a placeholder whose bytes live
only in the cloud. A refresh reads `FileAttributes` for a batch of files in one
PowerShell process and treats a file whose `FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS`
(`0x400000`) bit is set as a placeholder. `FILE_ATTRIBUTE_OFFLINE` (`0x1000`) is
*not* set on real placeholders, so it is not required.

For each placeholder the probe names its provider from the registered sync root
that contains the path (falling back to the path itself):

| Provider | What a refresh does with a placeholder |
| --- | --- |
| OneDrive | Downloads it within the budget below, parses and indexes it, then releases the local copy. The file stays in OneDrive and keeps its size and modification time, so an unchanged file is never downloaded again; a changed one is downloaded once and released again. |
| Google Drive, Dropbox, other providers | Does not read it (`cloud-placeholder-skipped`, `reason` is the provider), until that provider's release path has been verified live. A previously indexed copy is kept. Make the file or folder always available in the provider's app to index it. |

A release opens the file with a Cloud Files oplock (`CfOpenFileWithOplock`) and
calls `CfDehydratePlaceholder` synchronously, so the refresh learns the outcome
immediately. Dehydration is provider-agnostic and does not change the file's pin
state; it refuses a file the provider has not finished uploading, which is
reported as a per-file `cloud-evict-failed` and leaves the document indexed. The
`cloudMaxFiles`, `cloudMaxBytes`, `cloudEvictAfterParse`, and `cloudEvictRetries`
keys above apply unchanged; the deferred and retry behavior is the same as macOS.

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
