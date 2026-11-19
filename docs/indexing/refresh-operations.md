---
title: "Refresh operations"
description: "Refresh diagnostics and codes, the mass-deletion valve, cancellation, progress, skipped files, and retry and failure behavior."
---

# Refresh operations

This page documents what `autorag refresh` reports when part of a run fails, how it protects against losing documents, how it is cancelled, and how it records progress and retries.

## Running a refresh

```bash
autorag refresh --method parsed,minsync --json
```

`--force` re-indexes even when the mirror looks unchanged, and also lets the mass-deletion valve delete what it would otherwise suppress. `--retry-failed` retries files whose parse or read failed earlier, once, even if they are unchanged. The command is model-free and exits 0 on success, 2 on a config error, and 1 on a runtime error. `SIGINT`/`SIGTERM` exit 130/143.

## Searching while a refresh runs

Search is not blocked by a refresh. `minsync query` takes no lock and reads the index as of the moment it opens, so a search issued while `AutoRAGAgent.refresh` is updating an index that already exists answers from what is committed, and AutoRAG runs it beside the refresh instead of refusing it. The first index works the same way: the batches it has committed so far answer while the rest are still being read. Searches run one at a time per workspace, and a sync never starts under a running search.

Two things are worth knowing:

- A search is refused (`another sync is in progress`, reported as a `retrieval-method-failed` diagnostic) only while the index has nothing to read yet (no `cursor.json` and no `txn.json`), for example the first moments of a first sync or a forced rebuild, or while a sync follows an interrupted first sync, whose rows may belong to another embedder. This is why AutoRAG requires MinSync 0.7.4: from that version every query is checked against the embedder recorded in the index's `cursor.json`, so a search that reads beside a rewrite of a built index cannot score against another model's vectors.
- During an incremental refresh an **edited** document can briefly miss the lexical half of a search while its batch is being swapped. MinSync replaces an edited batch's dense rows first and encodes the new sparse vectors a moment later; MinSync 0.7.4 sweeps the old rows only after the new ones are sparse-encoded, so an edited document answers throughout. `scripts/manual-qa/run-qa-refresh-search-live.ts` runs this against a real index and fails if a search is refused or an edited document is never found during the refresh.

## Refresh diagnostics

Every diagnostic carries a stable code, a severity (`info`, `warning`, or `error`), a message, and a `source`. Human output prints diagnostics when the run did not fully succeed, or always with `--debug`; the JSON envelope always carries them.

Per-file parse and source diagnostics (an error in one file never stops the run):

| Code | Severity | Cause |
| --- | --- | --- |
| `source-vanished` | info | The source was deleted or replaced after the scan (`ENOENT`, `ENOTDIR`, `EISDIR`). Its mirror is removed. |
| `source-unreadable` | warning | The source could not be read (`EACCES`, `EPERM`). Paced with a skip record until the source, parser, or retry policy changes. |
| `source-read-failed` | warning | A read error not listed above (`EMFILE`, `EIO`, `EBUSY`, `ETIMEDOUT`, …); `reason` carries the errno code. |
| `source-changed` | warning | The source identity changed during parsing; no completion is committed and the next refresh retries it. |
| `parser-failed` | warning | kordoc, spreadsheet, CSV, PPTX, or EML parse failed, timed out, overran its heap, or its worker crashed. |
| `parser-crashed` | warning | The parser threw a plain `Error` or a non-`ParseError` value; `reason` carries the error name. The message is withheld because it may contain a path. |
| `parser-skipped` | warning | The source exceeds the parse-size cap (default 50 MiB), or an image is too small to OCR. |
| `parser-warning` | warning | A parser emitted a non-fatal warning (kordoc warnings, a truncated sheet). |
| `mirror-write-failed` | warning | Writing the parsed mirror failed; `reason` carries the errno code. |
| `duplicate-excluded` | info | An older exact duplicate was excluded; the newest copy remains indexed. |
| `cloud-placeholder-skipped` | info | A File Provider placeholder whose provider has no verified release path was not read; `reason` carries the provider. |
| `cloud-evict-failed` | warning | The file was indexed but its local copy could not be released after the retries; the message carries the release helper's error and attempt count. |
| `cloud-fingerprint-failed` | warning | A cloud or network file was indexed but its duplicate fingerprint could not be read or saved. It stays indexed and does not take part in exact-duplicate exclusion. |

Refresh-run diagnostics:

| Code | Severity | Cause |
| --- | --- | --- |
| `search-root-unavailable` | warning | A search root was skipped before walking (see below). `source` is the opaque root prefix and `reason` is `gone`, `not-a-directory`, `unreadable`, or `empty`. |
| `mass-delete-suppressed` | warning | The run would delete more than the allowed share of indexed documents (see below). Nothing was deleted. |
| `walk-truncated` | warning | A root exceeded `maxScannedEntries` or `maxDepth`; `reason` is `max-entries` or `max-depth`, and unreached mirrors were kept. |
| `directory-unreadable` | warning | A subdirectory could not be listed; its mirrors are kept and its siblings still index. |
| `minsync-file-skipped` | warning | MinSync skipped a file it could not read, chunk, or embed; its previous chunks are kept and it is retried. |
| `minsync-sync-failed` | error/warning | A MinSync sync failed, or a per-file sync warning after a partial failure. |
| `minsync-lock-held` | error | Another sync owns the index (a `minsync` run outlived its host and left `.minsync/lock`). Retryable: run refresh again. |
| `minsync-staging-excluded` | warning | A parsed-mirror id could not be staged into the MinSync workspace (no canonical source-id form, or a missing parsed output). |
| `embedder-unavailable` | warning | No embedding backend answers, so semantic search is unavailable. `autorag setup` and `autorag status` report why. |
| `embedding-identity-mismatch` | error | A stored vector identity disagrees with the configured embedder; the fix is a full reindex. |
| `refresh-interrupted` | error | A previous refresh stopped before it completed (recorded progress is `running` but the owner pid is gone). |
| `watch-failed` / `watch-limited` | warning | A filesystem watcher could not be established, or the watcher cap was hit. |

## Unavailable search roots and mass deletions

A parsed refresh never treats a search root it cannot see as proof that its documents are gone. Before walking, each root is probed. A root that no longer exists (`gone`), is no longer a directory (`not-a-directory`), cannot be listed (`unreadable`), or is empty although it previously held indexed documents (`empty`) is skipped: its previous mirrors and skip records stay untouched, and the run records one `search-root-unavailable` warning. `deleted-mirror` is never emitted for these roots. Healthy roots in the same refresh are still indexed and still delete their genuinely stale mirrors. A forced refresh (`autorag refresh --force`) treats an empty root as intentionally emptied.

A refresh that would delete more than `refresh.maxDeleteRatio` (default `0.3`) of the indexed documents, and more than 100 of them, is refused: nothing is deleted and one `mass-delete-suppressed` warning is recorded. The next refresh repeats the check; a forced refresh deletes them.

The directory walk is bounded by `refresh.maxScannedEntries` (default `2000000` entries per root) and `refresh.maxDepth` (default `64`). Hitting either records a `walk-truncated` warning and keeps mirrors for anything the walk did not reach. A single unreadable subdirectory records `directory-unreadable`, keeps the mirrors beneath it, and does not stop its siblings.

```json
{ "refresh": { "maxDeleteRatio": 0.3, "maxScannedEntries": 2000000, "maxDepth": 64 } }
```

## Transient read retries

`EMFILE`, `EBUSY`, `EIO`, and `ETIMEDOUT` are retried with doubling backoff (50 ms, 100 ms, 200 ms) before the file is skipped. The retry count is `refresh.transientRetries` (default `3`; `0` disables retries).

```json
{ "refresh": { "transientRetries": 3 } }
```

Unlike permission-denied failures, these are recorded without a retry deferral, so the next refresh retries them. Mirror writes go through a temp file removed on failure, so a failed write leaves no `*.tmp` orphan.

A write failure that makes the workspace itself unusable (`ENOSPC`, `EDQUOT`, `EACCES`, `EPERM`, `EROFS`) is not a per-file problem and aborts the refresh with a `MirrorWorkspaceError` naming the workspace. Files already mirrored in that run stay in the index, so the next refresh resumes incrementally.

## Cancelling a refresh

`SIGINT` and `SIGTERM` abort a refresh. The signal is forwarded to the MinSync `init`, `check`, and `sync` processes: the running `minsync` and every descendant it started are stopped (SIGTERM to the process group, SIGKILL after 5 seconds), the `config.toml` this run rewrote is restored, and `refresh` rejects with the signal's reason. On POSIX each MinSync process leads its own process group, so the host also reaps it on `SIGINT`, `SIGTERM`, `SIGHUP`, and exit; Windows uses `taskkill /t`. Only the MinSync phase observes the signal so far; parse, datasource, and Jikji phases do not.

A `minsync` run that outlives its host (for example after the host is `SIGKILL`ed, which no handler can catch) keeps `.minsync/lock`. The next `check` or `sync` exits 3, which AutoRAG reports as the retryable `minsync-lock-held` diagnostic instead of a generic failure.

## Progress and liveness

AutoRAG runs `minsync sync --progress jsonl`. Each MinSync `start`, `batch`, `heartbeat`, and `done` event updates `<workspace>/.autorag/refresh-progress.json` with `minsync: { synced, total }`, advances `updatedAt`, and is also emitted through `AutoRAGAgent.refresh(..., { onProgress })`. MinSync's heartbeat interval defaults to five seconds.

A stale `updatedAt` between events means the phase is still running; it is not proof of a hang. Use the owner pid through `autorag status`'s `progress.ownerAlive` for liveness, and use advancing progress to tell active work from a stalled batch. Progress is written every 25 files during parsing; it is a display marker, not the engine's recovery checkpoint.

## Skipped files (partial sync)

MinSync skips a file it cannot read, chunk, or embed instead of aborting the sync, and exits 6 with `partial: true` and a `files_failed` list. AutoRAG treats that as a completed sync (`minsync.ok: true`) and reports each skipped file as a `minsync-file-skipped` warning whose `source` is the path inside the MinSync workspace. The count appears as `minsync.filesSkippedCount` in the result, `minsync.filesSkipped` in the completed `refresh-progress.json`, and `skipped=N` / `filesSkipped=N` in human output. MinSync keeps the file's previous chunks and retries it on the next sync; a later successful retry omits the count again. Only an exit 6 without a `partial` report is a `minsync-sync-failed`.

## Retry and failure behavior

During a MinSync-backed refresh, three consecutive unsuccessful sync runs with the same structured reason and diagnostic code stop further MinSync work. Missing fields compare consistently; changing only the error message does not reset the count. A different reason or code starts a new streak, and any successful sync (including one with per-file skips) clears it. A held index lock stops immediately.

Parsing overlaps indexing, so in-flight parsing may finish its current batch before observing the stop at the next committed-batch callback. There is no final retry after the threshold. Committed mirrors and the durable stage journal remain available: fix the reported cause and run `autorag refresh --method minsync` again to resume incrementally without reparsing unchanged committed sources. Deletions remain deferred until the final parse batch.

A systemic stop reports one MinSync component failure in each diagnostic list, preserving the terminal cause and original error text. Other failed runs retain at most 100 per-file sync warnings plus one summary with the omitted count. A later successful sync clears these warnings and the omitted count.

## Migration and identity

A MinSync workspace keeps its state in the `.minsync` subdirectory of `<workspace>/.autorag/minsync`, alongside the sync cursor and `config.toml`. The embedding identity includes provider, model, immutable model revision, dimension, query prefix, and passage prefix. A mismatch in any field except the query prefix requires a full reindex; the query prefix only shapes query vectors, so changing it is recorded without re-embedding documents. A dimension mismatch is detected before vector reuse.

An identity mismatch is reported as `embedding-identity-mismatch`; the fix is a full reindex of the affected store. To rebuild a MinSync workspace:

```bash
autorag index rebuild --method minsync
```

`autorag index reset --method minsync --yes` followed by `autorag refresh --method minsync` has the same effect. A MinSync workspace created through the old embedding gateway (a `tei:`/`.gguf` id, or an old `profile` field) must be rebuilt this way; vectors from that removed runtime are not compatible. Legacy `profile` and `.gguf` ids in `config.json` are ignored on read, without an error. See [Embedding backends](embedding-backends.md#rebuilding-after-the-embedding-gateway).

## Related pages

- [Scheduling and exclusions](scheduling-and-exclusions.md) — schedule, exclusions, and the watch daemon.
- [Embedding backends](embedding-backends.md) — the embedder and its health diagnostics.
- [Reference: CLI](../reference/cli.md) — command and flag reference.
