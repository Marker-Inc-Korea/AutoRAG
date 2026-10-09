---
name: autorag-doctor
description: Diagnose and repair a broken or half-working AutoRAG install so every configured source is both indexed and searchable. Checks AutoRAG, MinSync, Jikji, the embedding gateway, and every CLI-backed datasource (lazykatok, discrawl, slacrawl, wacrawl, telecrawl, notcrawl, qmd, mailcrawl, rclone, Spotlight), then fixes orphan locks, orphan processes, embedding dimension or identity mismatches, stale indexes, and missing setup. Use when search returns nothing or too little, refresh hangs or fails, a datasource disappeared from results, indexes look stale, the gateway will not start, or the user asks to check, diagnose, verify, or repair AutoRAG.
license: MIT
---

# AutoRAG doctor

Goal: every configured datasource is **indexed** *and* **actually returns
hits**. A green `status` is not enough — the run is done only when a real query
returns a real hit per datasource, or the datasource is reported as genuinely
empty or not configured.

Always finish with the status table in [Report](#report).

## Safety

- Never delete, move, or edit source documents.
- Never delete a datasource's **native** store (`~/Library/Application Support/katok`, `~/.discrawl`,
  `~/.mailcrawl`, `.qmd`, Telegram/WhatsApp/Notion archives). AutoRAG only
  reads them; rebuilding them is the owning CLI's job.
- Only AutoRAG-owned state under `AUTORAG_HOME` and the workspace `.autorag`
  directory may be reset.
- Never print token or password values. Report credential **names** only.
- Kill a process only after confirming it is an AutoRAG-owned orphan.

## 1. Triage the core

```bash
autorag status --json
autorag health --json
autorag gateway status --format json
```

- `status` reports `state`, `stale`, `diagnostics`, and per-component state for
  `minsync`, `jikji`, and `datasources`. `stale: true` or any `stale-index`
  diagnostic means the corpus changed since the last successful refresh.
- `health` resolves the single search model and does one live completion probe.
  `--skip-probes` only proves config shape, never live access; do not claim a
  healthy model from it.
- The gateway is on-demand: `stopped` is normal when nothing is embedding.
  `unavailable` **while** a refresh is running is a real failure.

Config lives at `--config`, `AUTORAG_CONFIG`, `$AUTORAG_HOME/config.json`, or
`~/.autorag/config.json`. Read `searchPaths`, `workspacePath`, `minSync`,
`jikji`, and `datasources` before changing anything.

If the CLI itself is missing or the config does not exist, stop and run the
`autorag-setup` skill first — doctor repairs an existing install, it does not
create one.

## 2. Probe every datasource natively

Each CLI owns its archive, so ask the CLI, not AutoRAG. A datasource is only
`active` when its binary exists, its store is present, and its own check passes.

| Datasource | Native check | Re-sync when empty or stale |
|---|---|---|
| MinSync (local docs) | `minsync status`, `minsync check`, `minsync verify` | `autorag refresh --method minsync` |
| Jikji (discovery) | `jikji doctor` | `autorag refresh --method jikji` |
| Everything (Windows file names) | `autorag status --json` → `components.everything` | `autorag refresh --method everything --json` |
| KakaoTalk | `lazykatok doctor` | `lazykatok sync && lazykatok index` |
| Discord | `discrawl --json metadata` | `discrawl sync` |
| Slack | `slacrawl --json doctor` | `slacrawl sync` |
| WhatsApp | `wacrawl --json doctor` | `wacrawl import` |
| Telegram | `telecrawl --json doctor`, `telecrawl --json status` | `telecrawl import` |
| Notion | `notcrawl doctor`, `notcrawl status` | `notcrawl sync --source desktop` |
| Obsidian / notes | `qmd status` | `qmd update && qmd embed` |
| Mail | `mailcrawl doctor`, `mailcrawl status` | `mailcrawl sync && mailcrawl index` |
| Cloud drive | `rclone listremotes` | `autorag refresh --method datasources` |
| Spotlight (macOS) | `mdutil -s /` | indexed by the OS; no AutoRAG sync |
| Lark / Feishu | `lark-cli auth status --format json` | no local sync; search is remote |

Rules:

- A missing binary is **not configured**, not a failure. Report it with the
  install command and move on.
- An empty store is a legitimate zero-hit result. `telecrawl` and `slacrawl`
  return JSON `null` (not `[]`) for no hits — treat that as empty, not broken.
- A configured datasource whose native check fails is a **failure** and must be
  repaired or reported explicitly. Never report a skip as a pass.
- Zero hits with a non-empty store means the CLI's search path is suspect, not
  the data. Cross-check against the raw index before concluding (for example
  `slacrawl sql "select count(*) from message_fts where message_fts match 'x';"`
  against `slacrawl search x`). A populated index plus an empty CLI result is an
  upstream bug: report it with the exact reproduction instead of re-syncing.

## 3. Prove searchability

Indexing without retrieval is a failed run. Probe retrieval per datasource
through the model-free MCP tools, then once end to end:

```text
autorag.status {}
autorag.search {"query":"a word that certainly appears","topK":3}
autorag.search {"query":"recent topic","tags":["discord"],"topK":3}
autorag.search {"query":"recent mail subject","scope":"/mailcrawl/**","topK":3}
```

```bash
autorag search "summarize the collection" --top-k 3 --json --debug
```

- **Always read the `diagnostics` returned by MCP `autorag.search`.** A run
  that silently dropped a whole retrieval method still looks successful, just
  with fewer results; only the diagnostics name it (`minsync-unavailable`,
  `retrieval-method-failed`). The model-backed CLI `autorag search --json`
  hides `diagnostics`, `sessionId`, and per-result evidence unless `--debug` is
  set, so pass `--debug` when diagnosing that path.
- A method missing from the returned `method` values means that method
  contributed nothing. During a full MinSync re-sync this is expected: the
  store is being rebuilt, `minsync status` reports `NotSynced`, and local-file
  hits stay absent until it finishes. Confirm with `minsync status` before
  treating it as a failure, and never kill a running sync to "fix" it.
- MCP `autorag.search` needs no model, so it isolates retrieval from model
  failures.
- Use the MCP `tags` / `scope` arguments to select one configured datasource;
  `tags` match a datasource's descriptive metadata and `scope` narrows a query
  to a sub-path, both ordinary filtering. Every configured connection is
  searchable — if one returns nothing, investigate its native store, connector,
  or the query itself.
- `autorag.evidence {"sessionId":"...","resultNumber":N}` shows the exact chunk
  behind a numbered result; use it to confirm a hit is real and its source is
  readable. The CLI `autorag evidence SESSION --json` remains for terminal
  repair.
- Local-file hits must map to an absolute, existing path. Datasource hits keep
  source-native identities such as `/kakao/personal/chunks/42`; those are not
  filesystem paths and must never be passed to `cat`.

## 4. Repair playbook

### Orphan lock

The embedding runtime keeps `embedding-runtime.lock`, `embedding-runtime.pid`,
and `embedding-runtime.port` in `AUTORAG_HOME`. It reclaims them automatically
when the recorded PID is dead; a `lock-conflict` means the PID is **alive**.

```bash
autorag gateway status --format json
autorag gateway stop
```

Only when `gateway stop` cannot clear it, and the recorded PID is confirmed
dead (`kill -0 PID` fails), remove the three files by hand and retry.

Index locks are owned by their engines: MinSync/tantivy locks under
`<workspace>/.autorag/`, and per-CLI locks such as
`<workspace>/.autorag/datasources/discrawl/.discrawl-sync.lock`. Delete one
only after confirming no owning process is alive; otherwise wait for the run
that holds it.

### Orphan process

```bash
pgrep -fl 'autorag|autorag-gateway|minsync' | grep -v pgrep
```

A refresh that was killed mid-run can leave the gateway or a `minsync` child
alive. Stop the gateway with `autorag gateway stop` first; only kill a PID
directly when it is confirmed orphaned (no parent CLI, no live refresh).
Re-run `autorag status --json` afterwards to confirm `inFlight: false`.

### Embedding model mismatch

`embedding-identity-mismatch` or a dimension error means the vectors on disk
were built with a different embedder than the configured one. Vectors of two
different dimensions can never be compared, so the index must be rebuilt:

```bash
autorag models prefetch --profile qwen3-embedding-0.6b
autorag index rebuild --method minsync
```

The current default is the local `native:Qwen/Qwen3-Embedding-0.6B` runtime at
1024 dimensions. A workspace still pinned to the legacy 768-dimension
Ollama/TEI path must be reindexed explicitly, or pinned to an explicit profile.
Changing `embedder.dimension` in config **without** a rebuild leaves retrieval
silently empty. Keep embeddings local; do not point the embedder at a remote
endpoint to work around a local failure.

### Stale index

`stale-index` diagnostics or `stale: true` mean sources changed after the last
refresh.

```bash
autorag refresh --method parsed,minsync --json
autorag refresh --force --json
```

Use `--force` only when incremental refresh does not clear it. For continuous
freshness install an hourly `autorag watch --once` job (cron, launchd, systemd
timer, or Task Scheduler).

### Missing or wrong setup

- `unknown-datasource-skill`: the config key is not a builtin template and has
  no `"type"`. It is skipped, not fatal — fix the name or add `"type"`.
- `datasource-index-failed` / `sync-failed`: the backing CLI errored. Run that
  CLI's own check from the table above and fix it there.
- `minsync-unavailable` / `jikji-unavailable`: the binary is missing and
  auto-install failed. Both install through cargo; verify the Rust toolchain,
  then `autorag refresh --method minsync` to retry.
- `auth-error` / `rate-limited`: model or datasource credentials. Report the
  missing environment-variable **name** and let the user supply it.
- A configured datasource that returns nothing is a native store, connector, or
  query problem — every configured connection is searchable. Run its native
  check from the table above and fix it there.

## Diagnostic codes

| Code | Meaning | First move |
|---|---|---|
| `stale-index` | Sources changed since last refresh | `autorag refresh --method parsed,minsync` |
| `index-not-ready` | Index missing or never built | `autorag refresh --json` |
| `minsync-unavailable` | MinSync binary missing or install failed | Check cargo, retry refresh |
| `jikji-unavailable` | Jikji binary missing or install failed | Check cargo, retry refresh |
| `everything-index-failed` | Windows Everything instance could not start or index; message carries ES exit code and stderr | Fix the reported cause, `autorag refresh --method everything --json` |
| `embedding-identity-mismatch` | Indexed vectors use a different embedder | `autorag index rebuild --method minsync` |
| `embedder-unavailable` | Embedding gateway or runtime down | `autorag gateway status --format json` |
| `lock-conflict` | Another runtime holds the lock | `autorag gateway stop` |
| `datasource-index-failed` | Backing CLI failed to index | Run that CLI's own doctor |
| `datasource-empty` | Store has no matching content | Re-sync with the CLI, or accept as empty |
| `unknown-datasource-skill` | Config name is not a known template | Fix the name or add `"type"` |
| `retrieval-method-failed` | One method errored during the query | Read `--debug` diagnostics |
| `auth-error` | Credentials missing or rejected | Report the env var name |
| `query-route-fallback` | Jev routing unavailable (often `OPENROUTER_API_KEY` unset); searched local with the original question | `test -n "$OPENROUTER_API_KEY"`; report the env var name, never its value |
| `query-decomposition-failed` | Decomposition model call failed; searched the original question | Check `queryDecomposition.model` resolves (`autorag models list --provider openrouter`) |
| `follow-up-check-fallback` | Jev post-fast-answer check unavailable; the run verified | Same as `query-route-fallback` |
| `datasource-selection-fallback` | Jev datasource check unavailable; no datasource was searched before the fast answer | Same as `query-route-fallback` |
| `query-routed` / `datasources-selected` / `follow-up-skipped` | Info: Jev's branch and queries / datasources searched and skipped / fast answer judged final | None; working as intended. A datasource that is never selected usually needs a clearer `description` in the config |

## Report

Always end with this table, one row per datasource and per AutoRAG component:

| Source | Configured | Indexed | Searchable | Issue found | Fix applied |
|---|---|---|---|---|---|
| minsync | yes | yes | yes (3 hits) | – | – |
| discord | yes | yes | no | stale archive | `discrawl sync` |
| telegram | no | – | – | CLI not installed | reported |

`Searchable` must come from an actual query in step 3, never inferred from
index state. Follow the table with the exact remaining action for every row
that is not fully green.

## Completion condition

Done only when: core triage is clean or every remaining diagnostic is
explained, every configured datasource passed its native check, every one of
them returned a real hit or is proven empty, every repair was re-verified by
re-running the failing check, and the report table was delivered.
