---
title: "Troubleshooting"
description: "Triage AutoRAG when search returns nothing or a datasource is missing, with the diagnostic codes and repair steps."
---

# Troubleshooting

When search returns nothing, a datasource disappears from results, refresh hangs, or the embedder is unavailable, work through this page. It covers the first triage commands, per-datasource native checks, every diagnostic code AutoRAG emits, and the repair playbook.

**Indexing is not searchability.** A green `status` proves the corpus was indexed, not that a source actually returns hits. Always confirm retrieval with a real query before calling a source healthy.

## Start with the doctor skill

Run the **`autorag-doctor`** agent skill: point your coding agent at [skills/autorag-doctor/SKILL.md](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/skills/autorag-doctor/SKILL.md) and say *"check AutoRAG"*. It walks the full diagnose-and-repair procedure — core triage, per-datasource native probes, searchability proofs, repairs — and ends with the per-datasource status table below. Use it for anything beyond the quick checks on this page.

## First triage

```bash
autorag status --json          # freshness, per-component state, diagnostics
autorag health --json          # model resolution + one live completion probe
autorag setup --format json    # resolves the embedding backend and reports it
```

- `status` reports `state`, `stale`, `diagnostics`, and per-component state for `minsync`, `jikji`, and `datasources`. `stale: true` or any `stale-index` diagnostic means the corpus changed since the last successful refresh.
- `health` resolves the single search model and does one live completion probe. `--skip-probes` proves config shape only — never claim a healthy model from it.
- `setup --format json` reports `embedding.backend` (`remote`, `daemon`, or `in-process`) with the resolved provider/model/dimension, or `embedding.reason` plus a `remediation` when none is alive. It only probes; it never starts a daemon.

Config lives at `--config`, `AUTORAG_CONFIG`, `$AUTORAG_HOME/config.json`, or `~/.autorag/config.json`. Read `searchPaths`, `workspacePath`, `minSync`, `jikji`, and `datasources` before changing anything.

Then prove searchability through the model-free MCP server, which isolates retrieval from model failures:

```text
autorag.status {}
autorag.search {"query":"a word that certainly appears","topK":3}
autorag.search {"query":"recent topic","datasourceIds":["discord"],"topK":3}
```

**Always read the `diagnostics` returned by MCP `autorag.search`.** A run that silently dropped a whole retrieval method still looks successful, just with fewer results; only the diagnostics name it (`retrieval-method-failed`). The model-backed CLI path hides diagnostics, `sessionId`, and per-result evidence unless `--debug` is set:

```bash
autorag search "summarize the collection" --top-k 3 --json --debug
```

Local-file hits must map to an absolute, existing path. Datasource hits keep source-native identities such as `/kakao/personal/chunks/42`; those are not filesystem paths and must never be passed to `cat`. Inspect the exact chunk behind a numbered result with `autorag evidence <session> --json` (or MCP `autorag.evidence`).

## Per-datasource status

Each CLI owns its archive, so ask the CLI, not AutoRAG. A datasource is `active` only when its binary exists, its store is present, and its own check passes.

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
| Obsidian / notes | `autorag status` (AutoRAG-owned index) | `autorag refresh --method datasources`; if corrupt, delete `.autorag/datasources/obsidian/<instance>` first |
| Mail | `mailcrawl doctor`, `mailcrawl status` | `mailcrawl sync && mailcrawl index` |
| Coding-agent sessions | `wombat doctor --json` | `wombat sync --index` |
| Lark / Feishu | `lark-cli auth status --format json` | no local sync; search is remote |

Rules:

- A missing binary is **not configured**, not a failure. Report it with its install command and move on.
- An empty store is a legitimate zero-hit result. `telecrawl` and `slacrawl` return JSON `null` (not `[]`) for no hits — treat that as empty, not broken.
- A configured datasource whose native check fails is a **failure** and must be repaired or reported explicitly. Never report a skip as a pass.
- Zero hits with a non-empty store means the CLI's search path is suspect, not the data. Cross-check against the raw index before concluding; a populated index plus an empty CLI result is an upstream bug — report it with the exact reproduction instead of re-syncing.

Native stores stay owned by their CLIs — AutoRAG never rebuilds them. Fix a broken archive with the owning CLI's doctor, then `autorag refresh --method datasources --json`.

## Diagnostic codes

| Code | Meaning | First move |
|---|---|---|
| `stale-index` | Sources changed since last refresh | `autorag refresh --method parsed,minsync --json` |
| `index-not-ready` | Index missing or never built | `autorag refresh --json` |
| `minsync-sync-failed` | MinSync indexing failed; the message carries MinSync's own reason | Fix the reported cause, then `autorag refresh --method minsync` |
| `minsync-file-skipped` | MinSync skipped one file (unreadable, unparsable, or rejected by the embedder) and kept its older chunks; retried on the next sync | Fix that file's permissions/content, then `autorag refresh --method minsync` |
| `minsync-lock-held` | Another `minsync` run holds `.minsync/lock`, often an orphan left by a killed host; retryable | Wait for it to finish or stop that `minsync` process, then `autorag refresh --method minsync` |
| `embedder-unavailable` | No remote endpoint answered, no `inl serve` daemon could start, and the in-process addon is missing or failed | Run `inl serve` by hand to read its error, install MinSync (`cargo install minsync`) or `@nomadamas/minsync-native`, or configure a remote `minSync.embedder` |
| `embedding-identity-mismatch` | Indexed vectors use a different embedder | `autorag index rebuild --method minsync` |
| `embedding-unavailable` | A datasource sidecar could not embed (same cause as `embedder-unavailable`) | Start the daemon or install the addon, then re-sync that datasource |
| `jikji-unavailable` | Jikji binary missing or install failed | Check the Rust toolchain, retry `autorag refresh --method jikji` |
| `jikji-unsupported-version` | A `jikji` older than the pinned version was found and not run; the message names the version and path | `autorag refresh --method jikji` installs the pinned build, or upgrade it yourself |
| `jikji-prepare-failed` | Jikji prepare of a root failed (e.g. per-root timeout); the refresh still finishes and other roots index | Bound the per-file/root budget (`jikji.timeoutMs`, `jikji.prepareTimeoutMsPerFile`), then retry |
| `jikji-discover-failed` | A `jikji_find` miss whose retry step failed; the message carries the CLI exit status and stderr verbatim | Read the quoted stderr; the failed root is dropped and healthy roots still answer |
| `everything-index-failed` | Windows Everything instance could not start or index; the message carries the ES exit code and stderr | Fix the reported cause, then `autorag refresh --method everything --json` |
| `fsearch-index-failed` | The macOS/Linux fsearch-cli index could not be built | Check `fsearch-cli`, then `autorag refresh --method fsearch` |
| `datasource-index-failed` | Backing CLI failed to index | Run that CLI's own doctor from the table above |
| `datasource-empty` | Store has no matching content | Re-sync with the CLI, or accept as empty |
| `datasource-auth-error` | Datasource credentials missing or rejected | Report the environment-variable **name**; never its value |
| `datasource-rate-limited` | Datasource quota exhausted | Wait and retry, or raise the quota with the provider |
| `unknown-datasource-skill` | Config key is not a known template and has no `"type"` | Fix the name or add `"type": "<template>"` |
| `retrieval-method-failed` | One method errored during the query | Read `--debug` diagnostics; the message carries the method's own error |
| `retrieval-source-missing` | Warning: hits whose source file was moved or deleted since the last refresh were skipped; the other hits are returned | Run `autorag refresh` to drop the stale entries |
| `rerank-disabled` | Warning: `rerank` names no model, so reranking is off and each source contributes its own top 10 | Set `rerank.model` to turn it on, or remove the block; nothing else is wrong |
| `mass-delete-suppressed` | A refresh would delete too large a share of indexed documents | Investigate the search root, then `autorag refresh --force` if the deletion is intended |
| `walk-truncated` | The directory walk hit its entry/depth cap under a root | Raise `refresh.maxScannedEntries` / `refresh.maxDepth` |
| `search-root-unavailable` | A search root is missing, unlistable, or empty while it previously held documents; mirrors kept | Restore the root, then `autorag refresh` |
| `parser-warning` | kordoc parsed a file with a warning; the file is still indexed | Inspect the warning; fix the source if the parse matters |
| `decision-model-unavailable` | The session decision model (Jev) cannot answer; `autorag search` exits 3 | Sign in with `/login` or set `OPENROUTER_API_KEY`; the `reason` names what failed |
| `decision-model-fallback` | Jev was unavailable; the session ran on the fallback small chat model | Fix Jev's credential, or keep the fallback intentionally |
| `query-route-fallback` | Jev routing unavailable; searched local with the original question | `test -n "$OPENROUTER_API_KEY"`; report the env var name, never its value |
| `datasource-selection-fallback` | Jev's per-datasource check unavailable; no datasource searched before the fast answer | Same as `query-route-fallback` |
| `query-decomposition-failed` | Decomposition model call failed; searched the original question | Check `queryDecomposition.model` resolves (`autorag models list --provider openrouter`) |
| `follow-up-check-fallback` | Jev post-fast-answer check unavailable; the run verified anyway | Same as `query-route-fallback` |
| `evidence-judgment-fallback` | Post-answer evidence judgment unavailable; nothing was stored to memory | Same as `query-route-fallback` |
| `citation-without-result` | A cited `[eN]` matched no recorded evidence; the citation was removed from the persisted answer | Re-run with `--debug` to see the retrieval diagnostics |
| `organize-skill-unavailable` | An organize request could not load the packaged organize skill; fell back to `local` | Re-install/repair the package so the packaged skill is present |
| `query-routed`, `datasources-selected`, `follow-up-skipped` | Info: Jev's branch and queries / datasources searched and skipped / the fast answer was judged final | None; working as intended. A datasource never selected usually needs a clearer `description` in the config |

## Repair procedures

### Stuck or missing embedding daemon

AutoRAG owns no lock or pid file for embeddings. A running `inl serve` publishes `host`, `port`, and `pid` under `~/.cache/im-not-llm/serve/<instance>/` and removes them on a clean stop; a leftover file from a crashed daemon is harmless (AutoRAG only adopts a daemon whose `GET /health` answers as a loopback server of the wanted model).

```bash
ls ~/.cache/im-not-llm/serve/* 2>/dev/null
autorag setup --format json
```

AutoRAG never starts the daemon. Run `inl serve --idle-timeout 0` by hand (or as a launchd/systemd service), read its own error, or press Ctrl+Shift+E in the TUI. `inl serve` stops itself only if started with `--idle-timeout N`; inl-server 0.3.3 and older (and `minsync serve` for now) default to 300 s, so with those a semantic failure after a quiet spell usually means it timed out. Stop an orphaned daemon by the pid from `~/.cache/im-not-llm/serve/<instance>/pid` only after confirming no live refresh holds it.

Index locks are owned by their engines: MinSync/tantivy locks under `<workspace>/.autorag/`, per-CLI locks such as `<workspace>/.autorag/datasources/discrawl/.discrawl-sync.lock`. Delete one only after confirming no owning process is alive.

### Orphan process

```bash
pgrep -fl 'autorag|minsync' | grep -v pgrep
```

A refresh killed mid-run can leave an `inl serve` daemon or a `minsync` child alive. A daemon idle-stops itself; stop a confirmed orphan by the pid above. Never stop a process another clone started. Re-run `autorag status --json` afterwards to confirm `inFlight: false`.

### Embedding model mismatch

`embedding-identity-mismatch` or a dimension error means the vectors on disk were built with a different embedder than the configured one. Vectors of two different dimensions can never be compared, so the index must be rebuilt:

```bash
autorag index rebuild --method minsync
```

The default is MinSync's local `Qwen/Qwen3-Embedding-0.6B` engine at 1024 dimensions, reached through a running daemon or MinSync's in-process engine (both report the identity `provider: minsync`). Changing the embedder's `dimension` in config **without** a rebuild leaves retrieval silently empty.

### Stale index

`stale-index` diagnostics or `stale: true` mean sources changed after the last refresh.

```bash
autorag refresh --method parsed,minsync --json
autorag refresh --force --json      # only when incremental refresh does not clear it
```

A file the parser failed on (corrupt PDF, encrypted HWP) or that was unreadable is not re-parsed on every refresh; it is skipped until its mtime, size, or parser version changes. After fixing the cause, retry those files once:

```bash
autorag refresh --retry-failed --json
```

For continuous freshness install an hourly `autorag watch --once` job (cron, launchd, systemd timer, or Task Scheduler).

### Missing or wrong setup

- `unknown-datasource-skill`: the config key is not a builtin template and has no `"type"`. It is skipped, not fatal — fix the name or add `"type"`.
- A configured datasource that returns nothing is a native store, connector, or query problem — every configured connection is searchable. Run its native check from the table above.
- A missing MinSync binary is not a diagnostic: `autorag search`, `autorag refresh`, and the MCP tools fail with `MinSync is required ...` and a non-zero exit. Install it (`cargo install minsync`) or leave `minSync.autoInstall` on, then retry.
- For browser history, `autorag setup --format json` reports `installed`, `running`, and `embedding` for Hister with a `remediation` per problem; `autorag setup --sync-hister` shares AutoRAG's embedding model with Hister (and refuses a remote or in-process backend).

If the CLI itself is missing or no config exists, run the [installation](installation.md) and [quickstart](quickstart.md) flow first — repair assumes an existing install.

## Report

Always end with this table, one row per datasource and per AutoRAG component:

| Source | Configured | Indexed | Searchable | Issue found | Fix applied |
|---|---|---|---|---|---|
| minsync | yes | yes | yes (3 hits) | – | – |
| discord | yes | yes | no | stale archive | `discrawl sync` |
| telegram | no | – | – | CLI not installed | reported |

`Searchable` must come from an actual query, never inferred from index state. Follow the table with the exact remaining action for every row that is not fully green.
