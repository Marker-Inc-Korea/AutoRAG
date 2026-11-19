---
title: "CLI commands"
description: "Every autorag command with its flags, defaults, exit codes, JSON output, and runnable examples."
---

# CLI commands

The `autorag` executable installs, indexes, searches, and maintains the librarian. This page lists every command the CLI router accepts, with each flag's type, default, meaning, and environment equivalent, plus exit codes and JSON output. For install steps see [Installation](../guide/installation.md).

## Global flags

Every command accepts these flags. They resolve a value in the order **flag → environment variable → config file → built-in default** (see [Configuration file](configuration.md)).

| Flag | Type | Default | Meaning | Environment |
|---|---|---|---|---|
| `--json` | boolean | off | Emit machine-readable JSON instead of human text | — |
| `--debug` | boolean | off | Reveal opaque internal diagnostics (never filesystem paths) | — |
| `--config <path>` | string | `$AUTORAG_HOME/config.json` | Use a specific config file | `AUTORAG_CONFIG` |
| `--search-paths <csv>` | string | `.` | Folders to index and search | `AUTORAG_SEARCH_PATHS` |
| `--languages <csv>` | string | `ko,en` | OCR and parser languages | `AUTORAG_LANGUAGES` |
| `--workspace <dir>` | string | current directory | Workspace that owns `.autorag` state | `AUTORAG_WORKSPACE` |
| `--memory-path <file>` | string | `$AUTORAG_HOME/memory.json` | Retrieval-memory file | `AUTORAG_MEMORY_PATH` |
| `--model-provider <name>` | string | from config | Override the model provider where a model is resolved | `AUTORAG_MODEL_PROVIDER` |
| `--model-id <id>` | string | from config | Override the model id where a model is resolved | `AUTORAG_MODEL_ID` |
| `--permission-mode <m>` | `auto` \| `ask` \| `full` | `auto` | Approval gate for file changes by the agent | `AUTORAG_PERMISSION_MODE` |
| `-h`, `--help` | boolean | — | Print command-specific usage | — |
| `-V`, `--version` | boolean | — | Print the package version | — |

`--model-provider` and `--model-id` must be given together; one without the other is a usage error.

## Config path resolution

The CLI resolves the config file in this order, taking the first that applies:

1. `--config <path>` — explicit.
2. `AUTORAG_CONFIG` — explicit.
3. `$AUTORAG_HOME/config.json` — implicit; `$AUTORAG_HOME` defaults to `~/.autorag`.
4. `~/.autorag/config.json` when `AUTORAG_HOME` is unset.

An explicit path that does not exist is an error (`Config file not found: <path>`). When the implicit home config is missing but a legacy `autorag.config.json` exists in the current directory, commands that write migrate it into the home config (normalizing its relative paths against the legacy directory); read-only commands read it in memory only.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Runtime failure (indexing, datasource, network, or model error) |
| `2` | Usage or configuration error (unknown flag/command, bad value, missing config) |
| `3` | `autorag search` only: the decision model is unavailable (`decision-model-unavailable`) |
| `130` / `143` | A signal (SIGINT/SIGTERM) stopped a one-shot refresh, rebuild, or watch tick |

Config errors return `2`; runtime errors return `1`. Unknown flags and unknown commands are rejected by the argument parser before any command runs and return `2`.

## init

```text
autorag init [flags]
```

Writes the config (default `~/.autorag/config.json`) for a local collection. An existing config is not overwritten without `--force`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--search-paths <csv>` | string | `.` | Folders to index and search |
| `--exclude <path>` | string, repeatable | — | Exclude a file or folder (relative to the current directory; globs are kept verbatim) |
| `--languages <csv>` | string | `ko,en` | OCR and parser languages |
| `--workspace <dir>` | string | current directory | Workspace that owns `.autorag` state |
| `--memory-path <file>` | string | `$AUTORAG_HOME/memory.json` | Retrieval-memory file |
| `--model-provider <name>` | string | — | Librarian model provider (with `--model-id`) |
| `--model-id <id>` | string | — | Librarian model id (with `--model-provider`) |
| `--embedder-id <id>` | string | MinSync native Qwen3-Embedding-0.6B | MinSync embedder id; `none` selects a sparse-only SPLADE index with no dense model |
| `--embedder-base-url <url>` | string | — | MinSync embedder endpoint |
| `--embedder-api-key-env <var>` | string | — | Env var holding the embedder key (name only, never the key) |
| `--embedder-dimension <n>` | integer > 0 | — | Embedding dimension |
| `--embedder-query-prefix <text>` | string | — | Prefix added to queries |
| `--embedder-passage-prefix <text>` | string | — | Prefix added to passages |
| `--embedder-timeout-ms <n>` | integer > 0 | — | Per-request timeout |
| `--embedder-batch-size <n>` | integer > 0 | — | Embedding batch size |
| `--minsync-max-chunk-size <n>` | integer > 0 | MinSync default | MinSync chunk size limit |
| `--force` | boolean | off | Replace an existing config |

**`--force` rules:** a plain `autorag init` never overwrites an existing config. `--force` replaces an existing config **only when the path is explicit** (`--config` or `AUTORAG_CONFIG`); against the implicit home config it refuses with `Refusing to overwrite existing config ... without an explicit config path`. To reconfigure in place, pass `--config`, or delete the file first.

JSON output: `{"ok":true,"wrote":["<absolute config path>"]}`.

```bash
autorag init --search-paths ~/Documents/research --languages ja,en
autorag init --config ./autorag.json --search-paths ./docs --force
```

## setup

```text
autorag setup [flags]
```

Probes MinSync, the embedding backend, and every known datasource, and reports what is ready and what is blocked. When no config exists yet, a default one is written first from `--search-paths`/`--workspace`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--search-paths <csv>` | string | `.` | Folders for the config written when none exists |
| `--workspace <dir>` | string | current directory | Workspace for the config written when none exists |
| `--sync-hister` | boolean | off | Rewrite Hister's `semantic_search` keys to AutoRAG's shared model (backup kept), restart Hister, and re-embed when needed. Without it, setup only reports differences and never writes Hister's config |
| `--format json` | string | — | Same as `--json` |

Exit codes: `0` when every probe is ready, `1` when something is blocked, `2` on config error.

```bash
autorag setup --format json
```

## refresh

```text
autorag refresh [flags]
```

Parses configured search paths, then resyncs the parsed mirror, MinSync, datasources, Jikji, and the platform file-name index. Model-free: no LLM is constructed.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--method <csv>` | `minsync,parsed,datasources,jikji,everything,fsearch,all` | all | Restrict the run; `all` expands to every method |
| `--force` | boolean | off | Re-index even when the mirror looks unchanged |
| `--retry-failed` | boolean | off | Retry files whose parse or read failed earlier, once, even if unchanged |

SIGINT/SIGTERM abort the run (children stopped, run recorded as `failed`) and exit `130`/`143`.

JSON output (`--json`) returns an envelope with `ok`, `counts` (`scanned`, `written`, `deleted`, `skipped`), `diagnostics`, and per-component blocks (`minsync`, `datasources`, `everything`, `fsearch`). `ok` is `true` only when every index the run touched succeeded; a missing optional `fsearch-cli` is a warning, not a failure.

```bash
autorag refresh
autorag refresh --method minsync,parsed --force
```

## status

```text
autorag status
```

Path-opaque snapshot of corpus staleness and the cached last-refresh outcome. Model-free; never emits filesystem paths. With `--json` it serializes the `AutoRAGRefreshStatus` object (`state`, `inFlight`, `stale`, `counts`, `components`, `diagnostics`, optional `progress`, `lastError`).

Exit codes: `0` success, `2` config error, `1` runtime error.

## search

```text
autorag search <query> [flags]
```

Runs the librarian agent loop: retrieval, reading, curation, numbered results. Requires a configured, reachable decision model (the `jev` config section); the first question probes it and stops with code `3` (`decision-model-unavailable`) when it cannot answer.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--top-k <n>` | integer | agent default | Maximum curated results |
| `--scope <scope>` | string | — | Narrow retrieval to a scope |
| `--fast-thinking <lvl>` | `off,minimal,low,medium,high,xhigh,max` | minimal | Thinking level for the fast first answer |
| `--final-thinking <lvl>` | same set | high | Thinking level for the verified answer |
| `--permission-mode <m>` | `full` \| `auto` \| `ask` | `full` here (no terminal to approve) | Approval gate for file changes; `--permission-mode` and `AUTORAG_PERMISSION_MODE` override the config value |

Streaming output: tasks print progress lines, then the preliminary answer, then the final answer. Under `--json` each is a separate JSON line: `{"type":"progress","text":...}`, then the preliminary envelope, then the final envelope.

Exit codes: `0` answered, `1` search failed, `2` usage or model configuration, `3` decision model unavailable.

```bash
autorag search "What are our primary Q3 deliverables?" --top-k 5
```

## evidence

```text
autorag evidence <session> [flags]
```

Shows the persisted source and chunk evidence attached to a search session, as recorded in retrieval memory.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--result <n>` | integer > 0 | all results | Show evidence for a single numbered result |

JSON output serializes `{sessionId, results:[...]}`. MCP clients normally use the `autorag.evidence` tool; this command stays available for terminal maintenance.

## memory

```text
autorag memory <inspect|export-ltr> [flags]
```

| Subcommand | Effect |
|---|---|
| `inspect` | Renders a path-opaque snapshot of judged evidence, curated results, evidence chunks, and insights. Model-free; the storage path is never emitted |
| `export-ltr` | Writes the learning-to-rank log beside the memory file as LightGBM LambdaRank files: `train`/`valid` `.tsv` with `.query` and `.position` sidecars, row ids, and a manifest. Model-free; reads the log only |

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--output <dir>` | path | none | `export-ltr` only; directory to write the dataset into (required) |
| `--valid-fraction <0..1>` | number | `0.2` | `export-ltr` only; newest share of runs held out for validation |

Any other subcommand is a usage error (exit `2`). See [learning-to-rank.md](../retrieval/learning-to-rank.md).

```bash
autorag memory export-ltr --output ./ltr --valid-fraction 0.2
```

## ltr

```text
autorag ltr <status|list|train|enable|disable|use|auto-train|delete> [flags]
```

Trains a learning-to-rank model from the search log and, only when enabled, serves it after the reranker. Training runs in process (XGBoost compiled to WebAssembly); nothing else needs installing.

| Subcommand | Effect |
|---|---|
| `status` | Logged runs, trained models, and whether serving or retraining is due |
| `list` | Trained models with their holdout verdict |
| `train` | Fit a ranker on the log and store it. A model passes only when it beats the reranker's own order on the newest runs it never saw |
| `enable`, `disable` | Turn serving on or off in the `ltr` block of the active config file |
| `use <model-id\|latest>` | Serve one model, or the newest passing one |
| `auto-train <on\|off>` | Retrain in the background once enough new runs are logged |
| `delete <model-id>` | Remove a stored model |

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--min-new-runs <n>` | integer > 0 | `200` | `auto-train` only; new logged runs between automatic retrains |

Setting subcommands change only the `ltr` block of the active config file. See [learning-to-rank.md](../retrieval/learning-to-rank.md).

```bash
autorag ltr train
autorag ltr enable
```

## index

```text
autorag index <reset|rebuild> [flags]
```

| Subcommand | Effect |
|---|---|
| `reset` | Remove the parsed mirror and MinSync indexes; keeps `bin`, `datasources`, and the memory file |
| `rebuild` | Reset, then re-run a forced refresh |

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--method <csv>` | `minsync,parsed,all` | all | Scope the reset |
| `--yes` | boolean | off | Skip the interactive confirmation prompt |

Reset targets are confined to `<workspace>/.autorag`; a symlinked or out-of-workspace index directory is refused rather than followed. Exit codes: `0` success, `2` usage or declined, `1` runtime error, `130`/`143` when a rebuild is stopped by a signal.

```bash
autorag index reset --method minsync --yes
autorag index rebuild
```

## watch

```text
autorag watch [flags]
```

Keeps the parsed mirror and MinSync indexes current. With no flags it runs a long-lived recursive fs watch with debounced refreshes; `--once` runs a single refresh tick for cron, launchd, or Task Scheduler.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--once` | boolean | off | Run one refresh tick and exit |
| `--immediate` | boolean | true | Refresh once before reading fs events |
| `--debounce-ms <n>` | integer > 0 | `1500` | Debounce window for fs events |
| `--force` | boolean | off | Re-index even when the mirror looks unchanged |
| `--power <any\|ac>` | string | config `refresh.schedule.power` | With `ac`, index only while plugged in (pauses on battery) |
| `--window <HH:MM-HH:MM>` | string, repeatable | config | Index only inside this local-time window; `22:00-07:00` wraps midnight |
| `--days <csv>` | `mon,tue,wed,thu,fri,sat,sun` | config | Index only on these weekdays |

Defaults come from config `refresh.schedule` (`{ power, windows, days }`); flags override it. A `--once` tick skipped by the schedule exits `0` with `skipped: true`. An explicit `autorag refresh` is never gated. SIGINT/SIGTERM stop a `--once` tick with `130`/`143`; the long-running watch handles its own signals and exits `0`.

## health

```text
autorag health [flags]
```

Checks model resolution, provider auth, and a live completion probe. It does **not** check index health — use `autorag status` for that.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--skip-probes` | boolean | off | Skip the network completion probe (auth checks still run) |
| `--timeout-ms <n>` | integer | `10000` | Per-probe timeout |

JSON output is a `HealthReportV1` (`healthSchemaVersion`, `ok`, `category`, `coverage`, `config`, `model`, `probe`, `indexHealth`). Exit codes: `0` ok, `2` config or model-resolution failure, `1` auth, provider, completion, or timeout failure.

```bash
autorag health --skip-probes
```

## duplicates

```text
autorag duplicates [DIR]
```

Scans `DIR`, or the configured search paths when `DIR` is omitted, for exact and near-duplicate document families with `dupey`. Read-only: it never deletes or moves source files. JSON output lists `exactGroups`, `families`, and `extractionErrors` with `action: "review"`. Exit codes: `0` success, `2` config error, `1` runtime error.

## lite

```text
autorag lite <subcommand> [args] [flags]
```

Model-free lifecycle commands. They build the `AutoRAGLite` runtime (see [TypeScript API](typescript-api.md)) and never construct an LLM.

| Subcommand | Purpose |
|---|---|
| `lite init` | Write the model-free lifecycle config (`--search-paths`, `--exclude` (repeatable), `--workspace`, `--memory-path`, `--force`) |
| `lite refresh` | Incremental index refresh (`--full`, `--force`, `--retry-failed`, `--method`) |
| `lite watch` | Watch configured roots, or `--once` for a single tick (`--immediate`, `--debounce-ms N`, `--force`) |
| `lite status` | Path-opaque corpus freshness and index health |
| `lite index` | Reset or rebuild parsed/MinSync indexes (`reset`\|`rebuild`, `--yes`, `--method`) |
| `lite duplicates` | Duplicate scan without deleting files |
| `lite health` | Alias of `lite status` |
| `lite retrieve <query>` | Retrieve documents without model curation |
| `lite report <query>` | Persist a structured report from external curation |
| `lite find <query>` | Whole search — file-name index, local content, and datasources — streamed as NDJSON |

`lite refresh`, `lite watch`, `lite status`, `lite index`, `lite duplicates`, and `lite health` decompose to the same implementations as their top-level commands; the differences are model-free construction and the `lite refresh`/`lite watch` flag sets above. Exit codes follow the shared conventions (`0`/`1`/`2`, plus `130`/`143` for an interrupted one-shot refresh).

### lite retrieve

```text
autorag lite retrieve <query> [flags]
```

Retrieves merged results without entering the model-backed agent loop.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--top-k <n>` | integer > 0 | engine default | Maximum results |
| `--scope <scope>` | string | — | Narrow retrieval to a scope |
| `--since <iso>` | ISO-8601 | — | Keep only evidence dated at or after this instant (date-only is local midnight) |
| `--until <iso>` | ISO-8601 | — | Keep only evidence dated before this instant |
| `--refresh` | boolean | off | Rebuild indexes incrementally first |
| `--strict` | boolean | off | Exit `2` when the index is stale instead of answering from it |

Results with no date are excluded by `--since`/`--until`, and the method is reported in `unsearched`. A stale index is reported as `"stale": true` plus stale-index diagnostics and still answers unless `--strict` is set. JSON output is a `LiteRetrieveEnvelope` (`ok`, `query`, `stale`, `results`, `unsearched`, optional `window`, `diagnostics`).

Exit codes: `0` answered, `2` index not ready or `--strict` on a stale index, `1` runtime error.

### lite report

```text
autorag lite report <query> [flags]
```

Persists a structured report curated externally into retrieval memory, validated against `reportSchema` (see [TypeScript API](typescript-api.md)).

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--input <file>` | string | stdin | Read the report JSON body from `FILE` instead of stdin |

JSON output is `{ok, sessionId, query, answer, resultCount, diagnostics?}`. `2` when the query is missing or the report fails schema validation, `1` on runtime error.

### lite find

```text
autorag lite find <query> [flags]
```

A model-free whole search for a Cmd+F style launcher: the platform file-name index, local document content, and every connected datasource, streamed as NDJSON.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--query <text>` | string | positional | Search `TEXT` as the query (wins over the positional; use for queries beginning with `-`) |
| `--top-k <n>` | integer > 0 | `10` | Maximum hits per group |
| `--groups <key,key>` | string, repeatable | every group | Narrow to group keys: `files`, `content`, `source:<datasourceId>` |

With `--json` each line is NDJSON, in completion order: a `groups` line first, one `group` line per surface, then a `done` line.

## attachments

```text
autorag attachments [status|download] [flags]
```

Each Discord connection saves attachments under `~/.autorag/attachments/discord/<connection>/<server>/<channel>/` (moved with `AUTORAG_HOME`) and records every attachment in `download-status.json` beside them.

| Action | Effect |
|---|---|
| `status` (default) | Re-check saved files and rewrite `download-status.json` |
| `download` | Also save every attachment whose Discord link still works |

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--connection <name>` | string | all connections | Only this Discord datasource |
| `--debug` | boolean | off | List every attachment, not only the downloadable ones |

A saved file that was deleted, renamed, or edited no longer counts as downloaded: edited files are kept as yours and the original is saved again beside them while Discord's 24-hour signed link is valid. Nothing is overwritten or deleted. Refreshes download automatically unless `connector.downloadAttachments` is `false`. Exit codes: `0` success, `2` config/usage error, `1` when a connection could not be checked.

## tui

```text
autorag tui [flags]
```

Opens Pi's interactive mode over the AutoRAG librarian. Pi owns the terminal UI, provider login, model selection, session persistence, resume/fork/compaction, and settings; AutoRAG adds its retrieval tools and runs the two-phase search in the same session. Native commands (`/login`, `/model`, `/resume`, `/new`, `/tree`, `/compact`, `/settings`) are available in-session.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--fast-thinking <lvl>` | thinking level | minimal | Thinking level for the fast first answer |
| `--final-thinking <lvl>` | thinking level | high | Thinking level for the verified answer |
| `--permission-mode <m>` | `auto` \| `ask` \| `full` | `auto` (or `full` with a bridge) | Approval gate; Shift+Tab or `/permissions` switches it in-session |
| `--telegram` | boolean | off | Load the `@llblab/pi-telegram` bridge so a paired Telegram chat can ask, steer, and read answers |
| `--whatsapp` | boolean | off | Control the session from WhatsApp (installs `whatsapp-pi`; pair once with `/whatsapp`) |
| `--whatsapp-group <jid>` | string | — | With `--whatsapp`, answer only this WhatsApp group |
| `--discord` | boolean | off | Control the session from Discord via `pi-discord-remote`; needs `/pi-discord-remote setup` or `DISCORD_TOKEN` + `DISCORD_GUILD_ID` |

Typing while a search runs steers it: Enter applies a correction before the next model request, Alt+Enter queues a new question, Esc stops the search. Exit code `1` on runtime error.

```bash
autorag tui
autorag tui --telegram --permission-mode ask
```

## models list

```text
autorag models list [flags]
```

Lists chat models from the pi runtime (built-in catalog plus `models.json`, custom, and extension providers) with provider auth status. Never prints credential values.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--provider <id>` | string | all providers | Restrict to one provider |
| `--available` | boolean | off | Show only providers with configured credentials |
| `--decision` | boolean | off | List decision models (the `jev.backend`/`jev.model` candidates) instead of chat models |
| `--format json` | string | — | Same as `--json` |

`ls` is accepted as an alias for `list`. JSON output is `{ok, action, provider?, type?, count, models:[{provider,id,name,api,available}]}`.

```bash
autorag models list --available
autorag models list --decision --format json
```

## update-check

```text
autorag update-check
```

Queries the published `@autorag/librarian` version and reports whether a newer release is available. Set `AUTORAG_NO_UPDATE_CHECK=1` to skip the lookup; the TUI also runs this check on launch. JSON output merges the update result into `{ok, action:"update-check", status, packageName, currentVersion, latestVersion?, installCommand}`.

## automate

```text
autorag automate daemon
autorag automate install
autorag automate uninstall
autorag automate status
autorag automate run <id> [--event <create|update|delete>:<path>]
```

Reads `~/.autorag/automations.json` (or `$AUTORAG_HOME/automations.json`). `daemon` watches configured folders with `@parcel/watcher` and runs matching automations; `install`/`uninstall` manage the OS login service (launchd, systemd `--user`, or Windows Task Scheduler); `status` reports service state and each automation's last run; `run` executes one automation immediately.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--event <type:path>` | string, repeatable | synthetic | Provide a synthetic file event for `run` |

Exit codes: `0` success (including `daemon` when another daemon already runs), `1` runtime error, `2` unknown subcommand, unknown automation id, or malformed `--event`.

## See also

- [Configuration file](configuration.md) for every config key.
- [Environment variables](environment-variables.md) for every variable the code reads.
- [TypeScript API](typescript-api.md) to embed the same runtime in code.
- [Connecting datasources](../datasources/connecting.md) for the `datasources` config section.
- [Refresh operations](../indexing/refresh-operations.md) for what `refresh`, `watch`, and `index` do internally.
