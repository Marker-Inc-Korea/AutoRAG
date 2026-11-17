---
title: "Configuration file"
description: "Every top-level and nested key of the AutoRAG config.json, its type, default, meaning, and validation rules."
---

# Configuration file

The CLI and the TypeScript API read one JSON config file per run — by default `~/.autorag/config.json` (or `$AUTORAG_HOME/config.json`). This page documents every key the loader accepts, its type, default, and meaning, with a minimal and a fully annotated example. For how the path is chosen see [CLI commands](cli.md#config-path-resolution); for the environment overrides see [Environment variables](environment-variables.md).

## Precedence

For a given setting, the first source that supplies a value wins:

1. **Command-line flag** (`--search-paths`, `--workspace`, `--memory-path`, `--languages`, `--permission-mode`, `--model-provider`/`--model-id`).
2. **Environment variable** (`AUTORAG_SEARCH_PATHS`, `AUTORAG_WORKSPACE`, `AUTORAG_MEMORY_PATH`, `AUTORAG_LANGUAGES`, `AUTORAG_PERMISSION_MODE`, `AUTORAG_MODEL_PROVIDER`/`AUTORAG_MODEL_ID`).
3. **Config file**.
4. **Built-in default**.

Relative paths in the config file (and in a migrated legacy config) resolve against the config file's directory; relative `memoryPath` resolves against the resolved `workspacePath`. `excludePaths` entries that look like globs are kept verbatim; other entries resolve against the workspace.

## Top-level keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `configPath` | `string` | resolved path | Read-only; the absolute path of the file the config was resolved from. Never set by hand |
| `searchPaths` | `string[]` | `["."]` | Folders to parse, index, and search |
| `workspacePath` | `string` | current directory | Directory that owns `.autorag` state and the index cache |
| `memoryPath` | `string` | `$AUTORAG_HOME/memory.json` | Retrieval-memory file |
| `languages` | `string[]` | `["ko","en"]` | OCR and parser languages |
| `model` | `AgentModelConfig` | none | Librarian chat model (see below) |
| `minSync` | object \| `false` | `{enabled:true,autoInstall:true}` | MinSync indexing. `false` and `enabled:false` are rejected — MinSync is required |
| `jikji` | object \| `false` | `{}` | Jikji find-first discovery; `false` disables |
| `everything` | object \| `false` | enabled on Windows | Windows-only Everything file-name search; `false` disables |
| `fsearch` | object \| `false` | enabled on macOS/Linux | `fsearch-cli` file-name search; `false` disables |
| `webSearch` | object | enabled | `web_search` + `web_fetch` tools |
| `jev` | object | `{backend:<default>}` | The required decision model |
| `queryDecomposition` | object \| `false` | `{model:<default>}` | Question decomposition; `false` lets the session model decompose |
| `permissionMode` | `"auto"` \| `"ask"` \| `"full"` | `"auto"` | Approval gate for file changes by the agent |
| `rerank` | object \| `false` | absent (off) | Post-merge reranking; opt-in |
| `parserOptions` | object | `{}` | Parser and OCR tuning |
| `dupey` | object | `{enabled:true}` | Exact/near-duplicate scan settings |
| `excludeExactDuplicates` | `boolean` | `true` | Drop exact-duplicate copies from the local index |
| `filee` | object | enabled | `filee-cli` behind the conversion tools; `enabled:false` removes them |
| `excludePaths` | `string[]` | `[]` | Files/dirs (absolute or workspace-relative, or a glob) omitted from local indexing |
| `limits` | object | shipped defaults | Hard caps on retrieval, prefetch, and candidate lists |
| `refresh` | object | shipped defaults | Refresh limits, retry tuning, and the watch schedule |
| `datasources` | object | `{}` | Datasource skills: name → config or `boolean` |

### model

| Field | Type | Default | Meaning |
|---|---|---|---|
| `provider` | `string` | required | Provider identity for auth lookup and `Model.provider` (e.g. `openrouter`, `fireworks`) |
| `id` | `string` | required | Wire model id sent to the provider API |
| `name` | `string` | `id` | Display name |
| `api` | string | `openai-completions` | Wire format when `baseUrl` is set: `openai-completions`, `openai-responses`, `anthropic-messages`, `openai-codex-responses`, `azure-openai-responses` |
| `baseUrl` | `string` | catalog endpoint | Endpoint base URL; overrides only the catalog endpoint for a known `provider/id` |
| `apiKeyEnv` | `string` | `${PROVIDER}_API_KEY` when `baseUrl` is set | Environment variable holding the key (**name only**). Rejected for provider `dazzi` |
| `reasoning` | `boolean` | — | Whether the model emits reasoning |
| `input` | `("text"\|"image")[]` | `["text"]` | Accepted input modalities |
| `contextWindow` | integer > 0 | catalog | Context window |
| `maxTokens` | integer > 0 | catalog | Max output tokens |

The same shape is used by `jev.sllm.model` and `queryDecomposition.model`.

### minSync

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | `true` | `true` | Accepted for compatibility; `false` is rejected |
| `autoInstall` | `boolean` | `true` | Download/install the MinSync binary into the workspace cache when missing |
| `workspacePath` | `string` | workspace | Override where MinSync stores its workspace |
| `maxChunkSize` | integer > 0 | MinSync default | Chunk size limit |
| `installer` | object | — | Installer tuning forwarded to the MinSync binary fetcher |
| `embedder` | object | native Qwen3-Embedding-0.6B | Vector embedder (below) |

`minSync.embedder` fields (unknown keys are rejected):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | `string` | `native:Qwen/Qwen3-Embedding-0.6B` | Embedder id; `none` selects a sparse-only SPLADE index |
| `baseUrl` | `string` | — | Remote embedder endpoint |
| `apiKeyEnv` | `string` | — | Env var holding the bearer key (**name only**) |
| `dimension` | integer > 0 | — | Vector dimension |
| `queryPrefix` | `string` | — | Prefix added to queries |
| `passagePrefix` | `string` | — | Prefix added to passages |
| `timeoutMs` | integer > 0 | — | Per-request timeout |
| `batchSize` | integer > 0 | — | Batch size |
| `maxRetries` | integer > 0 | — | Retry budget |
| `maxConcurrent` | integer > 0 | — | Concurrency |

### jev

The decision model is required; an absent section or `{}` uses the default backend (`dazzi` when signed in to Dazzi Models, else `openrouter`).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `backend` | `string` | `dazzi` \| `openrouter` | `dazzi`, a pi provider with decision models (`openrouter`, `openai`, `typesafe`, `vercel-ai-gateway`, `cloudflare-workers-ai`, `opencode`), or `sllm` |
| `model` | `string` | backend default | Decision model id within the backend. Not valid with `backend:"sllm"` |
| `confidenceThreshold` | number 0–1 | per-source | Escalate verdicts below this confidence |
| `sllm.model` | `AgentModelConfig` | decomposition model | The small chat LLM used with `backend:"sllm"` or as the fallback |
| `fallback` | `boolean` | `true` | Move to the sLLM when Jev is unavailable; `false` fails the session. Ignored with `backend:"sllm"` |

### rerank

Opt-in: reranking runs only when `model` is named. A block without a model is treated as off and reported once as a `rerank-disabled` diagnostic.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | `boolean` | — (`false` disables) | Turn the block off without deleting it |
| `provider` | `"openrouter"` \| `"dazzi"` | `openrouter` | Provider |
| `model` | `string` | — (required to enable) | Wire model id |
| `apiKeyEnv` | `string` | `OPENROUTER_API_KEY` for `openrouter` | Env var holding the key (**name only**). Rejected for `dazzi` |
| `baseUrl` | `string` | provider default | Override the provider base URL |
| `topN` | integer > 0 | `25` | Return only the top N merged results |
| `timeoutMs` | integer > 0 | provider default | Per-request timeout |

### everything

Windows only; ignored on other platforms.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | `boolean` | `true` | Enable the bundled file-name search |
| `timeoutMs` | integer | test seam | Search timeout |
| `startupTimeoutMs` | integer | test seam | Startup timeout |
| `indexTimeoutMs` | integer | test seam | Index timeout |

### fsearch

macOS/Linux only; ignored on other platforms.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | `boolean` | `true` | Enable `fsearch-cli` file-name search |
| `binaryPath` | `string` | on `PATH` | Path to the client |
| `timeoutMs` | integer | test seam | Search timeout |
| `startupTimeoutMs` | integer | test seam | Startup timeout |
| `indexTimeoutMs` | integer | test seam | Index timeout |
| `watch` | `boolean` | `true` | Keep a `fsearch-cli watch` daemon live |

Without `fsearch-cli`, name search falls back to a bounded filesystem walk.

### webSearch

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | `boolean` | `true` | Enable `web_search` and `web_fetch`; `false` disables both |
| `provider` | `string` | `auto` | Preferred provider (e.g. `gemini`, `parallel`, `searxng`) |
| `order` | `string[]` | shipped chain | Provider order |
| `exclude` | `string[]` | — | Providers to skip |
| `timeoutSeconds` | integer 1–300 | `60` | Per-provider transport timeout |
| `fetch.enabled` | `boolean` | `true` | Enable `web_fetch` |
| `fetch.timeoutSeconds` | integer 1–300 | — | Page fetch/render timeout |

The default chain is credential-free: model-native providers reuse the agent's own model credentials; a self-hosted `SEARXNG_ENDPOINT` is the only env-gated option. Set `"fetch": false` to disable page fetching alone.

### limits

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mergedEvidenceCeiling` | integer > 0 | shipped | Cap on merged evidence |
| `singleDatasourceTopK` | integer > 0 | shipped | Per-datasource result cap |
| `minSyncTopK` | integer > 0 | shipped | MinSync result cap |
| `minSyncScopedQueryTopK` | integer > 0 | shipped | MinSync cap for a scoped query |
| `toolDescriptionInstanceScopes` | integer > 0 | shipped | Instance scopes listed per tool description |
| `prefetch.jikjiTopK` | integer > 0 | shipped | Jikji prefetch cap |
| `prefetch.minSyncTopK` | integer > 0 | shipped | MinSync prefetch cap |
| `prefetch.datasourceTopK` | integer > 0 | shipped | Datasource prefetch cap |
| `prefetch.jikjiPathLimit` | integer > 0 | shipped | Jikji path cap |
| `prefetch.sectionLimit` | integer > 0 | shipped | Section cap |
| `recordCut.enabled` | `boolean` | shipped | Toggle record cutting |
| `recordCut.depth` | integer 1–max | shipped | Fusion depth |
| `recordCut.newRecordShare` | number 0–1 | shipped | New-record share |
| `recordCut.perRecordFloor` | integer > 0 | shipped | Per-record floor |
| `recordCut.perRecordDivisor` | integer > 0 | shipped | Per-record divisor |
| `recordCut.resultChars` | integer > 0 | shipped | Result character budget |
| `recordCut.minReadChars` | integer > 0 | shipped | Minimum read characters |

Unknown `limits.*` keys are rejected.

### refresh

| Field | Type | Default | Meaning |
|---|---|---|---|
| `maxDeleteRatio` | number 0–1 | shipped | Delete-ratio valve for the parsed mirror |
| `maxScannedEntries` | integer > 0 | shipped | Walk cap |
| `maxDepth` | integer ≥ 0 | shipped | Walk depth cap |
| `transientRetries` | integer ≥ 0 | `3` | Per-file retries for `EMFILE`/`EBUSY`/`EIO`/`ETIMEDOUT` |
| `cloudMaxFiles` | integer > 0 | shipped | Cloud placeholder file cap |
| `cloudMaxBytes` | integer > 0 | shipped | Cloud placeholder byte cap |
| `cloudEvictAfterParse` | `boolean` | shipped | Evict a cloud placeholder after parsing |
| `cloudEvictRetries` | integer ≥ 0 | shipped | Placement retry budget |
| `schedule` | object | — | Watch schedule: `{ power, windows, days }` (see [Scheduling and exclusions](../indexing/scheduling-and-exclusions.md)) |

`refresh.schedule` drives `autorag watch`; the `--power`/`--window`/`--days` flags override it. Invalid schedule values raise a `ConfigError` carrying the underlying validation message.

### parserOptions

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ocr` | object | off | OCR tuning; `ocr.enabled` is the single opt-in |
| `kordoc` | object | — | kordoc parser tuning; an explicit `kordoc.ocr` always wins |
| `languages` | `string[]` | global `languages` | Parser-local language override |
| `fileTimeoutMs` | integer | `120000` | Hard per-file parse deadline; a parse past it is skipped as `parser-failed` |
| `maxHeapMb` | integer | — | V8 old-generation heap limit for the parse worker |

### dupey, filee

| Key | Field | Type | Default | Meaning |
|---|---|---|---|---|
| `dupey` | `enabled` | `boolean` | `true` | Enable duplicate scanning |
| `dupey` | `binaryPath` | `string` | on `PATH` | Path to `dupey` |
| `dupey` | `timeoutMs` | integer | — | Scan timeout |
| `filee` | `enabled` | `boolean` | `true` | `false` removes the conversion tools |
| `filee` | `binaryPath` | `string` | on `PATH` | Path to `filee`/`filee-cli` |
| `filee` | `autoInstall` | `boolean` | `true` | Download the pinned filee-cli into `<AUTORAG_HOME>/tools` |
| `filee` | `timeoutMs` | integer | — | Conversion timeout |
| `filee` | `dataDir` | `string` | Filee's per-user folder | Folder for Filee's own settings and engines |

### datasources

`datasources` maps a skill name (or connection alias) to a config object or `boolean`. `false` and `enabled:false` skip the entry; a configured name with no matching skill is reported by `autorag setup` and skipped.

| Field | Type | Meaning |
|---|---|---|
| `enabled` | `boolean` | Turn this entry off |
| `description` | `string` | Operator-authored context shown to the agent |
| `type` | `string` | Built-in template used when the key is a connection alias |
| `instanceId` | `string` | Stable instance id |
| `pollingIntervalMs` | integer | Poll interval |
| `tags` | `string[]` | Descriptive tags |
| `connector` | object | Connector-specific options (token env names, repos, feeds, paths) |
| `channels.ids` / `channels.names` | `string[]` | Chat/channel allowlist; absent means all |

Secrets never appear in the config: a connector references a credential by environment-variable **name** (for example `connector.tokenEnv: "SLACK_BOT_TOKEN"`). See [Connecting datasources](../datasources/connecting.md) and the per-datasource pages for connector keys.

## Minimal example

```json
{
  "searchPaths": ["/path/to/docs"],
  "workspacePath": "/path/to/docs",
  "memoryPath": "/path/to/.autorag/memory.json",
  "languages": ["ko", "en"],
  "minSync": { "enabled": true, "autoInstall": true }
}
```

## Full annotated example

```json
{
  "searchPaths": ["/path/to/docs", "/path/to/notes"],
  "workspacePath": "/path/to/docs",
  "memoryPath": "/path/to/.autorag/memory.json",
  "languages": ["ko", "en"],
  "excludePaths": ["/path/to/docs/archive", "**/node_modules/**"],

  "model": { "provider": "<provider>", "id": "<model-id>" },
  "minSync": {
    "enabled": true,
    "autoInstall": true,
    "maxChunkSize": 1500,
    "embedder": {
      "id": "native:Qwen/Qwen3-Embedding-0.6B",
      "dimension": 1024,
      "queryPrefix": "Instruct: retrieve\nQuery:",
      "batchSize": 32
    }
  },

  "jikji": { "autoInstall": true, "timeoutMs": 60000 },
  "everything": { "enabled": true },
  "fsearch": { "enabled": true, "watch": true },
  "webSearch": { "enabled": true, "provider": "auto", "timeoutSeconds": 60 },

  "jev": { "backend": "openrouter", "model": "<decision-model-id>", "fallback": true },
  "queryDecomposition": { "model": { "provider": "<provider>", "id": "<model-id>" } },
  "rerank": {
    "provider": "openrouter",
    "model": "<rerank-model-id>",
    "apiKeyEnv": "OPENROUTER_API_KEY",
    "topN": 25
  },
  "permissionMode": "auto",

  "parserOptions": { "ocr": { "enabled": false }, "fileTimeoutMs": 120000 },

  "dupey": { "enabled": true },
  "excludeExactDuplicates": true,
  "filee": { "enabled": true, "autoInstall": true },

  "limits": { "minSyncTopK": 50, "prefetch": { "datasourceTopK": 20 } },
  "refresh": { "transientRetries": 3, "schedule": { "power": "any", "days": ["mon", "tue", "wed", "thu", "fri"] } },

  "datasources": {
    "github": { "enabled": true, "connector": { "repos": ["owner/repo"], "tokenEnv": "GITHUB_TOKEN" } },
    "obsidian": { "enabled": true, "connector": { "vaultPath": "/path/to/vault" } }
  }
}
```

## Validation and rejected values

Malformed config fails fast with a `ConfigError`; the command exits `2`. Unknown keys are rejected rather than ignored. Examples of the error classes:

- `Unknown flag: --<name>` / `Unknown command: <name>` — argument parser.
- `Config file not found: <path>` — explicit `--config`/`AUTORAG_CONFIG` path missing.
- `Config file must be a JSON object` — the file is not an object.
- ``Config field 'excludePaths' must be a non-empty string array``.
- `MinSync is required and cannot be disabled; remove `minSync: false` / `minSync.enabled: false` from the config.`
- `jev.enabled is no longer supported: ...` — the old `enabled` switch was removed; the decision model is required.
- `--permission-mode must be one of: ask, auto, full`.
- `rerank.provider must be one of: openrouter, dazzi`.

::: warning **Secrets**
Never put API keys, tokens, or personal paths in the config. The only credential-shaped field is an environment-variable **name** (`apiKeyEnv`, `connector.tokenEnv`, `connector.*Env`); the value stays in the environment. See [Environment variables](environment-variables.md).
:::

## See also

- [CLI commands](cli.md) for the flags that override these keys.
- [Environment variables](environment-variables.md) for the environment half of precedence.
- [Indexing overview](../indexing/overview.md) and [Scheduling and exclusions](../indexing/scheduling-and-exclusions.md).
- [Connecting datasources](../datasources/connecting.md).
- [TypeScript API](typescript-api.md) for `resolveConfig` and `writeDefaultConfig`.
