---
title: "Connecting a datasource"
description: "How to declare datasource connections, the config keys, aliases, shared embedding, sync timeouts, index retries, and the access model."
---

# Connecting a datasource

This page shows how to declare a datasource connection in trusted config, what each key means, and how connection aliases, shared embedding, timeouts, and retries behave. Read it before adding a `datasources` entry.

## Config entry shape

A connection lives under `datasources.<name>`. The key is the connection's datasource id (used for tools, sources, and provenance); the value is an object:

```json
{
  "datasources": {
    "work-slack": {
      "type": "slack",
      "instanceId": "work",
      "enabled": true,
      "description": "Engineering and product channels for the current project.",
      "pollingIntervalMs": 900000,
      "tags": ["slack", "chat"],
      "connector": { "configPath": "/secure/slacrawl.toml", "workspace": "T0123456789" },
      "channels": { "ids": ["C0123"], "names": ["release-engineering"] }
    }
  }
}
```

| Key | Type | Meaning |
|---|---|---|
| `type` | string | Builtin template to build when the key is an alias. Absent means the key itself is the template name. |
| `instanceId` | string | Instance id inside the skill. Defaults to `default`, or to the key when `type` is set. |
| `enabled` | boolean | `false` skips the entry silently. |
| `description` | string | Operator-authored context shown to the agent (see below). |
| `pollingIntervalMs` | number | Poll interval passed to the skill; the default is normally 15 minutes. |
| `tags` | string[] | Descriptive metadata only, never authority. |
| `connector` | object | Template-specific options (tokens, paths, repos, feeds, sidecar). |
| `channels.ids` / `channels.names` | string[] | Chat/channel allowlist for chat crawlers; absent means every channel. |

A bare `true` value (`"telegram": true`) is accepted as an empty config object. A value of `false` disables the entry. A value that is not `true`, `false`, or an object is reported as unknown.

## Connection aliases

Any entry may reuse a builtin template through `type`, so one provider can back several independent connections. The alias key becomes the independent datasource id: its source scope, diagnostics, method names, and local storage namespace are all rewritten under the alias.

```json
{
  "datasources": {
    "personal-mail": { "type": "mailcrawl", "connector": { "account": "personal", "mailbox": "INBOX" } },
    "company-slack": { "type": "slack", "connector": { "configPath": "/secure/company-slack.toml" } },
    "family-kakao": { "type": "kakao", "channels": { "names": ["Family"] }, "connector": { "binaryPath": "lazykatok" } }
  }
}
```

The alias also becomes the model-visible tool name: `personal-mail` yields `search_datasource_personal_mail`, and `autorag.datasources.list` reports the id.

## Operator-authored descriptions

`description` is trusted operator context shown in the descriptor and the progressive-disclosure skill manifest. It does not change what is searchable. When Jev is enabled it is what the pre-answer datasource check reads to decide whether to search the connection ([Jev](../retrieval/jev.md)), so write it from what the connection actually holds — channels or rooms, people, topics, time range. Without it, Jev sees only the backend's generic text (for example "Discord datasource via the external discrawl CLI").

## Hierarchical instances

A skill can publish one or more instances (a Slack workspace, a KakaoTalk account, an email account). Every instance maps to a slash-hierarchical root `/<skill>/<instance>` and chunks hang under `/<skill>/<instance>/chunks/<id>`. These paths are opaque datasource identities, not filesystem paths — see [Search semantics](./search-semantics.md).

## Checking a connection with setup

`autorag setup --format json` probes the embedding backend and every known datasource and prints one entry per datasource:

```bash
autorag setup --format json
```

Each entry is `{ "name", "state", "reason" }`:

| State | Meaning |
|---|---|
| `configured` | Binary (if any), native store, and credentials are present, and an embedding backend is available for semantic datasources. |
| `skipped` | Not ready. `reason` names one of: `not configured`, `disabled by operator`, `not supported on <platform>`, `missing <binary> binary`, `credential <ENV> is unavailable`, `native store is not present`. |
| `blocked` | Configured and present, but a semantic datasource has no embedding backend (`embedding model is unavailable`). |

The report's top-level `ok` is true when semantic mode is available or at least one datasource is `configured`; the command exits 1 otherwise. `reason` strings are path-opaque. Setup never writes a datasource entry for you; add entries to trusted config yourself.

## Shared embedding runtime and native ownership

AutoRAG owns no embedding model and bundles no native runtime. im-not-llm provisions the embedder (`Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions) in its own cache, and one shared embedder serves everything that embeds inside AutoRAG: MinSync indexing and query, memory similar-question search, the chat-crawler sidecars, and the GitHub Gist vector index. It resolves a configured remote endpoint first, then a running `inl serve` daemon, then MinSync's in-process engine. See [Embedding backends](../indexing/embedding-backends.md).

It is a provider boundary, not a shared archive: each datasource keeps its native archive, chunking, credentials, metadata, vector store, and source identity. Embedding requests contain text and the selected model only — no archive ids, source paths, or credentials.

Two external CLIs embed for themselves and are pointed at the same backend through their own environment variables: `mailcrawl` (with a remote opt-in) and `lazykatok` (OpenAI-compatible only). `discrawl` can receive a managed `[search.embeddings]` section when no explicit `connector.configPath` is set. Everything else that embeds does so through the AutoRAG-owned embedder. If the embedder is unavailable, a datasource keeps its lexical/FTS lane where supported and reports a diagnostic.

## Connector sync/index timeouts

A connector `sync`/`index` step is not interactive, so AutoRAG gives it a one-hour budget (`connector.indexTimeoutMs`) rather than the short interactive default (`connector.timeoutMs`) used by `search`, `doctor`, and chunk reads. Precedence is `indexTimeoutMs`, then `timeoutMs`, then the one-hour default. Measured first-run costs:

| Connector | Workload | Measured |
|---|---|---|
| `mailcrawl` | Gmail INBOX full sync (~2,300 messages) + native index | ~12 min |
| `kakao` | Semantic index over a ~50k-message archive (~9,373 units) | ~11 min |

A step killed by its timeout does not commit its cursor, so the next refresh restarts from scratch instead of resuming; the diagnostic for a timeout carries a hint naming `connector.indexTimeoutMs`. Raise the budget per connection:

```json
{
  "datasources": {
    "family-kakao": {
      "type": "kakao",
      "connector": { "binaryPath": "lazykatok", "indexTimeoutMs": 3600000 }
    }
  }
}
```

## Index retry and backoff

A refresh retries a failed or thrown `index()` with jittered exponential backoff: 2 retries (3 calls in total), about 1 s then 2 s, capped at 30 s. Failures are not classified — error codes differ per datasource, so every failure is retried alike. Once the retries are used up, that datasource is reported as failed and the refresh moves on to the next one. A timed-out step still restarts from scratch each attempt.

The result gains `attempts` (total `index()` calls; absent on a first-time success). An exhausted failure's message ends with `(gave up after N attempts)`; a late success adds an info diagnostic naming the last error. Override the policy when constructing `AutoRAGAgent` with `datasourceRetry: { maxRetries, baseDelayMs, maxDelayMs }`; `maxRetries: 0` disables retrying.

## The datasource contract

A skill implements:

- `describe()` — a descriptor with `name`, `type`, `description`, `capabilities`, `tags`, and status.
- `index(options?)` — an ok/fail union that never throws; remote embedding egress is rejected here with a `datasource-embedding-egress-rejected` diagnostic.
- `polling()` — polling/indexing metadata.
- `retrievalMethods()` — the methods fed into the shared retriever.
- `describeSources()` — opaque source descriptions for the current evidence set.
- `skillManifest()` — the on-demand agent skill body.

Skills are server-bound: they never derive authority from model or tool arguments.

## Indexing metadata

`PollingMetadata` supports `mode: "none"` (manual only), `mode: "poll"` with `intervalMs`, and `mode: "cron"` with `cronExpr`. Current AutoRAG performs global refresh ticks and lets each skill decide what work is due; cron is validated and declared but not scheduled by AutoRAG itself.

## Access model and security

Every configured or already-connected datasource is searchable with no access setup: there is no tag/scope authorization layer and no default-deny gate. Trusted config decides which connections exist; each one gets its own generated `search_datasource_<id>` tool. Descriptor `tags` are descriptive metadata only, and tool arguments cannot change which connections exist. `scope` narrows results later in the pipeline, never widens what a connection reaches.

Retrieval results, diagnostics, and metadata are intentionally traceable: they carry real file paths, account identifiers, and message excerpts verbatim, and AutoRAG does not redact or opacify datasource content. If that content must not leave the machine, run AutoRAG with a local LLM instead of a cloud provider.
