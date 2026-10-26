# Datasource skills

CLI-backed datasource skills run their native binaries directly and use
operator-provided configuration paths or each CLI's own default store.

Datasource skills let AutoRAG search external, server-configured sources while preserving the same retrieval and curation model used for local document collections.

### ClawGallery

ClawGallery is a CLI-backed datasource for local screenshots and photos. Install
the upstream CLI with `cargo install clawgallery`, then configure a trusted
connection:

```json
{
  "datasources": {
    "screenshots": {
      "type": "clawgallery",
      "instanceId": "personal",
      "connector": {
        "binaryPath": "clawgallery",
        "syncVisual": true,
        "vdrBackend": "vsplade"
      }
    }
  }
}
```

Refresh runs ClawGallery's incremental `bootstrap` and, when enabled, its
trusted `vdr sync`. Keyword, V-SPLADE lexical, dense embedding, and hybrid
search are delegated to `clawgallery search --json`; AutoRAG never reads
`images.jsonl` or `vdr.sqlite3`, and does not trigger captioning or renaming.

### Spotlight

`spotlight` is a macOS-only connector skill. It drives the built-in `mdfind`
CLI (no extra install). Indexing re-runs configured Spotlight queries and
hydrates file text; result metadata carries the real absolute path. Grant Full
Disk Access to the host app when searching Mail, Messages, Safari, or other
protected locations. Unavailable on non-macOS hosts.

```json
{
  "datasources": {
    "mac-files": {
      "type": "spotlight",
      "instanceId": "local"
    }
  }
}
```

### Obsidian

`obsidian` indexes a markdown vault with MinSync — the same engine and
`minSync.*` binary/embedder configuration used for local files — in its own
MinSync workspace at
`<workspace>/.autorag/datasources/obsidian/<instanceId>/minsync`, so vault notes
never mix with the local-file index. There is no external CLI.

```json
{
  "datasources": {
    "obsidian": {
      "connector": { "vaultPath": "/path/to/vault" }
    }
  }
}
```

Only `.md` and `.markdown` notes are staged. Dot-prefixed folders and files
(`.obsidian`, `.trash`, `.git`), `node_modules`, and symlinks are skipped, and a
re-index is incremental: only changed or deleted notes are touched. The former
`connector.binaryPath` and `connector.configPath` options no longer exist.
Retrieval exposes `obsidian-bm25` (lexical) and `obsidian-semantic` (vector,
through the shared embedder), and both accept an ordinary `scope`. Sources are
vault-relative note paths as opaque slash paths —
`/obsidian/<instanceId>/<folder>/<note>.md`, for example
`/obsidian/default/Projects/Roadmap.md`, with `#` shown as `%23`. Folder
filtering is the ordinary `scope`: `scope=/obsidian/default/Projects` matches
that folder and everything under it, and a scope that already contains `*` is
matched literally as written, so a nested-folder glob needs an explicit trailing
`/**`, for example `/obsidian/default/**/Daily/**`. Result metadata `path`
carries the real note path on disk and `file` the vault-relative path. The index
is AutoRAG-owned: to reset it, delete
`<workspace>/.autorag/datasources/obsidian/<instanceId>` and run
`autorag refresh --method datasources`.

## Shared embedding runtime and native ownership

AutoRAG owns no embedding model and no bundled native runtime. MinSync provisions
the embedder (`Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions) in its own cache
(`~/.cache/minsync/models`), and one shared embedder — a small module AutoRAG
owns — serves everything that embeds inside AutoRAG. It is a provider boundary,
not a shared archive or vector-store boundary: each datasource keeps its native
archive, chunking, credentials, metadata, vector store, generation publication,
and source identity.

The shared embedder resolves in order: a configured remote endpoint
(`minSync.embedder`), then a running `minsync serve` daemon (loopback only), then
MinSync's in-process engine via the optional `@nomadamas/minsync-native` addon.
The daemon and in-process vectors are identical, so stores keyed on embedder
identity do not re-embed when the daemon appears or disappears.

The zero-configuration boundary is narrow:

- **MinSync** uses the shared embedder by default. AutoRAG supplies the
  endpoint; MinSync owns `.minsync`, CDC chunks, vectors, and reindex
  decisions.
- **discrawl** can receive a managed native `[search.embeddings]` section. When
  no explicit `connector.configPath` is supplied, AutoRAG writes only
  `.autorag/datasources/discrawl/config.toml`, preceded by the exact marker
  `# AutoRAG managed discrawl embeddings v1`. It writes `provider`
  (`openai_compatible`, with an underscore; discrawl speaks OpenAI
  `POST {base_url}/embeddings` only, so a TEI remote leaves discrawl on its own
  config), `model`, `base_url` (the daemon or remote `/v1` URL), `dimensions`,
  and `api_key_env` (the **name** of the environment variable; AutoRAG passes
  only that one variable through to discrawl's environment). It checks discrawl
  metadata before asking the native CLI to rebuild. An explicit `configPath` is
  authoritative and is never rewritten.
- **mailcrawl** (`@nomadamas/mailcrawl@0.2.2` or newer) and **kakao**
  (`lazykatok` 0.3.3 or newer) embed for themselves, so AutoRAG points them at
  the same backend through the environment variables each CLI documents, for
  `index` and for semantic and hybrid search only (BM25 never embeds).
  - mailcrawl runs its `openai` or `tei` provider (`MAILCRAWL_EMBEDDER_PROVIDER`,
    `MAILCRAWL_EMBED_URL`, `MAILCRAWL_EMBED_MODEL`, `MAILCRAWL_EMBED_DIM`) at a
    daemon or remote endpoint. A non-loopback remote also sets
    `MAILCRAWL_EMBED_ALLOW_REMOTE`, which mailcrawl requires and which is only
    set because the operator configured that remote.
  - lazykatok runs its OpenAI-compatible embedder (`KATOK_EMBEDDER`,
    `KATOK_EMBED_URL`, `KATOK_EMBED_MODEL`, `KATOK_EMBED_DIM`). It speaks
    `/v1/embeddings` only, so a TEI remote is reported as an error rather than
    silently falling back to fastembed.
  - A remote key is exported to that one child as the value of the variable
    `apiKeyEnv` names (`MAILCRAWL_EMBED_API_KEY_ENV` for mailcrawl,
    `KATOK_EMBED_API_KEY` for lazykatok); no other credential is passed.
  - Neither CLI falls back to its own model, and AutoRAG starts no daemon for
    them: with no daemon and no remote endpoint, `index` and the semantic call
    fail with "no embedding endpoint is available" while BM25 and `sync` keep
    working. Start `minsync serve --idle-timeout 0` first (see
    `docs/minsync-setup.md`), so the first `index` finds it.
  - mailcrawl identity is provider + model + dimension, and lazykatok records
    vectors under `openai/<model>`, so an archive built with the old native
    model rebuilds on the next `index` (`mailcrawl repair --semantic`,
    `lazykatok wipe-index` then `index` force it).
- **telegram**, **whatsapp**, and **slack** CLIs have no embedding; their
  AutoRAG sidecars use the shared embedder.
- **github-gist** semantic retrieval uses the shared embedder;
  `connector.semantic: false` keeps it lexical-only.
- **obsidian** semantic retrieval runs through MinSync with the shared embedder,
  in the datasource's own workspace under
  `.autorag/datasources/obsidian/<instanceId>/minsync`.
- **clawgallery** is untouched: it retains its VDR/native retrieval.

Embedding requests contain text and the selected model only. AutoRAG does not
send archive IDs, source paths, credentials, or native store paths to the
embedder. If the embedder is unavailable, a datasource keeps its native
lexical/FTS lane where supported and reports a diagnostic; a remote embedding
service is used only when the operator explicitly configures one.
## Contract

A datasource skill is both:

1. an indexing hook (`index()` plus `polling()` metadata); and
2. a retrieval method factory (`retrievalMethods()`).

The methods are registered in the normal AutoRAG pipeline:

```text
RetrievalMethodRegistry
  -> ParallelRetriever
  -> filterDatasourceScope
  -> ResultMerger
  -> memory / curation
```

A skill must also provide `describeSources()` entries so the librarian prompt can explain what data exists.

Every search-tool call counts toward the per-search tool budget. Calls whose
details carry a `results` array also join the run's retrieval trace and evidence
origins. Memory is written only from the Jev-judged evidence cited in the final
answer, never from how an individual search call turned out.

## Connector sync/index timeouts

Connector `sync`/`index` steps are not interactive, so AutoRAG gives them a
one-hour budget (`connector.indexTimeoutMs`) instead of the 60-second
interactive default (`connector.timeoutMs`) used by `search`, `doctor`, and chunk
reads. Measured first-run costs on a real archive:

| Connector | First-run workload | Measured |
|---|---|---|
| `mailcrawl` | Gmail INBOX full sync (~2,300 messages) + native index | ~12 min |
| `mailcrawl` | No-op `sync` of a 1,000-message INBOX (mailcrawl re-reads every message, NomaDamas/mailcrawl#48) | 16 to 27 min |
| `kakao` (lazykatok) | Semantic index over a ~50k-message archive (9,373 units) | ~11 min |

A step killed by its timeout does not commit its cursor, so the next refresh
restarts from scratch instead of resuming — the failure loops rather than makes
progress. Raise the budget per connection:

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

`indexTimeoutMs` takes precedence, then `timeoutMs`, then the one-hour default.
A `sync`/`index` diagnostic ending in `timeout` carries a hint naming
`connector.indexTimeoutMs` and the restart behaviour.

## Index retry and backoff

A refresh retries a failed (or thrown) datasource `index()` with jittered
exponential backoff: 2 retries (3 calls in total), about 1 s then 2 s, capped at
30 s. Failures are not classified: error codes differ per datasource, so every
failure is retried alike. Once the retries are used up that datasource is
reported as failed and the refresh moves on to the next one. A timed-out step
still restarts from scratch each attempt (see above), so a first sync that needs
longer than the budget still fails after three tries — raise
`connector.indexTimeoutMs`.

The result gains `attempts` (total `index()` calls; absent on a first-time
success). An exhausted failure's message/last diagnostic ends with
`(gave up after N attempts)`; a late success adds an info diagnostic naming the
last error. Override with `AutoRAGAgent` option
`datasourceRetry: { maxRetries, baseDelayMs, maxDelayMs }`; `maxRetries: 0`
disables retrying. Process-tree termination of a timed-out CLI is tracked
separately (#1587, #1586).

## mailcrawl

The `mailcrawl` datasource delegates local email synchronization and search to
the external `mailcrawl` CLI. Install `@nomadamas/mailcrawl@0.2.2` or newer
(Node.js 24+) and configure Himalaya separately; AutoRAG never opens
`archive.sqlite` directly. 0.1.3 and earlier fail a repeated `index` after a
no-op sync (`text array must be non-empty`). By default, mailcrawl uses its
native archive. Set `connector.dataDir` only when the operator explicitly
wants a different mailcrawl data directory. The semantic index embeds through
the shared backend (see the embedding section above); `connector.env` embedding
overrides are replaced for `index` and semantic search.

0.2.0 replaced the semantic store and embedder contract:

- Vectors live in a LanceDB table (`<data-dir>/semantic.lance`) with the
  embedder identity in `semantic.identity.json`. A mismatch rebuilds the
  table, and a search against a different embedder fails loudly instead of
  returning silently wrong neighbours.
- The default embedder is the in-process native `Qwen/Qwen3-Embedding-0.6B`
  (1024 dimensions, the same model and dimension as AutoRAG's shared embedder),
  so a cold cache downloads ONNX weights on the first `index`. AutoRAG gives
  `sync`/`index` a one-hour budget (`connector.indexTimeoutMs`) while search
  keeps its 60-second interactive budget.
- `index` reports `{ embedded, reused, archiveRevision, rebuilt, embedder }`
  instead of the 0.1.x generation report.
- An explicit `loopback-http` embedding endpoint is supported through
  `connector.env` (`MAILCRAWL_EMBEDDER_PROVIDER`, `MAILCRAWL_EMBED_URL`,
  `MAILCRAWL_EMBED_MODEL`, `MAILCRAWL_EMBED_DIM`). AutoRAG forwards only
  loopback endpoints and refuses a non-loopback URL before spawning the CLI.

```json
{
  "datasources": {
    "mailcrawl": {
      "instanceId": "personal",
      "connector": {
        "binaryPath": "mailcrawl",
        "account": "personal",
        "mailbox": "INBOX"
      }
    }
  }
}
```

Refresh runs `mailcrawl sync --json` followed by `mailcrawl index --json`.
Retrieval is hybrid-first: one `mailcrawl search --mode hybrid` per query, and BM25 only when hybrid cannot run
(for example when no semantic index exists). Results map to opaque `/mailcrawl/<instance>/chunks/<chunk-id>` sources. Use
`mailcrawl --help` for upstream commands; AutoRAG does not invent a shared
datasource command taxonomy. `mail-export` remains the static `.mbox`/`.eml`
path. Mailcrawl is the sole Gmail, IMAP, and Maildir path.

## Universal connection aliases

Every datasource entry can use a reusable template with a connection alias:

```json
{
  "datasources": {
    "personal-mail": {
      "type": "mailcrawl",
      "connector": { "account": "personal", "mailbox": "INBOX" }
    },
    "company-slack": {
      "type": "slack",
      "connector": { "configPath": "/secure/company-slack.toml" }
    },
    "family-kakao": {
      "type": "kakao",
      "channels": { "names": ["가족방"] },
      "connector": { "binaryPath": "lazykatok" }
    }
  }
}
```

### Operator-authored datasource descriptions

Each configured connection may include an optional `description`. This text is
trusted operator context shown in the datasource descriptor and progressive
disclosure skill manifest. It helps the librarian understand how a connection
is normally used without changing what is searchable. With Jev enabled, it is
also what the pre-fast-answer datasource check reads to decide whether to
search the connection (see [Jev Decisions](jev-decisions.md)), so write it from
what the connection actually holds: channels or rooms, people, topics, and
time range. Jev is told every description is a short, non-exhaustive summary,
so name the main content rather than trying to list everything. Without a
description, Jev sees only the connector's generic text (for example "Discord
datasource via the external discrawl CLI"), which cannot tell two Discord
servers apart.

```json
{
  "datasources": {
    "personal-google-drive": {
      "type": "cloud-drive",
      "instanceId": "personal",
      "description": "Project contracts and government-support documents. Prefer this connection for current agreements; treat Archive/ as historical.",
      "connector": {
        "provider": "google-drive",
        "remote": "personal-gdrive:"
      }
    }
  }
}
```

Descriptions are user-supplied, are not inferred automatically, and are
descriptive context only: they never change which connections are searchable
or widen a query `scope`.

The key is the independent datasource ID and becomes an independently
loadable `datasource-<alias>` skill. Its source scope, diagnostics, method
names, local storage/cache namespace, and descriptive metadata are rewritten
under the alias. This supports multiple connections of the same provider as
well as different providers in one agent.

## Access model

Every configured or already-connected datasource is searchable without access
setup: there is no tag/scope authorization layer and no default-deny gate.
Trusted server/API configuration only decides which connections exist (the
`datasources` section). Each configured connection gets its own generated
`search_datasource_<id>` tool, and each one's schema is exactly:

```ts
{ query: string; topK?: number; scope?: string }
```

There is no datasource fan-out tool: a question that spans every datasource (or everything else) uses `search_all_documents`, which already registers every configured connection's retrieval methods alongside the local ones. Skill descriptor `tags` are descriptive metadata only, never authority, and model-controlled tool arguments cannot change which connections exist.

`scope` is an ordinary per-query narrowing filter for datasource methods that advertise the `scoped` capability. A result from such a method is kept only when its source matches the requested scope, and datasource sources always reject `#` fragments. Datasources without that capability (for example, lazykatok's chat-identity results) own any narrower filtering themselves.

Generated tools, retrieval, and memory provenance reference the datasource IDs of configured connections; provenance is not an authorization decision.

### Migrating from datasource access controls

Datasource authorization was removed. Delete the `datasourceAccess` block; a
connection is searchable as soon as it is configured under `datasources` and
authenticated through its native CLI.

- Remove `datasourceAccess` and its `allowedTags` / `allowedScopes` retrieval fields.
- Remove CLI `--tags` and MCP `tags` filters used for datasource visibility.
- Query `scope` stays: it narrows a scope-capable datasource's results and can never widen visibility.
- Skill/descriptor `tags` remain descriptive metadata only.

## Security responsibility

Retrieval results, diagnostics, and metadata are intentionally traceable: they carry real file paths, account identifiers, and message excerpts verbatim. AutoRAG does not redact or opacify datasource content. If that content must not leave the machine, the operator is responsible for running AutoRAG with a local LLM (e.g. an Ollama-backed model) instead of a cloud provider.

## Indexing metadata

`PollingMetadata` supports:

- `mode: "none"` for manual-only indexing;
- `mode: "poll"` with `intervalMs` for routine refresh checks;
- `mode: "cron"` with `cronExpr` as descriptor metadata.

Current AutoRAG v1 performs global refresh ticks (`agent.refresh()` / auto-refresh) and lets each skill decide what work is due. Cron metadata is validated/declared but not scheduled by AutoRAG yet.

## Hierarchical instances

A skill can publish `instances`, for example:

- Slack workspace -> channel
- Google Drive account -> folder
- KakaoTalk account -> chat corpus
- Notion workspace -> database/page tree

Every instance maps to a slash-hierarchical datasource root like `/kakao/personal` or `/slack/local`. Chunks hang under `/<skill>/<instance>/chunks/<id>`.

## Slack via slacrawl

Slack can use the local Slack Desktop cache through `slacrawl`'s `wiretap`
source; this path does not require a Slack token. API/bot/user tokens are only
needed for server-side history, missing cache data, broader thread coverage,
or DM/MPIM access.

```json
{
  "datasources": {
    "slack-local": {
      "type": "slack",
      "instanceId": "local",
      "description": "Recent work conversations available in this Mac's Slack Desktop cache.",
      "connector": {
        "configPath": "~/.slacrawl/config.toml",
        "syncSource": "wiretap",
        "workspace": "T0123456789",
        "timeoutMs": 120000
      }
    }
  }
}
```

`workspace` is the Slack team id to scope reads to and it is effectively
required. `slacrawl`'s `search` and `messages` read paths return nothing unless
a workspace is named explicitly, even when the archive and its FTS index hold
matching rows — an unscoped search looks exactly like an empty archive, with no
diagnostic. `syncSource` is likewise required: `slacrawl sync` without
`--source` fails and surfaces as `datasource-index-failed`. `timeoutMs` bounds
the FTS search; the first `sync` (a wiretap import of the whole cache) uses
`indexTimeoutMs`, default one hour, so it is not cut off at 60 seconds.

Initialize and refresh the local mirror with:

```bash
slacrawl init -db ~/.slacrawl/slacrawl.db -workspace local
slacrawl sync --source wiretap

# `init -workspace local` only names the local config; it is not a Slack team
# id. List the real ids the wiretap import produced and use one of them as the
# connector's `workspace`:
slacrawl sql 'select id, count(*) from workspaces join messages on messages.workspace_id = workspaces.id group by 1;'
slacrawl search -workspace T0123456789 <term>   # must print rows before wiring it up

autorag refresh --method datasources
```

## Hybrid search for chat crawlers (sidecar)

`slack`, `telegram`, and `whatsapp` get BM25-ranked, semantic, and hybrid retrieval **by default**, through a
**read-only sidecar** kept under the AutoRAG workspace. Without it they only have the CLI's plain FTS5 query, and
slacrawl returns the *newest* matches and ANDs every term, so a natural-language question often matches nothing.

No configuration is needed:

```json
{ "datasources": { "slack-work": { "type": "slack", "connector": { "workspace": "T0123456789" } } } }
```

For these three crawlers, tune or opt out per datasource with `connector.sidecar`:

| `connector.sidecar` | Meaning |
|---|---|
| absent / `true` / `{}` | One hybrid method per connection; BM25 then the CLI search only if it throws (default) |
| `false` or `{ "enabled": false }` | CLI-backed FTS only, as before (kakao: hybrid, then BM25) |
| `semantic: false` | BM25 only; no embedding calls, and no `<name>-hybrid` method (the CLI search is the fallback) |
| `crawlerDbPath` | archive to read. Otherwise: `connector.databasePath`; for `slack`, the `db_path` in the slacrawl config that `connector.configPath` names (relative paths resolve against that file); otherwise the crawler default (`~/.slacrawl/slacrawl.db`, `~/.telecrawl/telecrawl.db`, `~/.wacrawl/wacrawl.db`) |
| `batchSize`, `maxEmbedPerSync` | embedding request size (256) and the most messages one refresh embeds (16,384) |

How it works and what it never does:

- The crawler archive is opened **read-only** (`ATTACH ... mode=ro`). AutoRAG never writes to it, so a crawler
  that migrates, purges, vacuums, or rebuilds its file cannot lose sidecar state. SQLite may leave `-wal`/`-shm`
  files beside a WAL-mode archive after any read-only open; the archive file itself is unchanged.
- The sidecar owns its own file, `<workspace>/.autorag/datasources/<name>/<instance>/sidecar.sqlite`. It stores
  only `message key -> vector`. Message text is always read from the crawler archive at query time.
- It is created lazily: building a skill creates no file, directory, or handle. Nothing is written until the
  crawler archive exists and a search or refresh actually runs. Without an AutoRAG workspace there is nowhere to
  keep it, so the datasource behaves as before.
- One method per connection, `<name>-hybrid` (reciprocal rank fusion, k=60): it runs the sidecar's hybrid search
  and never also runs the CLI or the sidecar's BM25 and semantic components. Only if the hybrid search throws does
  the sidecar's BM25 answer (`<name>-sidecar-bm25`, with `fallbackFrom` in each hit's metadata), and only if that
  throws the CLI-backed `<name>-fts`. A search that finds nothing is an answer and starts no fallback. Natural-language
  queries are OR-ed term by term; FTS operators in user text are quoted away. A message found by the CLI and by the
  sidecar carries the same id (`event_id` for telegram and whatsapp), so a fallback keeps stable evidence ids.
- `channels.ids` / `channels.names` are applied in the sidecar's SQL: excluded chats are never searched,
  embedded, or stored as vectors, and vectors of a chat removed from the allowlist are pruned on the next refresh.
- `autorag refresh --method datasources` runs the CLI sync first, then embeds new or edited messages with the
  shared embedder (the same Qwen3 model as MinSync), prunes vectors of purged or deleted messages, and
  re-embeds everything if the embedding identity changes.
- Embedding is **bounded per refresh** so the first sync of a large archive does not hold the whole refresh. While
  a backlog remains the datasource reports an info diagnostic (`datasource-index-partial`: `semantic index
  catching up`), and hybrid search uses BM25 for the messages not yet embedded. Measured on 16,591 WhatsApp
  messages: about 224 messages/s; at 16,384 per refresh that archive catches up in two refreshes (about 73 s each).
- Each message's embedding input is its first 8192 characters (BM25 still indexes all of it). A message the
  embedder rejects on its own is skipped with a warning and retried only after it is edited, so it cannot stall
  the backlog behind it.
- An embedding failure (for example the embedder is unavailable: install MinSync, start `minsync serve`, or
  configure a remote `minSync.embedder`) is a
  `semantic-unavailable` **warning**; BM25 stays available and the CLI sync result is never turned into a failure.
  Two AutoRAG processes (for example `autorag refresh` and a running TUI) share the sidecar file; a writer waits up
  to 5 s for the other's lock.
- If the crawler has not created its archive yet, sidecar methods return nothing and attach on first use. If the
  crawler replaces or deletes its archive, the next search or refresh re-attaches the new file (or detaches).
- Semantic search is a linear scan over stored vectors (about 280 ms at 16.6k messages). Fine for personal
  archives; measure before pointing it at a multi-million-message archive.
- Message text is sent only to the shared embedder (local by default; a remote endpoint only when the
  operator configures one). Vectors of private chats are derived data: treat the
  sidecar file with the same care as the archive itself.

`notion` (notcrawl) keeps its page/comment FTS. Without the Discord BM25 sidecar below, `discord` is hybrid-first
through discrawl (FTS only when embeddings are unavailable).

### Discord BM25 ranking

When an AutoRAG workspace is available, `discord` ranks lexical matches in discrawl's existing FTS5 index
by BM25 relevance. An older, relevant message can therefore outrank newer mentions before the result limit.
Queries use the same term-by-term OR matching as the other sidecars. Native FTS-only hits supplement the
lexical candidate list after BM25 hits; duplicate evidence contributes only once. Hybrid retrieval combines
this one lexical ranking with native semantic results using reciprocal rank fusion (RRF).

With the sidecar enabled, the skill registers one method matching `connector.defaultMode`:
`discord-hybrid` by default, `discord-fts` for lexical ranking, or `discord-semantic` for native semantic-only
search. This prevents the outer result merger from treating overlapping native FTS, native hybrid, and BM25
results as independent votes. Datasource-scoped searches use the configured method. Explicit method-name
selection must use that registered name. With the sidecar disabled, the skill registers the single native
method for `connector.defaultMode`.

To keep only the native method, set `connector.sidecar` to `false` (or `{ "enabled": false }`):

```json
{ "datasources": { "discord-local": { "type": "discord", "connector": { "sidecar": false } } } }
```

The archive path comes from the read-only `discrawl --json status` command, using the connector's config and
the crawler's platform-specific defaults. An explicit `connector.sidecar.crawlerDbPath` or
`connector.databasePath` takes precedence for the sidecar only; the native CLI still uses its own config.
The sidecar reads the archive through SQLite's read-only attachment;
it does not change discrawl's native embeddings, invoke an additional embedder, or build another vector index.
Native embedding configuration and refresh behavior remain the same. `connector.sidecar.semantic` does not
enable a separate vector index; `connector.defaultMode` selects the retrieval mode.

Discord message IDs remain strings and match the IDs returned by the native methods, so retrieval can merge
duplicate hits. Soft-deleted and empty normalized messages are excluded from BM25, and `channels.ids` /
`channels.names` filter its results in SQL. Native candidates are checked against the configured guild and
channel allowlists before combination. Without a workspace, the native method remains available. If BM25 or
semantic retrieval fails, healthy candidates can still answer; returned hit metadata retains the original
partial-failure reasons. If a backend fails and no candidates remain, retrieval surfaces the failure instead
of reporting a successful empty search.

BM25 ranks the base `message_fts` index and, when discrawl's `search.lexical.languages` has built them, the
tokenizer-specific `message_fts_ko`, `message_fts_ja`, `message_fts_zh`, and `message_fts_ar` indexes as well.
The sidecar discovers those tables from the archive's `sqlite_master` and fuses each present table's BM25
ranking with the base ranking using reciprocal rank fusion, so an inflected Korean, Japanese, Chinese, or
Arabic message that the base tokenizer splits differently is still ranked lexically instead of only arriving as
a supplemental native candidate. Tables that are not configured are ignored. Set `sidecar: false` to retain the
native method and its language-index fusion throughout.

## Cloud drives via rclone

The `cloud-drive` datasource uses the external [`rclone`](https://rclone.org)
CLI as the provider boundary. Configure OAuth, Apple ID/session, or other
credentials only in `rclone config`; AutoRAG receives the trusted remote name,
never provider secrets.

Tier-1 is Google Drive. OneDrive and mounted/network remotes use the same
provider-neutral contract. iCloud Drive is explicitly experimental because
its rclone backend is Tier 4 and periodically requires Apple ID/password,
2FA, and reauthentication.

```json
{
  "datasources": {
    "personal-google-drive": {
      "type": "cloud-drive",
      "instanceId": "personal",
      "connector": {
        "provider": "google-drive",
        "remote": "personal-gdrive:"
      }
    },
    "company-onedrive": {
      "type": "cloud-drive",
      "instanceId": "work",
      "pollingIntervalMs": 900000,
      "connector": {
        "provider": "onedrive",
        "remote": "onedrive:Team Docs",
        "include": ["**/*.md", "**/*.pdf"],
        "exclude": ["Archive/**"],
        "maxBytesPerFile": 52428800,
        "concurrency": 4,
        "bandwidthLimit": "10M",
        "dryRun": false
      }
    }
  }
}
```

`cloud-drive` is the reusable template, not the required connection name.
Every key whose `type` is `"cloud-drive"` becomes an independent datasource:

- its key is the datasource id and skill suffix;
- `datasource-personal-google-drive` and `datasource-company-onedrive` are
  independently loadable with `load_datasource_skill`;
- source scopes are isolated under the same aliases;
- manifests, mirrors, and chunks are stored independently under
  `.autorag/datasources/<alias>/<instance>/`.

This lets one process connect multiple accounts from the same provider as well
as different providers. For example, `personal-google-drive` and
`client-google-drive` may both use Google Drive but different rclone remotes.

### Migrating from the legacy `gdrive` datasource

The former REST-backed `gdrive` datasource and its `backend: "rclone"`
compatibility mode are removed. Configure every Google Drive connection as a
named `cloud-drive` alias and authenticate the remote with `rclone config`:

```json
{
  "datasources": {
    "google-drive": {
      "type": "cloud-drive",
      "instanceId": "default",
      "connector": {
        "provider": "google-drive",
        "remote": "my-google-drive:"
      }
    }
  }
}
```

Existing `gdrive` configurations must be renamed and converted before the next
refresh; the old REST token settings and `/gdrive/**` scopes are not read.

Run the CLI datasource refresh with:

```bash
rclone config
autorag refresh --method datasources --config ./config.json
autorag search "the renewal terms in the team drive"
```

Each refresh runs `rclone lsjson --recursive --files-only --hash`, compares
the result with the workspace-local manifest at
`.autorag/datasources/<connection-alias>/<instance>/manifest.json`, then copies only
added/changed indexable files into `mirror/`. Deleted and renamed virtual paths
are removed from the completed snapshot. A no-op refresh downloads zero bodies
and does not rewrite `chunks.json`. A failed copy leaves the previous manifest
and mirror available for query-time search. `include`, `exclude`,
`maxDocuments`, `maxContentChars`, `maxBytesPerFile`, `concurrency`,
`bandwidthLimit`, and `dryRun` are trusted server configuration; model/tool
arguments cannot change them. The RSS connector takes the same trusted
`maxDocuments`/`maxItemsPerFeed`/`maxContentChars` caps, and Spotlight takes
`maxDocuments`/`maxContentChars`/`maxResultsPerQuery`/`maxBytesPerFile`.

Before searching, the agent loads the datasource skill with
`load_datasource_skill`, then calls the connection's dedicated
`search_datasource_<name>` tool using a natural-language query and, when
useful, a narrowing scope such as `/company-onedrive/work/**`. Read-only
`rclone` inspection (e.g. `rclone lsl <remote>:<path>`) may run directly
through bash per the skill's Native CLI section; credentials stay with
rclone and the agent never requests them.

## Chat channel selection

Chat/archive datasources (`kakao`, `discord`, `telegram`, `whatsapp`, and
`slack`) search all channels, rooms, chats, and DMs by default. To expose a
subset of channels, create another alias and select them in trusted configuration:

```json
{
  "datasources": {
    "all-discord": {
      "type": "discord",
      "connector": { "root": "/managed/discrawl" }
    },
    "release-channel": {
      "type": "discord",
      "connector": { "root": "/managed/discrawl" },
      "channels": {
        "ids": ["1234567890"],
        "names": ["release-engineering"]
      }
    }
  }
}
```

The backend archive remains local and shared according to its CLI
configuration; the alias is the AutoRAG visibility boundary. The restricted
alias filters returned channel/chat metadata, while the default alias remains
all-channel. The agent skill manifest states whether it is all-channel or
allowlisted, so the orchestrator can select the correct datasource before
searching.

## KakaoTalk via lazykatok

KakaoTalk support is implemented through the external [`lazykatok`](https://github.com/changeroa/lazykatok) CLI.

Rules:

- AutoRAG never reads KakaoTalk databases directly.
- Missing binary, permission, sync, or indexing failures return diagnostics instead of throwing.
- `sync` names the live adapter explicitly (`--source macos` on macOS). A bare `sync --json`
  falls back to the CLI's config file, whose default adapter is `fixture` and fails without a
  JSONL path; set `connector.source` to use another adapter.
- `kakao` is macOS-only. On Linux and Windows it is not built: a `kakao` entry in `datasources` is
  reported as unknown by the agent, `autorag setup` lists it as `skipped` ("not supported on <platform>"),
  and setup never offers it unprompted. The default store
  (`~/Library/Application Support/katok`) is therefore only ever used on macOS.
- Remote embedding egress configuration is rejected before spawning `lazykatok`.
- `connector.indexTimeoutMs` (default 30 min) bounds `sync`/`index`;
  `connector.timeoutMs` (default 60 s) bounds interactive reads
  (`doctor`, `search`, chunk lookups). A first semantic index over a large
  archive was measured at ~11 minutes, so a timed-out `index` would otherwise
  restart from scratch on every refresh.
- Lazykatok stdout/stderr and thrown error text surface as datasource diagnostics.

Example:

```ts
import { AutoRAGAgent, LazykatokSkill } from "@autorag/librarian";

const agent = new AutoRAGAgent({
  searchPaths: ["/docs"],
  datasourceSkills: [new LazykatokSkill({ instanceId: "personal" })],
});

await agent.refresh();
const hits = await agent.searchSingleDatasourceDocuments("kakao", "contract renewal", { topK: 5 });
```

## Lark / Feishu via lark-cli

Lark (larksuite.com) and Feishu (feishu.cn) are the same product on separate
hosts. v1 searches the tenant in place through the official `lark-cli`. There
is no local archive and `index()` stores nothing (`chunkCount: 0`).
Credentials stay in the CLI keychain. AutoRAG never reads the desktop
client's private databases and never accepts an app secret.

```json
{
  "datasources": {
    "lark": {
      "instanceId": "default",
      "channels": { "ids": ["oc_xxx"] }
    }
  }
}
```

`channels.ids` narrows message search with `--chat-id`. It does not hide
documents. A query `scope` under `/lark/<instance>/messages` or
`/lark/<instance>/docs` selects one surface. Install with
`npm install -g @larksuite/cli`, then `lark-cli auth login` with explicit
scopes `search:message` and `search:docs:read`, and confirm with
`lark-cli auth status`. Read a hit with
`lark-cli im +messages-mget --message-ids <id> --format json` or
`lark-cli docs +fetch --doc <token> --doc-format markdown`. Sources are
`/lark/<instance>/messages/<message_id>` and
`/lark/<instance>/docs/<token>`. Server ranking is not BM25 or vector, and
coverage of older messages is not guaranteed.

## Browser history via Hister

`hister` searches the web pages your browser visited, through a running
[Hister](https://github.com/asciimoo/hister) server. Hister is a **separate
program under AGPL-3.0-or-later**: AutoRAG ships none of its code, does not
modify it, and only sends search requests to its public HTTP API (`GET /search`,
`GET /api/config`). Hister owns the index, the browser extension, incremental
capture, and (optionally) the embeddings. AutoRAG never copies the corpus, and
the datasource manifest names Hister, its license, and its source to the agent.
See [`licenses/hister-AGPL-3.0-notice.txt`](../licenses/hister-AGPL-3.0-notice.txt).

**Web pages only.** Every query is sent with `type:web`, and any document Hister
returns with another type is dropped. Hister can also index local directories and
file snapshots; leave `indexer.directories` empty (the default) and never run
`hister import file`. If such documents exist anyway, `index()` reports how many
are excluded.

```bash
brew install hister                      # or a release binary, Docker, Nix
hister listen                            # http://127.0.0.1:4433, keep it running
# install the Hister browser extension in Chrome/Chromium (Chrome Web Store: "Hister")
hister import browser history            # optional: backfill existing history
hister search 'domain:example.com'       # must print rows before wiring it up
```

```json
{
  "datasources": {
    "browser": {
      "type": "hister",
      "connector": {
        "baseUrl": "http://127.0.0.1:4433",
        "tokenEnv": "HISTER_TOKEN",
        "semantic": false
      }
    }
  }
}
```

`baseUrl` defaults to `http://127.0.0.1:4433`. `tokenEnv` names an environment
variable holding Hister's `app.access_token`; the token is never written to the
config, and a non-loopback `http://` URL is refused when a token is configured.
`semantic: true` also registers Hister's semantic search and requires
`semantic_search.enable` plus an embedding endpoint in the **Hister** config
(the page text is sent to that endpoint, so use a loopback one for privacy); with
it disabled on the server, the semantic method fails loudly instead of returning
nothing. Plain words in a query are alternatives, sent as one `(a|b|c)` group (Hister's only
OR; the word `OR` is searched literally); Hister syntax (`"phrases"`, `domain:`, `title:`,
`-term`, `added:<7d`, `sort:date`) passes through as written.

Incremental indexing is Hister's: each page is indexed when the extension sends
it, so `autorag refresh` only probes the server (reachability, authentication,
page count). Sources are `/<id>/<instance>/pages/<domain>/<hash of the URL>`; the
URL itself is in the hit content and `metadata.url`, so use `web_fetch` to read a
page as it is now. A scope of `/<id>/<instance>/pages/<domain>/**` is pushed down
to Hister as `domain:<domain>`.

### Running it day to day

Hister must be running for the extension to capture pages and for AutoRAG to
search; AutoRAG never starts or stops it (the same rule as the MinSync daemon).
On macOS, `brew services start hister` makes it a login item and
`brew services stop hister` removes it. A page visited while Hister is down is
not sent again later, so backfill the gap with
`hister import browser history --start-date YYYY-MM-DD`. The web UI that the same
server also serves is not used by AutoRAG.

### One embedding model for AutoRAG and Hister

Hister keeps its own vector store and embeds with whatever its `config.yml`
`semantic_search` block says. If that is a different model from AutoRAG's, nothing
fails: the vectors simply stop lining up and results get worse. So the two must share
the model, and `autorag setup` does that:

```bash
autorag setup                # reports `hister.embedding.differs`, the keys that disagree; writes nothing
autorag setup --sync-hister  # writes them, restarts Hister, waits for it, re-embeds pages when needed
hister doctor                # `server.index.embeddings` must be ok
```

The source of truth is the resolved shared embedder (`minsync serve`, model
`Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions, with Qwen3's query instruction), so
Hister embeds queries exactly as AutoRAG does. `--sync-hister` edits only
`enable`, `embedding_endpoint`, `embedding_model`, `dimensions`, `query_prefix` and
`document_prefix` in Hister's own config (a `config.yml.bak.<time>` stays next to
it); the rest of the file is untouched. It restarts the service with
`brew services restart hister`, waits until the server answers, and runs
`hister reindex` only when a changed key makes stored vectors stale (the query prefix
alone does not). It is idempotent. The report says `restartRequired` or
`reindexRequired` when it could not do those steps itself. The daemon's port is chosen
when it starts, so run `--sync-hister` again after restarting it.

It refuses, and says why, for a remote shared backend (browsing text would leave the
machine and Hister's config would have to hold the key) and for MinSync's in-process
engine (Hister cannot reach it): start `minsync serve --idle-timeout 0` first. Pages
indexed before semantic search was enabled need the reindex; a page whose chunks exceed
the model's context is kept for keyword search and skipped for vectors, which Hister logs.

### Backfilling existing history safely

`hister import browser history` does not copy page text out of the browser: it
reads the History database for URLs and then downloads each URL again **without
your logged-in cookies**. A login-gated URL is therefore saved as its login page,
and tokens in a URL are sent to that site again. Before importing, put a skip list
in Hister's `rules.json` (one regular expression per entry, matched against the
full URL, and also applied to what the extension captures from now on). Cover at
least login and SSO hosts, mail, documents, drive, social and console
domains, private hosts and explicit ports, URLs whose query carries a token, key,
code or session, and `/login`, `/account`, `/settings`, `/admin` style paths.
This is a deny list: a new, unlisted login page is still captured. Use an `allow`
list instead if you want the reverse. Check the result with `hister list-urls`.

## GitHub Gists

`github-gist` indexes the authenticated account's own Gists through the GitHub
REST API: public Gists and secret Gists permitted by the token's scopes.
The local index stores descriptions and file contents (code snippets, notes,
and design memos), plus Gist IDs, visibility, filenames, URLs, and update times.
Refresh is incremental: only new or updated Gists need a full content fetch;
deleted Gists are removed from the index.

```json
{
  "datasources": {
    "github-gist": {
      "type": "github-gist",
      "instanceId": "default",
      "connector": { "tokenEnv": "GITHUB_TOKEN" }
    }
  }
}
```

Set `GITHUB_TOKEN` (or the environment variable named by `connector.tokenEnv`);
if it is unavailable, authentication falls back to the `gh` CLI login. The
token is not stored in the index or configuration. Search supports lexical
BM25 and semantic retrieval through the shared embedder; Gist content stays on
the machine for local embeddings. If embeddings are unavailable, lexical search
continues with a `semantic-unavailable` diagnostic.
Set `connector.semantic: false` (or `{ "enabled": false }`) to keep Gists
lexical-only: no embedding calls, no vector index, and no
`<name>-semantic` method.
Result sources are `/github-gist/<instance>/chunks/<chunk-id>` (opaque
datasource identities, not filesystem paths); metadata retains the Gist ID
and URL. For the live QA harness, see
[manual datasource QA](manual-qa-datasources.md).

## New datasource checklist

- Implement `DatasourceSkill`.
- Return retrieval methods whose descriptors set `datasourceId` and descriptive `tags`.
- Emit slash-hierarchical `source` values.
- Include polling/cron metadata.
- Provide source descriptions that explain the data content.
- Add capability-specific scope tests for scope-capable datasources; only those need multi-scope and user-scope narrowing tests. `#`-fragment rejection applies to every datasource source.
- Add no-throw diagnostics for missing credentials/binaries/permissions.
- Add issue labels `datasource-skill`, `integration`, and a source-specific label.
