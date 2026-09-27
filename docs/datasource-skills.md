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
  },
  "datasourceAccess": {
    "allowedTags": ["clawgallery"],
    "allowedScopes": ["/screenshots/personal/**"]
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
  },
  "datasourceAccess": {
    "allowedTags": ["spotlight"],
    "allowedScopes": ["/mac-files/local/**"]
  }
}
```

## Shared embedding runtime and native ownership

The AutoRAG shared embedding runtime is a provider boundary, not a shared
archive or vector-store boundary. It computes embeddings through the
loopback-only `autorag-gateway`; each datasource keeps its native archive,
chunking, credentials, metadata, vector store, generation publication, and
source identity.

The zero-configuration boundary is narrow:

- **MinSync** uses the gateway by default. AutoRAG supplies the endpoint and
  records the profile identity; MinSync owns `.minsync`, CDC chunks, vectors,
  and reindex decisions.
- **discrawl** can receive a managed native `[search.embeddings]` section. When
  no explicit `connector.configPath` is supplied, AutoRAG writes only
  `.autorag/datasources/discrawl/config.toml`, preceded by the exact marker
  `# AutoRAG managed discrawl embeddings v1`. It writes `provider`, `model`,
  `base_url`, and `dimensions`, and checks discrawl metadata before asking the
  native CLI to rebuild. An explicit `configPath` is authoritative and is
  never rewritten.
- **mailcrawl** released its provider contract in 0.2.0: the default
  `native:Qwen/Qwen3-Embedding-0.6B` profile matches the AutoRAG gateway model
  and dimension, and `loopback-http` is available as an explicit override.
  AutoRAG does not rewrite mailcrawl's embedder; it forwards a loopback
  endpoint when one is configured and refuses a non-loopback one.
- **lazykatok** remains a pending upstream provider contract
  ([lazykatok](https://github.com/changeroa/lazykatok)). AutoRAG does not force
  the shared runtime into it before that contract is released.
- **qmd** and **clawgallery** are untouched. qmd retains its native retrieval;
  ClawGallery retains its VDR/native retrieval.
- **Lexical-only crawlers** are unchanged and receive no semantic provider
  configuration.

Embedding requests contain text and selected model/profile data only. AutoRAG
does not send archive IDs, source paths, credentials, or native store paths to
the gateway. If the runtime is unavailable, a datasource keeps its native
lexical/FTS lane where supported and reports a diagnostic; it does not silently
switch to a remote embedding service.
## Contract

A datasource skill is both:

1. an indexing hook (`index()` plus `polling()` metadata); and
2. a retrieval method factory (`retrievalMethods()`).

The methods are registered in the normal AutoRAG pipeline:

```text
RetrievalMethodRegistry
  -> ParallelRetriever
  -> DatasourceResultFilter
  -> ResultMerger
  -> memory / curation
```

A skill must also provide `describeSources()` entries so the librarian prompt can explain what data exists.

## mailcrawl

The `mailcrawl` datasource delegates local email synchronization and search to
the external `mailcrawl` CLI. Install `@nomadamas/mailcrawl@0.2.0` or newer
(Node.js 24+) and configure Himalaya separately; AutoRAG never opens
`archive.sqlite` directly. 0.1.3 and earlier fail a repeated `index` after a
no-op sync (`text array must be non-empty`). By default, mailcrawl uses its
native archive. Set `connector.dataDir` only when the operator explicitly
wants a different mailcrawl data directory.

0.2.0 replaced the semantic store and embedder contract:

- Vectors live in a LanceDB table (`<data-dir>/semantic.lance`) with the
  embedder identity in `semantic.identity.json`. A mismatch rebuilds the
  table, and a search against a different embedder fails loudly instead of
  returning silently wrong neighbours.
- The default embedder is the in-process native `Qwen/Qwen3-Embedding-0.6B`
  (1024 dimensions, the model identity AutoRAG's gateway profile pins), so a
  cold cache downloads ONNX weights on the first `index`. AutoRAG gives
  `sync`/`index` a 30-minute budget (`connector.indexTimeoutMs`) while search
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
  },
  "datasourceAccess": {
    "allowedTags": ["mailcrawl", "email"],
    "allowedScopes": ["/mailcrawl/personal/**"]
  }
}
```

Refresh runs `mailcrawl sync --json` followed by `mailcrawl index --json`.
Retrieval exposes independent BM25, semantic, and hybrid methods and maps
results to opaque `/mailcrawl/<instance>/chunks/<chunk-id>` sources. Use
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
is normally used without changing its access policy.

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

Descriptions are user-supplied, are not inferred automatically, and cannot
grant access or widen `datasourceAccess.allowedScopes`.

The key is the independent datasource ID and becomes an independently
loadable `datasource-<alias>` skill. Its source scope, diagnostics, method
names, local storage/cache namespace, and access policy are rewritten under
the alias. This supports multiple connections of the same provider as well as
different providers in one agent.

## Access model

Datasource access is default-deny. Trusted server/API configuration supplies:

- `datasourceAccess.allowedTags`
- `datasourceAccess.allowedScopes`

Model-controlled tool arguments cannot grant access. Every authorized connection gets its own generated `search_datasource_<id>` tool, and each one's schema is exactly:

```ts
{ query: string; topK?: number; scope?: string }
```

There is no datasource fan-out tool: a question that spans every datasource (or everything else) uses `search_all_documents`, which already registers every authorized connection's retrieval methods alongside the local ones.

`scope` is only a user-requested narrowing filter for datasource methods that advertise the `scoped` capability. A result from such a method must match both the trusted allow-scopes and the requested scope to survive. Datasources without that capability (for example, lazykatok's chat-identity results) are authorized at the datasource/tag level and own any narrower filtering themselves.

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
  },
  "datasourceAccess": {
    "allowedTags": ["slack", "chat"],
    "allowedScopes": ["/slack-local/local/**"]
  }
}
```

`workspace` is the Slack team id to scope reads to and it is effectively
required. `slacrawl`'s `search` and `messages` read paths return nothing unless
a workspace is named explicitly, even when the archive and its FTS index hold
matching rows — an unscoped search looks exactly like an empty archive, with no
diagnostic. `syncSource` is likewise required: `slacrawl sync` without
`--source` fails and surfaces as `datasource-index-failed`.

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
  },
  "datasourceAccess": {
    "allowedTags": ["cloud-drive"],
    "allowedScopes": [
      "/personal-google-drive/personal/**",
      "/company-onedrive/work/**"
    ]
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
`maxBytesPerFile`, `concurrency`, `bandwidthLimit`, and `dryRun` are trusted
server configuration; model/tool arguments cannot change them.

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
restricted datasource, create another alias and use trusted configuration:

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
  },
  "datasourceAccess": {
    "allowedTags": ["discord"],
    "allowedScopes": ["/all-discord/**", "/release-channel/**"]
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
- Remote embedding egress configuration is rejected before spawning `lazykatok`.
- Lazykatok stdout/stderr and thrown error text surface as datasource diagnostics.

Example:

```ts
import { AutoRAGAgent, LazykatokSkill } from "@autorag/librarian";

const agent = new AutoRAGAgent({
  searchPaths: ["/docs"],
  datasourceSkills: [new LazykatokSkill({ instanceId: "personal" })],
  datasourceAccess: {
    allowedTags: ["kakaotalk"],
    allowedScopes: ["/kakao/personal/**"],
  },
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
  },
  "datasourceAccess": {
    "allowedTags": ["lark:chat", "lark:docs"],
    "allowedScopes": ["/lark/default/**"]
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

## New datasource checklist

- Implement `DatasourceSkill`.
- Return retrieval methods whose descriptors set `datasourceId` and authorization `tags`.
- Emit slash-hierarchical `source` values.
- Include polling/cron metadata.
- Provide source descriptions that explain the data content.
- Add default-deny and capability-specific scope tests; only scope-capable datasources need multi-scope and user-scope intersection tests.
- Add no-throw diagnostics for missing credentials/binaries/permissions.
- Add issue labels `datasource-skill`, `integration`, and a source-specific label.
