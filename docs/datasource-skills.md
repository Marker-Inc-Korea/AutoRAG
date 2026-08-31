# Datasource skills

For CLI-backed datasource configuration, ownership, process boundaries, and
contributor requirements, see the normative
[managed CLI configuration guide](managed-cli-configuration.md).

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

Operators can add, test, enable, and remove connections from `autorag ui` instead of editing `config.json`. The UI is a local loopback control plane: it uses the same factory, never grants access from model tool arguments, and never writes token values into config.

The UI can also be deployed behind an explicitly configured reverse proxy. Set
`ui.allowRemote` to `true`, keep `ui.tokenEnv` in the process environment, and
list the exact browser origins in `ui.corsOrigins`; wildcard CORS is not
supported because the UI uses credentialed requests. `ui.publicOrigin` is the
URL printed for operators and used when opening the browser. Without
`allowRemote`, non-loopback binds are rejected.

## Universal connection aliases

Every datasource entry can use a reusable template with a connection alias:

```json
{
  "datasources": {
    "personal-gmail": {
      "type": "gmail",
      "connector": { "tokenEnv": "PERSONAL_GMAIL_TOKEN" }
    },
    "company-slack": {
      "type": "slack",
      "connector": { "configPath": "/secure/company-slack.toml" }
    },
    "family-kakao": {
      "type": "kakao",
      "channels": { "names": ["가족방"] },
      "connector": { "binaryPath": "katok" }
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

Model-controlled tool arguments cannot grant access. The LLM-visible `search_datasource_documents` tool schema is exactly:

```ts
{ query: string; topK?: number; scope?: string }
```

`scope` is only a user-requested narrowing filter. A result must match both the trusted allow-scopes and the requested scope to survive. Datasource paths are slash-hierarchical IDs such as `/kakao/personal/chunks/abc123`; fragment-style paths with `#` are denied.

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

Every instance maps to a datasource root like `/kakao/personal` or `/slack/workspace/channel`.

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

Initialize and refresh the local mirror with:

```bash
slacrawl init -db ~/.slacrawl/slacrawl.db -workspace local
slacrawl sync --source wiretap
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
`load_datasource_skill`, then calls `search_datasource_documents` using a
natural-language query and, when useful, a narrowing scope such as
`/company-onedrive/work/**`. It must not invoke `rclone` itself or request
credentials.

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

## KakaoTalk via katok

KakaoTalk support is implemented through the external [`katok`](https://github.com/NomaDamas/katok) CLI.

Rules:

- AutoRAG never reads KakaoTalk databases directly.
- Missing binary, permission, sync, or indexing failures return diagnostics instead of throwing.
- Remote embedding egress configuration is rejected before spawning `katok`.
- Katok stdout/stderr and thrown error text surface as datasource diagnostics.

Example:

```ts
import { AutoRAGAgent, KatokSkill } from "@autorag/librarian";

const agent = new AutoRAGAgent({
  searchPaths: ["/docs"],
  datasourceSkills: [new KatokSkill({ instanceId: "personal" })],
  datasourceAccess: {
    allowedTags: ["kakaotalk"],
    allowedScopes: ["/kakao/personal/**"],
  },
});

await agent.refresh();
const hits = await agent.searchDatasourceDocuments("contract renewal", { topK: 5 });
```

## New datasource checklist

- Implement `DatasourceSkill`.
- Return retrieval methods whose descriptors set `datasourceId` and authorization `tags`.
- Emit slash-hierarchical `source` values.
- Include polling/cron metadata.
- Provide source descriptions that explain the data content.
- Add default-deny, multi-scope, and user-scope intersection tests.
- Add no-throw diagnostics for missing credentials/binaries/permissions.
- Add issue labels `datasource-skill`, `integration`, and a source-specific label.
