# Datasource skills

Datasource skills let AutoRAG search external, server-configured sources while preserving the same retrieval and curation model used for local document collections.

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

## Access model

Datasource access is default-deny. Trusted server/API configuration supplies:

- `datasourceAccess.allowedTags`
- `datasourceAccess.allowedScopes`

Model-controlled tool arguments cannot grant access. The LLM-visible `search_datasource_documents` tool schema is exactly:

```ts
{ query: string; topK?: number; scope?: string }
```

`scope` is only a user-requested narrowing filter. A result must match both the trusted allow-scopes and the requested scope to survive. Datasource paths are slash-hierarchical IDs such as `/kakao/personal/chunks/abc123`; fragment-style paths with `#` are denied.

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
