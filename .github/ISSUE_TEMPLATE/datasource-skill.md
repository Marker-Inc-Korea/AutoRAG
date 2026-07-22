---
name: Datasource skill integration
description: Track a new AutoRAG datasource skill
labels: ["datasource-skill", "integration"]
---

## Datasource

Name:
External API/CLI/SDK:
Expected owner/maintainer:

## Data description

Describe what the datasource contains in human terms. This description may appear in the AutoRAG prompt, so do not include secrets, account IDs, phone numbers, or real local paths.

## Hierarchy

Examples:

- workspace -> channel
- account -> folder
- chat export -> room
- repository -> issue/PR

Opaque source root pattern:

```text
/<datasource>/<instance-or-scope>/...
```

## Indexing behavior

- [ ] Manual refresh supported
- [ ] Polling metadata (`mode: "poll"`, `intervalMs`) defined
- [ ] Cron metadata (`mode: "cron"`, `cronExpr`) considered if routine schedule is needed
- [ ] Missing credentials/binary/API permission returns diagnostics, not throws

## Search behavior

- [ ] Retrieval methods plug into `RetrievalMethodRegistry`
- [ ] Results use opaque slash-hierarchical source IDs
- [ ] Source descriptions explain content without leaking storage paths
- [ ] Public `SearchDocumentsResponse` stays path/PII opaque

## Access boundaries

- [ ] Default deny without trusted `allowedTags`
- [ ] Trusted `allowedScopes` enforced before merge
- [ ] User/tool `scope` only narrows trusted scopes
- [ ] LLM tool arguments cannot grant tags/scopes
- [ ] Permission tags listed:

## Tests

- [ ] Default deny
- [ ] Multi-scope filtering
- [ ] User-scope intersection
- [ ] Hash-fragment rejection
- [ ] No-throw diagnostics
- [ ] Path/PII leak regression
- [ ] Datasource-specific auth/rate-limit failures

## Manual QA

Describe the smallest safe manual QA scenario and any credentials or local-app permissions required.
