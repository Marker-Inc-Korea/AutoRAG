---
name: autorag-lite-search
description: Retrieve documents from an indexed AutoRAG Lite corpus without a model, then persist externally curated reports, inspect evidence, and record numbered feedback. Use when the user wants model-free search, scoped or tag-narrowed retrieval, or the report/evidence/feedback workflow over local documents and authorized datasources.
license: MIT
---

# AutoRAG Lite search

Use this skill when AutoRAG Lite is already configured and indexed, and the
user wants model-free retrieval, or wants to persist an externally curated
report, inspect its evidence, or record feedback. Use `autorag-lite-setup`
when config, indexing, or refresh is missing or broken. Use the `autorag`
skill when a model-backed curated answer is requested.

AutoRAG Lite never mutates sources. Retrieval reads local source documents and
authorized datasource content; writes go only to workspace `.autorag/` indexes,
the memory file, and Jikji `.jikji/` caches.

## Preflight

Retrieval requires a completed refresh. If `autorag lite retrieve` returns
exit code 2 with an `index-not-ready` diagnostic, run `autorag lite refresh`
(or `autorag refresh`) first, then retry. Confirm freshness with:

```bash
autorag lite status --json
```

`status` is model-free and path-opaque.

## Retrieve

```bash
autorag lite retrieve "key findings in the Q3 report" --top-k 5 --json
```

The JSON envelope is:

```json
{
  "ok": true,
  "query": "key findings in the Q3 report",
  "results": [
    {
      "number": 1,
      "source": "<opaque source identifier>",
      "method": "minsync",
      "score": 0.83,
      "metadata": {},
      "content": "..."
    }
  ],
  "diagnostics": []
}
```

- `--top-k N` must be a positive integer. Invalid values exit with code 2 and
  an `{"ok": false, "error": "..."}` rejection envelope.
- Before any refresh, the envelope is `{"ok": false, "query": ..., "diagnostics":
  [{"code": "index-not-ready", "severity": "error", ...}]}` with exit code 2.
- Each result carries a provenance pair: the `source` identity and the
  `method` that produced it. Source identities are source-native: real file
  paths for local files, datasource identities for datasource results. Never
  rewrite, guess, or flatten them.
- `diagnostics` are path-opaque: `source` is a component or method label,
  never a real filesystem path. Codes include `retrieval-method-failed` and
  `minsync-unavailable`. A degraded component produces a diagnostic, not a
  failed run; check `diagnostics` before trusting an empty result set.
- `--debug` adds diagnostic detail to human output without printing real
  filesystem paths.
- Exit codes: 0 on success (results may be empty), 2 for usage, config, or
  not-ready errors, 1 for runtime errors.

Retrieval methods run in parallel and merge: parsed mirrors and MinSync
lexical/vector/hybrid methods, plus configured datasource methods. Jikji is a
local discovery/indexing preparer, not a retrieval method in the lite registry.
`--method` selection applies to refresh, not retrieve; retrieval always queries
every active retrieval method.

## Scope and tag narrowing (default-deny)

Datasource access is default-deny from trusted config
(`datasourceAccess.allowedTags`, `allowedScopes`). Retrieval flags can only
narrow that access, never widen it:

- `--scope SCOPE` narrows datasource retrieval to a requested sub-path; it
  cannot grant access to anything the config does not already allow.
- `--tags tag1,tag2` further narrows already-authorized datasource results by
  intersecting with configured allowed tags; it never grants new access.

## Persisting a curated report

An external agent curates the raw retrieval results, then persists the
structured report so evidence and feedback commands work against it:

```bash
autorag lite report "key findings in the Q3 report" --input report.json --json
cat report.json | autorag lite report "key findings in the Q3 report" --json
```

The input is the `emit_autorag_results` JSON schema: `answer`, numbered
`results` (with `number`, `title`, `summary`, `confidence` in `[0, 1]`,
optional `source`), and a `mapping` whose entries map each result number to
its `method` and `source`, plus optional `evidenceRefs` and `warnings`.
Result and mapping numbers must be one-to-one positive integers. Evidence
confidence values must be in `[0, 1]`. Invalid input exits with code 2 and
`{"ok": false, "error": "..."}`.

`source` values in a report are opaque identifiers carried from retrieval
output. Report persistence performs no filesystem reads against them; it
stores the identifiers as given. Preserve the exact source-native identities
from the retrieve envelope.

On success the JSON envelope is:

```json
{
  "ok": true,
  "sessionId": "<uuid>",
  "query": "key findings in the Q3 report",
  "answer": "...",
  "resultCount": 3
}
```

Use the returned `sessionId` for evidence inspection and feedback.

## Evidence and feedback

```bash
autorag evidence <sessionId> --result 1 --json
autorag feedback <sessionId> --useful 1,3 --not-useful 2 --json
```

- `evidence` shows the persisted source, retrieval method, stable evidence ID,
  excerpt or content, and any `chunkIndex`, `lineNumber`, and
  `retrievalResultId` behind a numbered result. Omit `--result` to inspect the
  whole session.
- `feedback` records numbered usefulness so retrieval memory can learn.
  Numbers refer to the report's result numbers. Supply at least one feedback
  list.
- Preserve real source mapping and numbered identifiers end to end: retrieve
  output feeds the report, the report's session feeds evidence and feedback.

## Rules

- Refresh before retrieval; refresh again when roots change.
- Use only configured and approved search paths and datasources.
- Never move, rename, edit, or delete source documents.
- Never expose provider credentials or authentication payloads.
- Treat report `source` values as opaque: persist them verbatim, never read
  the filesystem through them.
- Keep diagnostics path-opaque; do not reconstruct or disclose real paths
  from them.
- Prefer `--json` whenever another agent consumes the output.
