---
name: autorag
description: Search, summarize, compare, and answer questions from an already configured AutoRAG librarian over local documents and authorized datasources. Use when the user asks AutoRAG to search PDFs, wikis, notes, or a knowledge base. Use autorag-setup for install, model, roots, indexing, or datasource changes.
license: MIT
---

# AutoRAG Librarian Skill

Use this skill when AutoRAG is already configured and the user asks to search,
summarize, compare, or answer questions from local PDFs, wikis, notes, research
papers, knowledge bases, or authorized datasources.

AutoRAG is the specialized librarian agent. One configured model plans the
search, calls MinSync, Jikji, datasource, and filesystem tools, reads
sources, judges evidence, and curates the final answer. There is no subagent or
separate model role.

AutoRAG reads source documents and writes indexes only under the configured
workspace `.autorag/` directory and Jikji's per-source `.jikji/` caches. Never
move, rename, edit, or delete source files.

## Preflight

Confirm a config exists at `--config`, `AUTORAG_CONFIG`,
`$AUTORAG_HOME/config.json`, or `~/.autorag/config.json`. Inspect only
non-secret `searchPaths` and `model` metadata, then run:

```bash
autorag duplicates --json
autorag status --json
autorag health --json
```

### Duplicate-file review

Use `autorag duplicates` when the user asks to find duplicate files, choose
likely latest copies, or reduce corpus/index space. The command and the
`scan_duplicate_documents` Agent tool are read-only and never delete or move
source files. Exact means dupey's canonical extracted-text hash matches;
near/contains families require human review. Exact duplicate exclusion during
refresh is enabled by default and can be disabled with
`"excludeExactDuplicates": false` in `config.json`.

`status` is model-free and path-opaque. `health` resolves the single model,
checks credential presence, and normally probes one live completion. If the
model, authentication, configuration, or indexes are unhealthy, use
`autorag-setup` rather than guessing private provider details.

MinSync and Jikji should normally be healthy. MinSync and Jikji
auto-install on first use by default. If they are missing or stale, run a full
`autorag refresh` or return to setup rather than silently degrading to
lexical-only search.

## Search

Prefer `--json --debug` when another agent will consume the result or record
feedback. `--json` alone omits `sessionId`. `--debug` adds session/diagnostics
fields and does not print filesystem paths.

```bash
autorag search "what were the key findings in the Q3 report" --top-k 5 --json --debug
```

`--json --debug` includes `answer`, numbered `results` (`number`, `title`,
`summary`, optional `source`), and `sessionId`. Use that `sessionId` for
feedback. `--json` without `--debug` is only `answer` plus `results`.

To inspect the exact persisted evidence behind numbered results, use:

```bash
autorag evidence <sessionId> --result 1 --json
```

The response includes the original source, retrieval method, stable evidence
ID, raw excerpt/content, and any available `chunkIndex`, `lineNumber`,
`retrievalResultId`, and metadata. Omit `--result` to inspect every result in
the session. Prefer this command whenever the caller wants detailed chunk text
rather than only the curated summary.

- `--scope` narrows datasource retrieval to a requested sub-path; it cannot
  grant access.
- `--tags` further narrows already-authorized datasource results and never
  grants new access.
- `--json` is required for programmatic consumption.
- `--debug` is required for `sessionId` and diagnostics in search output.
- `autorag evidence` is the detailed source/chunk inspection path.

Do not bypass the librarian with ad hoc raw search when the user requested
AutoRAG. The search loop can use Jikji, MinSync lexical/vector/hybrid retrieval, datasource retrieval, and
direct source reading as appropriate. If search fails because of model,
provider, auth, or timeout problems, diagnose with `autorag health --json`.

Record feedback so retrieval memory can learn. Numbers refer to the returned
knowledge units. Supply at least one feedback list:

```bash
autorag feedback <sessionId> --useful 1,3 --not-useful 2 --json
```

## Maintenance

```bash
autorag status --json
autorag health --json
autorag refresh --json
autorag refresh --method bm25,minsync,jikji --json
autorag watch --once --json
autorag watch
autorag refresh --force --json
autorag index rebuild --yes --json
autorag index reset --method bm25 --yes --json
autorag memory inspect --json
```

Prefer a full refresh so parsed mirrors, MinSync, Jikji, and configured
datasources stay aligned. `--method` accepts
`parsed,minsync,datasources,jikji,all`. Use it only for deliberate
narrowing. Scheduled maintenance should use non-daemon `autorag watch --once`,
typically every 15–30 minutes, with the same config used by search and no
overlapping runs.

Reset and rebuild commands remove only selected workspace `.autorag` indexes.
Never target source documents. `memory inspect` is read-only and path-opaque.

## Rules

- Use only configured and approved search paths.
- Never expose provider credentials or authentication payloads.
- Never invent provider identities or model ids.
- A Pi-usable subscription is valid; a subscription Pi cannot invoke is not.
- Preserve real source mapping and numbered feedback identifiers.
- Prefer `--json --debug` when another agent consumes search output or will
  call `autorag feedback`.
