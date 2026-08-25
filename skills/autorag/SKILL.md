---
name: autorag
description: Use an already configured AutoRAG librarian agent to search, summarize, compare, and answer questions from local document collections. Use autorag-setup for configuration or indexing changes.
---

# AutoRAG Librarian Skill

Use this skill when AutoRAG is already configured and the user asks to search,
summarize, compare, or answer questions from local PDFs, wikis, notes, research
papers, or knowledge bases.

AutoRAG is the specialized librarian agent. One configured model plans the
search, calls BM25, MinSync, Jikji, datasource, and filesystem tools, reads
sources, judges evidence, and curates the final answer. There is no subagent or
separate model role.

AutoRAG reads source documents and writes indexes only under the configured
workspace `.autorag/` directory and Jikji's per-source `.jikji/` caches. Never
move, rename, edit, or delete source files.

## Preflight

Confirm `~/.autorag/config.json`, an explicit `--config`, or `AUTORAG_CONFIG`
exists. Inspect only non-secret `searchPaths` and `model` metadata, then run:

```bash
autorag duplicates              # read-only duplicate-family review
autorag duplicates --json       # machine-readable cleanup planning input
autorag status
autorag health
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

BM25, MinSync, and Jikji should normally be healthy. If they are missing or
stale, run a full `autorag refresh` or return to setup rather than silently
degrading to lexical-only search.

## Search

```bash
autorag search "what were the key findings in the Q3 report" --top-k 5 --json
```

The response contains a `sessionId`, numbered curated results, source mapping,
and an answer grounded in those results.

- `--scope` narrows to a configured virtual sub-path.
- `--tags` narrows already-authorized datasource results and never grants new
  access.
- `--json` is preferred for programmatic consumption.
- `--debug` is for diagnostics only.

Do not bypass the librarian with ad hoc raw search when the user requested
AutoRAG. The search loop can use Jikji, BM25, MinSync, datasource retrieval, and
direct source reading as appropriate. If search fails because of model,
provider, auth, or timeout problems, diagnose with `autorag health`.

Record feedback so retrieval memory can learn:

```bash
autorag feedback <sessionId> --useful 1,3 --not-useful 2
```

Supply at least one feedback list. Numbers refer to the returned knowledge
units.

## Maintenance

```bash
autorag status
autorag health
autorag refresh
autorag refresh --method bm25,minsync,jikji
autorag watch --once
autorag watch
autorag refresh --force
autorag index rebuild --yes
autorag index reset --method bm25 --yes
autorag memory inspect
```

Prefer a full refresh so parsed mirrors, BM25, MinSync, Jikji, and configured
datasources stay aligned. Use `--method` only for deliberate narrowing.
Scheduled maintenance should use non-daemon `autorag watch --once`, typically
every 15–30 minutes, with the same config used by search and no overlapping
runs.

Reset and rebuild commands remove only selected workspace `.autorag` indexes.
Never target source documents. `memory inspect` is read-only and path-opaque.

## Rules

- Use only configured and approved search paths.
- Never expose provider credentials or authentication payloads.
- Never invent or reveal private provider names or model ids.
- A Pi-usable subscription is valid; a subscription Pi cannot invoke is not.
- Preserve real source mapping and numbered feedback identifiers.
- Prefer `--json` when another agent consumes the response.
