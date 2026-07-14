---
name: autorag
description: Use an already configured AutoRAG librarian agent to search, summarize, compare, and answer questions from local document collections. Use autorag-setup instead for provider/model discovery, first-time configuration, or selecting folders to index.
---

# AutoRAG Librarian Skill

Use this skill when AutoRAG is already configured and the user asks to search,
summarize, compare, or answer questions from local PDFs, wikis, notes, research
papers, or knowledge bases. For first-time configuration, missing model/provider
settings, subscription or API-provider detection, or changing indexed folders,
use the `autorag-setup` skill instead.

AutoRAG is invoked through the `autorag` CLI. It is non-destructive: it reads
source files and writes indexes under the configured workspace's `.autorag/`
directory and, when enabled, Jikji's `.jikji/` caches. Never move, rename, or
delete source files.

## Preflight

Confirm that `~/.autorag/config.json` exists and has usable search paths. Run:

```bash
autorag status
```

If configuration, authentication, role models, or indexes are missing, stop this
workflow and use `autorag-setup`; do not guess private providers or model IDs.

## Searching

```bash
autorag search "what were the key findings in the Q3 report" --top-k 5
```

AutoRAG returns curated, numbered knowledge units grounded in sources. Use
`--scope` to narrow to a configured virtual sub-path, `--json` for structured
output, and `--debug` only when diagnostics are needed. Do not bypass AutoRAG
with ad hoc raw search when the user explicitly requested the librarian agent.

Record feedback so retrieval memory learns which results were useful:

```bash
autorag feedback <sessionId> --useful 1,3 --not-useful 2
```

## Maintenance

```bash
autorag status
autorag refresh
autorag refresh --force
autorag index rebuild --yes
autorag memory inspect
```

Use `refresh` after source documents change. Use destructive index-reset commands
only for generated AutoRAG indexes, never for source documents.

## Rules

- Use only configured and approved search paths.
- Never expose provider credentials or authentication payloads.
- Never invent or reveal private provider names or model IDs.
- Never treat a consumer subscription as API access without runtime evidence.
- Preserve real source mapping and numbered feedback identifiers.