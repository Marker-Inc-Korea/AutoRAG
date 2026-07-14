---
name: autorag-setup
description: Configure AutoRAG for first use or repair its setup. Detect the current agent's usable subscription-backed runtime or API provider without exposing private provider identities, select orchestrator and explorer models, approve document folders, initialize configuration, and build indexes.
---

# AutoRAG Setup Skill

Use this skill for first-time setup, missing or broken model configuration,
provider/authentication discovery, changing indexed folders, or rebuilding the
initial indexes. After setup succeeds, use the separate `autorag` skill for
normal searches and feedback.

## Safety boundaries

- Inspect only non-secret provider/model metadata and credential availability.
- Never print, copy, migrate, compare, or persist credential values.
- Never expose private provider aliases or internal model catalogs in generated
  configuration, logs, or user-facing explanations.
- Do not scan the whole filesystem or home directory without explicit approval.
- Never move, rename, edit, or delete source documents.

## Detect an authenticated model runtime

Do not begin by asking the user to manually name a provider. Determine the best
usable configuration from available evidence.

1. Preserve explicit user choices and a working `~/.autorag/config.json`.
2. Inspect the current agent's exposed provider/model capabilities.
3. Inspect compatible non-secret local metadata, including configured model
   registries and `~/.autorag/pi-agent/models.json`. Provider-specific local
   configuration may be used only to determine endpoint compatibility, model
   IDs, credential environment-variable names, and whether credentials exist.
4. Check Pi authentication entries by provider identity without reading or
   showing their secret payloads.
5. Treat ChatGPT, Claude, Gemini, and other consumer subscriptions as usable
   only when the active runtime can demonstrably delegate that authenticated
   session to AutoRAG or compatible authentication already exists in AutoRAG's
   Pi state. A subscription is not automatically an API entitlement.
6. Do not infer usability from an installed CLI, a config filename, or an
   environment-variable name alone. Authentication and protocol compatibility
   must both be established.

If no compatible authenticated runtime exists, report the exact missing public
provider/authentication requirement. Do not write a configuration that cannot
run. Ask one concise question only when multiple equally suitable public
providers remain and runtime evidence cannot choose between them.

## Select role models

Select only models actually advertised by the authenticated runtime:

- `agents.orchestrator`: strongest reliable reasoning and high-context model.
- `agents.explorer`: faster, cheaper high-recall model with sufficient context.
- If only one usable model exists, configure it for both roles.
- Preserve an existing working explicit pair.
- Never invent model IDs or write a private provider alias into distributed or
  user-facing configuration.

Provider and model ID must always be supplied together for each configured role.
If role models are intentionally omitted, leave them unset so search-time
resolution can still pick up an authenticated local runtime when available.

## Discover and approve document folders

1. Use a directory explicitly named by the user. Otherwise infer a safe root
   from the current project or conversation; ask only when no safe root exists.
2. Prefer folders dense with formats AutoRAG's parsers can index into
   `.autorag/parsed` for BM25/MinSync:
   - text notes: `md`, `markdown`, `txt`, `text`
   - documents: `pdf`, `docx`, `pptx`, `xlsx`, `hwpx`, `eml`
   - optional OCR images only when enabled: `jpg`, `jpeg`, `png`, `bmp`, `tiff`
3. Do not present legacy binary shells that currently fail pure-JS parsing as
   fully supported: bare `doc`, `xls`, and legacy `hwp` (use `docx` / `xlsx` /
   `hwpx` instead). Explorers can still open any readable text under approved
   paths with `read`/`grep`/`find`/`ls`, but those files will not contribute to
   BM25/MinSync indexes.
4. Skip generated/vendor directories including `node_modules`, `.git`, `dist`,
   `build`, `target`, `.cache`, `.autorag`, and `.jikji`.
5. Recommend a small set of dense, relevant folders with approximate counts.
   Do not silently index large or sensitive trees. Reuse paths already approved
   by the user without asking again.

## Initialize

Write the approved paths and selected role models:

```bash
autorag init \
  --search-paths "/path/to/docs,/path/to/notes" \
  --orchestrator-model-provider PROVIDER \
  --orchestrator-model-id ORCHESTRATOR_MODEL \
  --explorer-model-provider PROVIDER \
  --explorer-model-id EXPLORER_MODEL
```

This writes `~/.autorag/config.json` by default. Use `--config PATH` (or
`AUTORAG_CONFIG`) for an explicit location. Optional:

- `--workspace DIR` for the workspace that owns `.autorag` indexes
- `--memory-path FILE` for retrieval-memory storage
- `--force` only when intentionally replacing an existing config

If role models are intentionally omitted, `autorag init` must leave `agents`
unset; it must never inject a private provider default. Legacy cwd
`autorag.config.json` is a migration source only and is never deleted by init.

Per-run overrides (flags and env take precedence over the file) are available
via the role-specific flags or:

- `AUTORAG_ORCHESTRATOR_MODEL_PROVIDER` / `AUTORAG_ORCHESTRATOR_MODEL_ID`
- `AUTORAG_EXPLORER_MODEL_PROVIDER` / `AUTORAG_EXPLORER_MODEL_ID`
- legacy single-model aliases: `AUTORAG_MODEL_PROVIDER` / `AUTORAG_MODEL_ID`
- `AUTORAG_SEARCH_PATHS`, `AUTORAG_WORKSPACE`, `AUTORAG_MEMORY_PATH`

## Build and verify

```bash
autorag refresh
autorag status
```

`refresh` parses approved sources and resyncs active indexes (parsed mirror,
BM25, MinSync, datasources, optional Jikji prepare). `status` is model-free and
path-opaque: inspect its freshness/index health only; do not expect absolute
paths. Separately confirm both role models can be resolved from the
authenticated runtime or written config without displaying credentials or
private provider details. Do not claim setup succeeded when authentication,
indexes, or role-model resolution remain unverified.

After setup succeeds, hand normal queries to the `autorag` skill (`autorag
search`, then `autorag feedback` with the returned `sessionId`).
