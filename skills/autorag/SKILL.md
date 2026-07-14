---
name: autorag
description: Set up and use AutoRAG (the librarian agent CLI) for local document collections. On first setup, detect the current agent's usable model provider or subscription-backed runtime, select appropriate orchestrator and explorer models, configure approved search paths, and build the indexes.
---

# AutoRAG Librarian Skill

Use this skill when the user asks you to search, summarize, compare, or answer
questions from their local documents (PDFs, wikis, notes, research papers,
knowledge bases) with AutoRAG. AutoRAG searches, reads, and curates numbered
knowledge units instead of dumping raw grep output.

AutoRAG is invoked through the `autorag` CLI (bin name `autorag`). It is
non-destructive: it only reads source files and writes indexes under `.autorag/`
(and, when jikji is enabled, jikji's own `.jikji/` caches). Never move, rename,
or delete the user's source files.

## First-time setup

Run this setup workflow when `~/.autorag/config.json` is absent, incomplete, or
the user asks to configure AutoRAG. Setup has two independent decisions: which
models AutoRAG can actually authenticate, and which folders it may index.

### Detect and configure the agent models

Do not begin by asking the user to manually name a provider. First inspect the
available non-secret runtime metadata and configure the best usable pair.

1. **Preserve explicit choices.** Prefer model/provider choices stated by the
   user or already present in `~/.autorag/config.json`. Do not replace a working
   explicit configuration merely because another provider is detectable.

2. **Detect the current agent runtime.** Inspect the current agent's exposed
   provider/model metadata, then check compatible local metadata such as
   `~/.codex/config.toml` and `~/.autorag/pi-agent/models.json`. You may check
   whether provider credential environment variables or Pi auth entries exist,
   but never print, copy, persist, or compare credential values. Never read a
   credential merely to show it to the user.

3. **Distinguish subscriptions from API access.** A ChatGPT, Claude, Gemini, or
   other consumer subscription is not automatically an API entitlement. Treat
   a subscription-backed login as usable only when the current agent runtime
   can delegate that authenticated provider to AutoRAG or compatible auth is
   already available in `~/.autorag/pi-agent`. Otherwise prefer a detected,
   Responses-compatible API provider with an available credential. Do not claim
   that a provider is usable based only on an installed CLI or config filename.

4. **Select models by role from models the provider actually exposes.** Use the
   strongest reliable reasoning/high-context model for `agents.orchestrator`.
   Use a faster and cheaper high-recall model with sufficient context for
   `agents.explorer`. When only one usable model exists, configure it for both
   roles rather than inventing a model ID. For the standard `myproxy` catalog,
   prefer `gpt-5.6-sol` for the orchestrator and `gpt-5.6-luna` for explorers.
   Never hard-code those IDs for a provider that does not advertise them.

5. **Resolve ambiguity without leaking secrets.** If several authenticated
   providers are equally suitable, prefer the provider already used by the
   current agent, then the existing AutoRAG/Pi provider. Ask one concise choice
   only when no evidence distinguishes them. If no compatible authenticated
   provider exists, report the exact missing provider/auth requirement instead
   of writing a configuration that cannot run.

6. **Write both role settings.** Configure the selected pair with role-specific
   flags; provider and ID must always be supplied together:

   ```bash
   autorag init \
     --search-paths "/path/to/docs,/path/to/notes" \
     --orchestrator-model-provider PROVIDER \
     --orchestrator-model-id ORCHESTRATOR_MODEL \
     --explorer-model-provider PROVIDER \
     --explorer-model-id EXPLORER_MODEL
   ```

   This writes `~/.autorag/config.json`. For a different config location, use
   `--config PATH`. Use `--force` only when intentionally replacing an existing
   config. Per-run overrides are also available through the same role-specific
   flags or `AUTORAG_ORCHESTRATOR_MODEL_PROVIDER`,
   `AUTORAG_ORCHESTRATOR_MODEL_ID`, `AUTORAG_EXPLORER_MODEL_PROVIDER`, and
   `AUTORAG_EXPLORER_MODEL_ID`.

### Discover and approve search folders

1. **Infer the target area.** If the user named a directory, use it. Otherwise
   use the current project or a document directory already established by the
   conversation. Ask for a root only when no safe target can be inferred. Do
   not scan the whole filesystem or home directory without explicit permission.

2. **Discover candidate folders.** Explore the target area and find folders that
   contain many document-like files. Prefer document extensions —
   `pdf, md, markdown, txt, rtf, docx, doc, xlsx, pptx, hwp, hwpx, epub, eml,
   csv, html` — and count files per folder. Skip noise: `node_modules`, `.git`,
   `dist`, `build`, `target`, `.cache`, `.autorag`, `.jikji`, and other
   generated/vendor directories.

3. **Recommend the densest relevant folders.** Present path and approximate file
   count. Do not silently index a large or sensitive tree. Use folders already
   explicitly approved by the user without asking again.

4. **Initialize and verify.** Run `autorag init` once with the approved paths and
   detected role models, inspect the resulting non-secret configuration, then
   build the indexes:

   ```bash
   autorag refresh
   autorag status
   ```

   `refresh` parses the configured folders into `.autorag/parsed`, builds the
   BM25 and MinSync indexes, indexes configured datasource skills, and runs
   `jikji prepare` per folder when Jikji is enabled. Re-run it after documents
   change.

## Searching

Once folders are configured and indexed:

```bash
autorag search "what were the key findings in the Q3 report" --top-k 5
```

`autorag search` returns curated, numbered knowledge units grounded in the
sources. The persistent role models are configured under `agents.orchestrator`
and `agents.explorer` in `~/.autorag/config.json`; the role-specific CLI flags
or environment variables may override them for one run. Use `--scope` to narrow
to a sub-path, `--json` for machine-readable output, and `--debug` to reveal
diagnostics.

Record feedback so AutoRAG learns which results were useful:

```bash
autorag feedback <sessionId> --useful 1,3 --not-useful 2
```

## Maintenance commands

```bash
autorag status            # corpus freshness + index health
autorag refresh --force   # full re-parse and re-index
autorag index rebuild --yes   # reset then rebuild all indexes
autorag memory inspect    # inspect the retrieval memory snapshot
```

## Rules

- Non-destructive: only read source files; never move/rename/delete them.
- Do not index outside the folders the user approved; require or confirm an
  explicit root before scanning.
- Prefer recommending a few file-dense, relevant folders over indexing large or
  sensitive trees; ask before indexing anything large or private.
- Detect providers from non-secret metadata and authentication availability;
  never expose, migrate, or persist credential values.
- Never equate a consumer subscription with API access unless the active runtime
  demonstrably supports reusing that authenticated subscription for AutoRAG.
