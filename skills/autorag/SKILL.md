---
name: autorag
description: Use AutoRAG (the librarian agent CLI) to search and curate answers from a local document collection. On first setup, discover file-dense folders, recommend them to the user, configure them as search paths, and build the indexes.
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

## First-time setup: discover and recommend folders

If there is no `autorag.config.json` in the working directory (or the user has
not told you which folders to index), do this before searching:

1. **Ask or infer the target area.** If the user named a directory, use it. If
   not, ask for a root to look under (default to the current project or a
   document directory the user mentions). Do not scan the whole filesystem or
   the home directory without explicit permission.

2. **Discover candidate folders.** Explore the target area and find folders that
   contain many document-like files. Prefer document extensions —
   `pdf, md, markdown, txt, rtf, docx, doc, xlsx, pptx, hwp, hwpx, epub, eml,
   csv, html` — and count files per folder. Skip noise: `node_modules`, `.git`,
   `dist`, `build`, `target`, `.cache`, `.autorag`, `.jikji`, and other
   generated/vendor directories.

3. **Recommend the densest folders.** Present the top folders by document count
   (path + approximate file count) and let the user confirm or edit the
   selection. Recommend the folders with the most relevant documents; do not
   silently index everything.

4. **Configure the chosen folders as search paths:**

   ```bash
   autorag init --search-paths "/path/to/docs,/path/to/notes"
   ```

   This writes `autorag.config.json`. You can also set `--workspace`,
   `--memory-path`, `--model-provider`, and `--model-id`. Multiple folders are
   comma-separated. To change the selection later, edit `searchPaths` in
   `autorag.config.json` or re-run `autorag init --force`.

5. **Build the indexes:**

   ```bash
   autorag refresh
   ```

   `refresh` parses the configured folders into `.autorag/parsed`, builds the
   BM25 and MinSync indexes, indexes any configured datasource skills, and —
   when jikji is enabled in the config — runs `jikji prepare` per folder to
   build each folder's local file map/caches. Re-run `autorag refresh` after the
   documents change (or use it periodically).

## Searching

Once folders are configured and indexed:

```bash
autorag search "what were the key findings in the Q3 report" --top-k 5
```

`autorag search` returns curated, numbered knowledge units grounded in the
sources (requires a configured model via `--model-provider`/`--model-id` or the
config `model` key). Use `--scope` to narrow to a sub-path, `--json` for
machine-readable output, and `--debug` to reveal diagnostics.

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
