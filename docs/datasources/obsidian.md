---
title: "Obsidian"
description: "Index an Obsidian markdown vault with MinSync in its own workspace and search notes lexically or semantically."
---

# Obsidian

The `obsidian` datasource indexes a markdown vault with MinSync — the same engine and `minSync.*` binary and embedder AutoRAG uses for local files — in a workspace of its own, so vault notes never mix with the local-file index. Use it for questions about personal notes and knowledge-base content.

## Prerequisites

- A vault directory containing markdown notes.
- A workspace: AutoRAG stages the notes under `<workspace>/.autorag/datasources/obsidian/<instance>/` and runs a MinSync index there. Without a workspace the vault cannot be indexed or searched.
- MinSync, resolved exactly as for the local-file index (`minSync.binaryPath`, then `PATH`, then the workspace cache, then the pinned auto-install). See [Embedding backends](../indexing/embedding-backends.md).

## Connect

Add a `datasources` entry to `~/.autorag/config.json`:

```json
{
  "datasources": {
    "obsidian": {
      "connector": { "vaultPath": "/path/to/vault" }
    }
  }
}
```

Configuring several vaults under separate identities is a connection alias:

```json
{
  "datasources": {
    "notes-work": { "type": "obsidian", "connector": { "vaultPath": "/path/to/work-vault" } }
  }
}
```

Then index it:

```bash
autorag refresh --method datasources
```

## What is indexed

- Marks of notes only: files with a `.md` or `.markdown` extension.
- Dot-prefixed folders and files (`.obsidian`, `.trash`, `.git`), `node_modules`, and symlinks are skipped.
- Notes are staged under `<workspace>/.autorag/datasources/obsidian/<instance>/minsync/files`; MinSync embeds and indexes them incrementally, so only new, changed, and deleted notes are touched.
- A source is the note's vault-relative path: `/obsidian/<instance>/<folder>/<note>.md`. `#`, `%`, `*`, and `\` in a note or folder name are escaped as `%23`, `%25`, `%2A`, and `%5C`. Paths are NFC-normalized.
- Result metadata carries `file` (vault-relative), `title` (the file stem), `path` (the real path on disk), and `sourceMtimeMs` (the staged note's last-modified time).

## Search behavior

- Two methods: `obsidian-sparse` (SPLADE) and, unless the index is sparse-only, `obsidian-semantic` (vector), both through MinSync and the shared embedder.
- A folder is an ordinary `scope`: `scope=/obsidian/default/Projects` matches that folder and everything below it (see [Scope narrowing](search-semantics.md#scope-narrowing)). A glob needs an explicit trailing `/**`, for example `/obsidian/default/**/Daily/**`.
- Time windows use the note's `sourceMtimeMs`, the last-modified time recorded when it was staged (see [Time windows](search-semantics.md#time-windows)).
- Default poll cadence is 15 minutes while auto-refresh runs.
- The index is AutoRAG-owned. Reset it by deleting `<workspace>/.autorag/datasources/obsidian/<instance>` and running:

```bash
autorag refresh --method datasources
```

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| `no vaultPath is configured for this Obsidian datasource` | `datasource-unavailable` | Set `connector.vaultPath`. |
| `no workspace is configured to hold the vault index` | `datasource-unavailable` | Run AutoRAG with a workspace. |
| The vault has no markdown notes | info `datasource-empty` | Nothing to index; the previous index is kept rather than wiped. |
| The vault is not indexed yet when searching | thrown error naming `autorag refresh --method datasources` | Run the refresh before searching. |
| Unreadable folders, colliding note paths, or notes MinSync skipped | warning `datasource-index-partial` | Their notes keep the previous index; the next refresh retries them. |
