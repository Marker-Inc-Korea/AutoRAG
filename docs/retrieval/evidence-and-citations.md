---
title: "Evidence and citations"
description: "Evidence ids, the ledger that resolves them, numbered results, the mapping and report schema, and how file locations are measured."
---

# Evidence and citations

This page explains how AutoRAG tracks retrieved evidence, how the model's inline citations become numbered, source-backed results, and how to inspect the evidence behind a search — including how a chunk's line, page, and section are measured.

## Evidence ids and the ledger

Every retrieval tool registers what it returns in the conversation's `EvidenceLedger` (`https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/agent/evidence-ledger.ts`) and prints an `[eN]` id (`e1`, `e2`, …) beside each result. The model cites those ids; the harness resolves each one back to the exact source, method, content, and chunk metadata it recorded. The model never retypes a path or a chunk, so it cannot misquote or invent one.

- The ledger keys an entry on `source` + a digest of the full normalized chunk. Retrieving the same chunk again returns the **same** id and records the extra method in `retrieverMix`.
- Ids count up for as long as the process lives and are **never reissued**, so an id the model saw in any earlier turn keeps naming the chunk it named.
- Stored content is capped at 2,000 characters per entry; the id is keyed on the full chunk, not the stored prefix, so two chunks sharing their first 2,000 characters keep separate ids.
- `lookup` accepts an id (`e3` or `[e3]`), the exact source a tool returned (a URL, a datasource id, a path), or — in local sessions only — the absolute path of a real local file the model opened itself.

### Persistence across resume

A bound TUI conversation persists its state as custom session entries and rebuilds it when the branch in view changes:

- Evidence is appended under `autorag.evidence` entries as tools return it, and result numbers under `autorag.citation-numbers`; the user's typed queries are saved alongside. On `session_start` (resume, `/new`, fork) and `session_tree`, the ledger, numbers, and queries are restored from the branch now in view.
- A malformed persisted entry is skipped, so its id resolves to nothing and the citation is dropped with a diagnostic — it is never mis-resolved.
- New evidence and new result numbers keep counting past every restored id/number **and** past the highest id/number issued anywhere in the conversation (other branches included), so an id or number a transcript may still show is never reissued.
- A programmatic run that owns no transcript starts from an empty ledger; a one-off `searchDocuments()` call is its own conversation.

## The output contract

The answer is the model's last assistant message when it ends its turn (`stopReason: "stop"`) without a tool call. Text written before a tool call is a progress note, never the answer. The model cites inline — `[e3]`, `[e3, e7]`, or `[file:<absolute path>]` for a local file it opened itself — and the harness (`deriveResultsFromAnswer`) resolves each marker against the ledger, rewrites it to a result number (`[1]`, `[2]`, …) in order of first appearance, and derives **one result per cited source**.

- A source's number comes from the conversation's `CitationNumbers`: a source numbered in an earlier answer (or an earlier phase of this turn) keeps its number, and a new source gets the next unused one. Numbers are never reissued.
- A marker that matches no recorded evidence, and any numeric marker the model wrote on its own, is **dropped** and reported (see below); it can never point at the wrong result.
- A source cited twice in a row is collapsed so the answer does not read `[1][1]`.

A derived result carries: `number`, `title` (from the source name), `summary` (the sentence the citation backs, capped at 280 characters), and `evidence[]` — one entry per cited chunk with an `excerpt` (capped at 400 characters) plus the measured `lineNumber`/`lineEnd`, `pageStart`/`pageEnd`, and `section`. Derived results carry no model-reported `confidence`, because nothing measured one.

`SearchDocumentsResponse` returns `sessionId`, `query`, `results`, `answer`, `searched`, `warnings`, `diagnostics`, and (on the degraded fallback path) `retrievalTrace`.

## Citation derivation diagnostics

Deriving the results produces diagnostics when the answer and the evidence disagree:

| Code | Meaning |
|---|---|
| `citation-without-result` | One or more answer citations matched no evidence retrieved in this conversation; they were dropped. Every remaining citation resolves to a recorded source |
| `line-reference-unverified` | The answer names line numbers that no cited evidence covers; trust `results[].evidence.lineNumber`/`lineEnd`, which the harness measured |
| `page-reference-unverified` | The answer names page numbers that no cited evidence covers; trust `results[].evidence.pageStart`/`pageEnd`, which the parser measured |
| `citation-irrelevant` | Jev removed cited chunks found unrelated to the sentence or the question (see [jev.md](jev.md#citation-relevance-check)) |
| `citation-relevance-fallback` | The citation relevance check could not run; every citation was kept |

A final reconciliation (`stripUnresolvedCitations`) enforces the response invariant: every `[n]` in the returned answer resolves to a `results[].number`, and a citation with no matching result is removed and reported.

## Mapping and report schema

Alongside the numbered results, the harness produces a `mapping`: one entry per result number with the `source`, `method`, raw `content`, and the full `evidenceRefs` (method, source, excerpt/content, chunk metadata, measured location). `results` and `mapping` are one-to-one on positive numbers.

A `reportSchema` ([src/agent/results.ts](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/agent/results.ts)) is the persisted/typed shape and the input contract for external curation, where an agent curated the evidence and no harness ledger exists. It requires a one-to-one `results`/`mapping`, result and evidence `confidence` in `[0, 1]`, and fills a default `evidenceRefs` entry from each mapping entry's `source`/`method`/`content` when none is given. The librarian itself never produces this shape from the model: it writes a plain answer and the harness derives the results.

- **`autorag lite report <query> [--input FILE]`** persists a structured report submitted by an external curator (reads the file, or stdin when `--input` is omitted). MCP exposes it as `autorag.report`.
- **`autorag evidence <sessionId> [--result N]`** shows the persisted source/chunk evidence attached to one search session (a `searchDocuments` run). It reads the curated results and evidence chunks from the retrieval memory file (`~/.autorag/memory.json` by default), filters to the session, and prints each result's chunks with `source`, `method`, `evidenceId`, `chunkIndex`, `lineNumber`/`lineEnd`, `pageStart`/`pageEnd`, `section`, `excerpt`, and `content`. `--result N` narrows to one result number; the command exits 2 when no evidence is found and 1 on error.
- MCP also exposes `autorag.evidence` for the same evidence, keyed by session.

## How locations are measured

A location (`locateChunk`) is **measured**, never model-written:

- `lineStart`/`lineEnd` are the 1-based first and last lines of the chunk in its parsed document. Lines are compared trimmed, so indentation and blank-line differences between the chunker's output and the file do not matter.
- `section` is the nearest article or heading line at or before the chunk's first line: a Korean statute heading (`제5조(…)`, `제2장 …`) or a markdown heading.
- `pageStart`/`pageEnd` are set only for a document the parser gave **real pages** (page mode `"layout"`) and only when both ends of the chunk land on a page the parser placed. A `"section"` page is a section-level approximation and is never reported as a page number; a line on an unplaced page (a table continued across pages) belongs to no known page, so no page is reported. Formats without pages (DOCX, text, spreadsheets) have none.
- A chunk is located only when it matches **exactly one place**. A chunk the chunker rewrote (table flattening) or one that occurs twice has no single correct line, and reporting nothing is better than a guess.

The measured location is rendered beside `[eN]` lines and CLI output as `(lines 7-8, section "…")`, `(page 3)`, and so on. For MinSync results the pages come from a per-document page index persisted beside the parsed mirror, and are used only while it was measured on exactly that markdown text.
