---
title: "Parsing and languages"
description: "Which file formats AutoRAG parses, how it selects OCR languages, and how it bounds, cancels, and resumes parsing."
---

# Parsing and languages

This page documents the parsers AutoRAG uses for each format, how the global `languages` setting selects OCR, and how a parse is bounded, cancelled, and resumed across refreshes.

## Document languages

One global `languages` setting describes the corpus. It is resolved from `--languages`, then `AUTORAG_LANGUAGES`, then `languages` in the config file, and falls back to `["ko", "en"]`. The first entry is the primary language.

Supported tags: `ko`, `en`, `ja`, `zh-hans`, `zh-hant`, `fr`, `de`, `es`, `ru`, `it`, `pt`, `vi`, `th`, `ar`, `hi`. The set is curated because every tag must map to a Tesseract traineddata code. Format parsing itself is language-agnostic; `languages` only selects the OCR engine (below).

## Parsers by extension

| Extension | Parser |
| --- | --- |
| `.hwp` `.hwpx` `.hml` `.hwpml` `.pdf` `.docx` | `kordoc` (nested tables, no Java) |
| `.xlsx` `.xlsm` `.xlsb` `.xls` `.ods` | built-in spreadsheet reader (SheetJS) |
| `.csv` `.tsv` | built-in CSV reader (CP949/EUC-KR aware) |
| `.pptx` | built-in PPTX reader |
| `.eml` | built-in mail reader |
| `.txt` `.text` `.md` `.markdown` | plain text (CP949/EUC-KR aware) |
| `.png` `.jpg` `.jpeg` `.webp` | `kordoc` image OCR, only when `parserOptions.ocr.enabled` is `true` |

`.bmp` and `.tiff` are not parsed: kordoc rejects both as unsupported. A file with no registered parser is not parsed. A source larger than the parse-size cap (default 50 MiB), or an image too small to OCR, is skipped with a `parser-skipped` diagnostic.

## Spreadsheets, CSV, and TSV

The spreadsheet reader writes one line per row. Each line names the sheet, the cell range, and the table title, and pairs every value with its column label:

```text
[예산총괄!A4:F4 · 2026년 연구개발비 편성표 (단위: 천원)] 구분: 직접비 | 비목: 인건비 | 편성액: 742,000 | 비율(%): 32.1%
```

Values are what the spreadsheet shows: cached formula results with their number formats (`742,000`, `32.1%`, dates). A format that scales the stored number keeps that number too (`742천원 (742000)`); a formula saved without a result shows its formula (`=SUM(B2:B3)`). Titles above a table become its caption, merged group headers become `상반기 > 1월`, several tables on one sheet are kept apart, and a row longer than a search chunk is split into lines that repeat its first value. A first row counts as the header only with clear evidence (a merged group header, or at least three rows below it with a mostly numeric, date, or formula column); otherwise rows are kept exactly as written, so a form's label/value cells are never relabelled. Sheets are read up to 100,000 rows each; a longer sheet is indexed partially with a `parser-warning` diagnostic.

`.csv` and `.tsv` produce the same lines, with the file name in place of the sheet name. Bytes decode as UTF-8 (with or without BOM) or CP949/EUC-KR, quoted fields keep separators, line breaks, and doubled quotes, and values stay exactly as written (`00123` and `=1+2` are not converted). `.tsv` is tab-separated; a `.csv` follows Excel's `sep=;` first line when present and otherwise picks the separator (`,` `;` tab `|`) that splits its first lines most consistently. CSV and TSV rows are capped at 100,000 per file.

## Images and scanned pages

Image files and scanned PDF pages become searchable through one switch, `parserOptions.ocr.enabled` (default `false`). It is off by default so indexing downloads no recognition model and never OCRs a photo library. With it off, image files are not parsed and scanned pages stay unrecognized.

```json
{ "languages": ["ko", "en"], "parserOptions": { "ocr": { "enabled": true } } }
```

`kordoc` owns both inputs. The configured `languages` pick the engine:

| `languages` | Engine | First-use download |
| --- | --- | --- |
| only `ko` and/or `en` | kordoc built-in PP-OCRv5 (local CPU) | ~18 MB from Hugging Face, cached in `~/.cache/kordoc/models` |
| any other tag (`ja`, `zh-hans`, `fr`, `de`, …) | Tesseract, all configured languages at once (e.g. `kor+jpn`) | traineddata per language from the jsDelivr CDN, cached under `<workspace>/.autorag/models/tessdata` |

The built-in model has no kana, han, Cyrillic, Thai, Arabic, or Devanagari characters and drops diacritics in accented Latin text, so those languages need Tesseract. Set `parserOptions.ocr.cachePath` to store the Tesseract models elsewhere, or `KORDOC_OFFLINE=1` to block the built-in model download. An explicit `parserOptions.kordoc.ocr` overrides the language-based choice.

Photos and videos are also reachable through two paths that are not this switch: the `clawgallery` datasource and Jikji's media index (`jikji.enableMediaIndex`, file facts such as pixel size and format only). Video content is not searchable.

## Bounding and cancelling a parse

`kordoc` (PDF, HWP/HWPX, DOCX, images), the spreadsheet reader, `.pptx`, and `.eml` parse in a worker thread during a refresh, so one corrupt or hostile file cannot hang or crash the run. A `Promise.race` deadline cannot stop a synchronous parse; terminating the worker can.

| `parserOptions` key | Default | Effect |
| --- | --- | --- |
| `fileTimeoutMs` | `120000` | Per-file deadline. A parse still running is killed with its worker. |
| `maxHeapMb` | Node's default | V8 old-generation limit for the worker. An out-of-memory parse is caught, not a host crash. |

A timeout, a heap overrun, and a worker crash each become one `parser-failed` skip for that file; the diagnostic message says which. The rest of the corpus is still indexed. Plain-text files and parsers configured with a function (`parserOptions.kordoc.parse`, a custom OCR provider) parse in-process.

```json
{ "parserOptions": { "fileTimeoutMs": 60000, "maxHeapMb": 2048 } }
```

To stop a long first index entirely, send `SIGINT` or `SIGTERM` to `autorag refresh`; see [Refresh operations](refresh-operations.md#cancelling-a-refresh).

## Durable incremental parsing

Completion is recorded only after the final Markdown file exists and MinSync has measured and verified it. Parsed mirrors require MinSync's extended stage protocol (protocol 2). AutoRAG probes capabilities before opening a run and refuses an older build before mutating stage state.

- Unchanged, verified mirrors are not read or parsed again. Their document-level timestamps stay stable, so warm refreshes do not copy the whole corpus into the retrieval workspace again. The durable source revision uses nanosecond mtime and size; `chmod`, xattr, inode, and device changes alone do not reparse a file. As with other stat-based scans, a content edit that preserves both source mtime and size between runs is not detected.
- Parser name/version and output-affecting options invalidate completed work. Execution deadlines, worker heap limits, and OCR cache location only invalidate recorded failure skips, not healthy outputs. Language and OCR enablement changes still invalidate output. Stage keys are fixed-length SHA-256 IDs.
- A source changed during parsing is not committed. Its last good mirror stays available, remains reported as stale, and is retried on the next refresh.
- Missing or damaged mirrors are rebuilt. MinSync's fast verification hashes an output when its modification time changes; it does not detect deliberate edits that preserve both size and modification time.
- An interruption leaves committed items in MinSync's journal. The next matching refresh resumes them, including an interrupted forced refresh. There is no AutoRAG-specific periodic full-index checkpoint.
- AutoRAG no longer writes a separate `.autorag/parsed/index.json` or reads MinSync's private snapshot/journal formats. Downstream source mappings come from MinSync's verified `inspect` response, including committed work left by a hard host crash.
- Failed work is never a successful completion. Permission failures are paced with a skip record until the source, parser, or retry policy changes or `--retry-failed` is requested.

## Related pages

- [Overview](overview.md) — the whole pipeline and workspace layout.
- [Refresh operations](refresh-operations.md) — diagnostics, retries, and cancellation.
- [Embedding backends](embedding-backends.md) — the embedding model and daemon.
- [Datasources overview](../datasources/overview.md) — parsing inside datasource skills.
