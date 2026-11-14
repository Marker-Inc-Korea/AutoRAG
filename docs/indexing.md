# Indexing options

Schedules, exclusions, parsers, OCR, and parse limits.

### Indexing only when you want it (power and time schedule)

Background indexing (`autorag watch`, auto-refresh, the fs watcher) can be limited by power source and local time. An explicit `autorag refresh` is never gated.

```jsonc
// ~/.autorag/config.json
{ "refresh": { "schedule": { "power": "ac", "windows": ["22:00-07:00"], "days": ["mon","tue","wed","thu","fri"] } } }
```

- `power`: `any` (default) or `ac` (index only while plugged in; pauses on battery). An unknown power source (desktop, probe failure) counts as plugged in.
- `windows`: local `HH:MM-HH:MM`, start inclusive, end exclusive; `22:00-07:00` wraps midnight and belongs to the day it starts on.
- `days`: `sun`..`sat`. Omit for every day.
- `autorag watch --power ac --window 22:00-07:00 --days mon,tue` overrides the config. `--once` while closed prints `skipped` and exits 0; a long-running watcher defers changes and indexes once the schedule opens.

### Excluding files and folders

Leave generated workspaces or archives out when initializing:

```bash
autorag init --search-paths "$HOME/Documents" \
  --exclude "$HOME/Documents/Build output" \
  --exclude "$HOME/Documents/Archive, old"
```

Repeat `--exclude` for each file or folder. Each argument is one complete path,
including spaces and commas; relative paths resolve from the current directory.
Excluding a folder also excludes its descendants. Paths are literal, not globs.

For an existing configuration, edit its `excludePaths` array and run
`autorag refresh` to update the indexes. File-name search applies exclusions to
native fsearch and Everything results as well as the filesystem walk. See the
[exclusion configuration examples](minsync-setup.md#excluding-files-and-folders).

### Document languages and parsers

One global `languages` setting describes the corpus. It is resolved from
`--languages`, then `AUTORAG_LANGUAGES`, then `languages` in the config file,
falling back to `["ko", "en"]`. Supported tags: `ko`, `en`, `ja`, `zh-hans`,
`zh-hant`, `fr`, `de`, `es`, `ru`, `it`, `pt`, `vi`, `th`, `ar`, `hi`.

| Extension | Parser |
|---|---|
| `.hwp` `.hwpx` `.hml` `.hwpml` `.pdf` `.docx` | `kordoc` (nested tables, no Java) |
| `.xlsx` `.xlsm` `.xlsb` `.xls` `.ods` | built-in spreadsheet reader (SheetJS) |
| `.csv` `.tsv` | built-in CSV reader (CP949/EUC-KR aware) |
| `.pptx` | built-in PPTX reader |
| `.eml` | built-in mail reader |
| `.txt` `.text` `.md` `.markdown` | plain text (CP949/EUC-KR aware) |
| `.png` `.jpg` `.jpeg` `.webp` | `kordoc` image OCR (only when `parserOptions.ocr.enabled` is `true`) |

`.bmp` and `.tiff` are not parsed: kordoc rejects both as unsupported.

The spreadsheet reader writes one line per row. Each line names the sheet, the
cell range and the table title, and pairs every value with its column label, so
a search hit always says what a number is:

```text
[예산총괄!A4:F4 · 2026년 연구개발비 편성표 (단위: 천원)] 구분: 직접비 | 비목: 인건비 | 편성액: 742,000 | 비율(%): 32.1%
```

Values are what the spreadsheet shows: cached formula results with their number
formats (`742,000`, `32.1%`, dates). A format that scales the stored number keeps
that number too (`742천원 (742000)`), and a formula saved without a result shows
its formula (`=SUM(B2:B3)`). Titles above a table become its caption, merged
group headers become `상반기 > 1월`, several tables on one sheet are kept apart,
and a row longer than a search chunk is split into lines that repeat its first
value. A first row counts as the header only with clear evidence (a merged group
header, or at least three rows below it with a mostly numeric, date or formula
column); otherwise rows are kept exactly as written, so a form's label/value
cells are never relabelled. Sheets are read up to 100,000 rows each; a longer
sheet is indexed partially with a `parser-warning` diagnostic.

`.csv` and `.tsv` files produce the same lines, with the file name in place of
the sheet name. Bytes decode as UTF-8 (with or without BOM) or CP949/EUC-KR,
quoted fields keep separators, line breaks and doubled quotes, and values stay
exactly as written (`00123` and `=1+2` are not converted). `.tsv` is
tab-separated; a `.csv` follows Excel's `sep=;` first line when present and
otherwise picks the separator (`,` `;` tab `|`) that splits its first lines
most consistently.

### Making photos and scans searchable

Image files and scanned PDF pages become searchable through one switch,
`parserOptions.ocr.enabled` (default `false`). It is off by default so indexing
downloads no recognition model and never runs OCR over a photo library. With it
off, image files are not parsed at all and scanned pages stay unrecognized.

```json
{ "languages": ["ko", "en"], "parserOptions": { "ocr": { "enabled": true } } }
```

`kordoc` owns both inputs. The configured `languages` pick the engine:

| `languages` | Engine | First-use download |
|---|---|---|
| only `ko` and/or `en` | kordoc built-in PP-OCRv5 (local CPU) | ~18 MB from Hugging Face, cached in `~/.cache/kordoc/models` |
| any other tag (`ja`, `zh-hans`, `fr`, `de`, …) | Tesseract, all configured languages at once (`kor+jpn`) | traineddata per language from the jsDelivr CDN, cached in `<workspace>/.autorag/models/tessdata` |

The built-in model has no kana, han, Cyrillic, Thai, Arabic or Devanagari
characters and drops diacritics in accented Latin text, so those languages need
Tesseract. Set `parserOptions.ocr.cachePath` to store the Tesseract models
elsewhere, or `KORDOC_OFFLINE=1` to block the built-in model download. An
explicit `parserOptions.kordoc.ocr` overrides the language-based choice.

### Bounding and cancelling a parse

`kordoc` (PDF, HWP/HWPX, DOCX, images), the spreadsheet reader, `.pptx` and `.eml` parse in a worker
thread during a refresh, so one corrupt or hostile file cannot hang or crash the run. A
`Promise.race` deadline cannot stop a synchronous parse; terminating the worker can.

| `parserOptions` key | Default | Effect |
|---|---|---|
| `fileTimeoutMs` | `120000` | Per-file deadline. A parse still running is killed with its worker. |
| `maxHeapMb` | Node's default | V8 old-generation limit for the worker. An out-of-memory parse is caught, not a host crash. |

A timeout, a heap overrun and a worker crash each become one `parser-failed` skip for that
file; the diagnostic message says which. The rest of the corpus is still indexed. Plain-text
files and parsers configured with a function (`parserOptions.kordoc.parse`, a custom OCR
provider) parse in-process.

```json
{ "parserOptions": { "fileTimeoutMs": 60000, "maxHeapMb": 2048 } }
```

Photos and videos are also reachable through two other paths that are not part
of this switch: the `clawgallery` datasource (captions and visual search over its
own folders) and Jikji's media index (`jikji.enableMediaIndex`, file facts such as
pixel size and format only, no content). Video content is not searchable yet.
