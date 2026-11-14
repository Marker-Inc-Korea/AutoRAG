# Licenses and third-party notices

Licensing detail for bundled and separately installed components. Full texts: [`NOTICE`](../NOTICE) and [`licenses/`](../licenses/).

## License

- **AutoRAG 2.0 (AutoRAG Agent):** Released under the [Apache License 2.0](../LICENSE).
- **Legacy Python AutoRAG (`legacy/`):** Released under the [Apache License 2.0](../legacy/LICENSE).
- Production third-party notices and licenses are documented in [`NOTICE`](../NOTICE).
- The bundled Windows binaries keep their own licenses: Everything (MIT, plus PCRE BSD-3-Clause) in [`licenses/everything-MIT-and-PCRE-BSD.txt`](../licenses/everything-MIT-and-PCRE-BSD.txt) and ES (MIT) in [`licenses/es-MIT.txt`](../licenses/es-MIT.txt). Everything's source code is not public; AutoRAG redistributes the unmodified voidtools portable ZIPs.
- FSearch/fsearch-cli (GPL-2.0-or-later) are **not** distributed with AutoRAG: no code is copied, linked, or bundled. Users install fsearch-cli themselves and AutoRAG communicates with it only through its command-line interface and daemon socket — separate programs, per the [FSF mere-aggregation FAQ](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation). Source: https://github.com/NomaDamas/fsearch-mac.
- Hister (AGPL-3.0-or-later) is **not** distributed with AutoRAG: no Hister code is copied, linked, bundled, or modified. Users install an unmodified Hister release themselves and AutoRAG talks to that separate process only over its public HTTP API. Notice: [`licenses/hister-AGPL-3.0-notice.txt`](../licenses/hister-AGPL-3.0-notice.txt). If you run Hister as a service for other people, AGPL section 13 applies to you as its operator.

## Acknowledgements

AutoRAG Agent stands on the shoulders of fantastic open-source projects:

- **[Pi Framework](https://github.com/earendil-works/pi-mono)** by [@earendil-works](https://github.com/earendil-works) — The foundational agent framework powering AutoRAG's reasoning loop.
- **[MinSync](https://github.com/Marker-Inc-Korea/minsync)** — Ultra-fast incremental Change Data Capture (CDC) chunking, local BM25/vector indexing, and the Qwen3-Embedding-0.6B embedding engine.
- **[Jikji](https://github.com/NomaDamas/jikji)** by [NomaDamas](https://github.com/NomaDamas) — High-performance find-first local document discovery.
- **[Everything](https://www.voidtools.com/)** and **[ES](https://github.com/voidtools/ES)** by voidtools (David Carpenter) — Instant Windows file-name indexing and its command-line interface, bundled under the MIT License.
- **[FSearch](https://github.com/cboxdoerfer/fsearch)** and **[fsearch-cli](https://github.com/NomaDamas/fsearch-mac)** — Everything-style file-name search for macOS/Linux (GPL-2.0-or-later). Never bundled or linked: users install fsearch-cli separately and AutoRAG spawns it as a separate process (mere aggregation per the FSF GPL FAQ).
- **[Hister](https://github.com/asciimoo/hister)** by [asciimoo](https://github.com/asciimoo) and contributors — Self-hosted browser web-history search (AGPL-3.0-or-later). Never bundled, linked, copied, or modified: users run Hister as a separate program and AutoRAG only calls its public HTTP search API (mere aggregation). Hister is always named as the backend of the `hister` datasource.
- **[dupey](https://github.com/NomaDamas/dupey)** by [NomaDamas](https://github.com/NomaDamas) — Fast duplicate and near-duplicate document family detection.
- **[Filee](https://github.com/KnifeLemon/Filee)** by [KnifeLemon](https://github.com/KnifeLemon) — Offline conversion of 240+ file formats (MIT). Never bundled or linked: AutoRAG downloads the SHA-256-pinned [`filee-cli`](https://github.com/NomaDamas/filee-cli) release (NomaDamas' CLI-only redistribution, MIT) into `<AUTORAG_HOME>/tools/filee-cli/` on `autorag setup` / first conversion and spawns it as a separate process. An existing `filee`/`filee-cli` on `PATH` or `filee.binaryPath` wins. Filee downloads its own engines (FFmpeg, Pandoc, LibreOffice, …) on first use.
- **Federated CLI Authors:** External datasource tools [`lazykatok`](https://github.com/changeroa/lazykatok), [`discrawl`](https://github.com/openclaw/discrawl), [`mailcrawl`](https://github.com/NomaDamas/mailcrawl), [`wacrawl`](https://github.com/openclaw/wacrawl), [`telecrawl`](https://github.com/openclaw/telecrawl), [`slacrawl`](https://github.com/openclaw/slacrawl), and [`notcrawl`](https://github.com/openclaw/notcrawl).
- **[kordoc](https://github.com/chrisryugj/kordoc)** by [chrisryugj](https://github.com/chrisryugj) — HWP/HWPX/HWPML, PDF, DOCX and XLSX parsing with nested-table fidelity, used as AutoRAG's default document parser.

---
