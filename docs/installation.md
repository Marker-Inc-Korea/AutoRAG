# Installation and prerequisites

Detailed install, prerequisites, and first-run setup. The fastest path is to hand the prompts in the [README](../README.md#install-it-with-your-agent) to your coding agent.

## Installation & Setup

AutoRAG Agent is published as `@autorag/librarian` (requires Node.js ≥ 24 or Bun):

```bash
# Install CLI globally
bun install -g @autorag/librarian
# or with npm:
npm install -g @autorag/librarian

# Add as a TypeScript/JavaScript library
bun add @autorag/librarian
```

### System Prerequisites

- **No Java required:** Document parsing (HWP/HWPX/HWPML, PDF, DOCX, XLSX/XLS) runs in-process through [`kordoc`](https://github.com/chrisryugj/kordoc).
- **Rust Toolchain (Optional):** Automatically compiles Jikji (`jikji-cli`, pinned to 0.2.1) if installed. A `jikji` older than 0.2.0 on your PATH is skipped and replaced by the pinned build on the next refresh.
- **MinSync:** Automatically downloaded and installed into `<workspace>/.autorag/bin` on first run.
- **Everything (Windows only, bundled):** The package ships the portable Everything 1.4.1.1032 and its ES 1.1.0.38 CLI (x64 and ARM64, SHA-256 pinned). On Windows, AutoRAG extracts them into `<workspace>/.autorag/everything/` and runs a private, user-level instance that indexes only your configured search roots: no administrator rights, no Everything service, no whole-drive scan, no HTTP/ETP server, and no change to any Everything you already run. Set `"everything": false` in the config to turn it off. macOS and Linux are not affected.
- **filee-cli (file conversion, automatic):** `autorag setup` and the TUI download the pinned filee-cli release (~85 MB, SHA-256 verified) into `<AUTORAG_HOME>/tools/filee-cli/<version>/`, so the agent can convert between 255 formats offline with `convert_files`. A `filee`/`filee-cli` already on `PATH` is used instead. Set `"filee": { "autoInstall": false }` to forbid the download or `"filee": { "enabled": false }` to remove the tools. Filee fetches its own engines (FFmpeg, Pandoc, LibreOffice, calibre, Ghostscript; hundreds of MB, only when a conversion needs one) on first use.
- **FSearch (macOS/Linux only, separate install):** Install [`fsearch-cli`](https://github.com/NomaDamas/fsearch-mac) yourself (e.g. `brew install NomaDamas/fsearch-mac/fsearch-mac` (installs the `fsearch-cli` binary); GPL-2.0, spawned as a separate process, never bundled or linked). On macOS and Linux, AutoRAG builds a per-workspace database at `<workspace>/.autorag/fsearch/` indexing only your configured search roots, and keeps a `fsearch-cli watch` daemon live (FSEvents/inotify) for sub-second name search. It never touches the FSearch app's own database. Without fsearch-cli, name search falls back to a slow bounded filesystem walk. Set `"fsearch": false` in the config to turn it off. Windows is not affected (Everything covers it).

---

## Quick Start

### 1. Initialize and Index

```bash
# Initialize configuration for your documents folder
autorag init --search-paths ~/Documents/research

# Korean + English is the default; set it explicitly for other corpora
autorag init --search-paths ~/Documents/research --languages ja,en

# Index documents (parses PDFs/Markdown, builds MinSync sparse and dense vectors)
autorag refresh

# Check indexing health
autorag status
```

### 2. Search from CLI

`autorag search` needs one chat model. `autorag init` does not pick one, so on
first run either sign in and choose a model inside the TUI:

```bash
autorag tui      # then /login (OAuth or API key) and /model
```

or add the `model` entry to your existing AutoRAG config, preserving its other
settings. Find provider/model ids with `autorag models list --available`:

```json
"model": { "provider": "<provider>", "id": "<id>" }
```

Then run `autorag health`. Do not re-run `autorag init --force` just to select
a model: it replaces the existing config, including datasource and parser settings.
Configure the provider's credentials through `/login` or its supported environment variables.

```bash
# Perform a curated search (uses your configured reasoning model)
autorag search "What are our primary Q3 deliverables?"

# Launch the interactive Terminal UI (Pi host: /login, /model, /resume, …)
autorag tui
```
