---
title: "Installation"
description: "Install the AutoRAG CLI, meet its prerequisites, verify the setup on macOS, Linux, or Windows, and keep it updated."
---

# Installation

AutoRAG Agent ships as the npm package `@autorag/librarian`, which installs two executables: `autorag` (the CLI) and `autorag-mcp` (the model-free MCP server). This page covers prerequisites, installation, verification, platform notes, and updates.

The fastest path is to hand a prompt to your coding agent: see [Agent skills](../agents/skills.md) for the `autorag-setup` and `autorag-lite-setup` skills, which install, configure, and verify AutoRAG for you.

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Node.js | `>=24.0.0` | The package's `engines` field requires Node 24. |
| Bun | `>=1.2` | Optional; Bun can install and run the CLI instead of Node. |
| Rust toolchain | optional | Only needed to auto-install Jikji from crates.io; without it, `jikji-unavailable` and Jikji discovery is skipped. |

No Java is required: document parsing for HWP/HWPX/HWPML, PDF, DOCX, and XLSX/XLS runs in-process through [kordoc](https://github.com/chrisryugj/kordoc).

Check the runtime before installing:

```bash
node --version      # must be >= 24
bun --version       # optional
```

## Install globally

```bash
# with Bun
bun install -g @autorag/librarian

# or with npm
npm install -g @autorag/librarian
```

To embed the librarian in another app, add it as a dependency instead: `bun add @autorag/librarian` or `npm install @autorag/librarian`.

Verify both executables are on `PATH`:

```bash
command -v autorag
command -v autorag-mcp
```

## Verify the install

```bash
autorag --version       # prints the package version, exit 0
autorag --help          # prints the command list, exit 0
autorag setup           # probes runtime and datasources, exit 0 when everything is ready
```

`autorag setup` probes MinSync, the embedding backend (a configured remote endpoint, a running `inl serve` daemon, or MinSync's in-process engine), and every known datasource. When no config exists yet it writes a default one first, using `--search-paths` and `--workspace`. It exits 0 when every probe is ready and 1 when something is blocked. Add `--format json` for machine-readable output.

```text
setup: semantic
embedding: daemon (minsync/Qwen/Qwen3-Embedding-0.6B, 1024d)
  kakao: configured
  rss: skipped (not configured)
```

**Note:** `autorag setup` only probes; it never starts the embedding daemon. Semantic search needs a running `inl serve` daemon or a configured remote `minSync.embedder`. The write-side setup (indexing, model configuration) is covered in [Quickstart](quickstart.md).

## Platform notes

**MinSync (all platforms).** MinSync is required and auto-installs into `<workspace>/.autorag/bin`, from crates.io (`cargo install minsync`) or, when cargo is unavailable, a verified GitHub release asset. It shares im-not-llm's model cache at `~/.cache/im-not-llm/models`. There is no supported way to disable MinSync.

**Jikji (all platforms, optional).** AutoRAG auto-installs the pinned `jikji-cli` through cargo when a Rust toolchain exists. A `jikji` on `PATH` older than the pinned version is skipped and replaced by the cached build on the next refresh.

**filee-cli (all platforms, automatic).** `autorag setup` and the TUI download the pinned filee-cli release (SHA-256 verified) into `<AUTORAG_HOME>/tools/filee-cli/<version>/` so the agent can convert between formats offline. A `filee`/`filee-cli` already on `PATH` is used instead. Set `"filee": { "autoInstall": false }` to forbid the download.

**Windows file-name search (bundled).** The package bundles the portable voidtools Everything 1.4.1.1032 and its ES 1.1.0.38 CLI (x64 and ARM64, SHA-256 pinned). AutoRAG extracts them into `<workspace>/.autorag/everything/<version>/<arch>/` and runs a private, user-level instance that indexes only your configured search roots. It never requests elevation, installs the Everything service, indexes whole volumes, or enables the HTTP/ETP servers, and it does not touch an Everything you already run. Set `"everything": false` to turn it off.

**macOS/Linux file-name search (separate install).** Install [`fsearch-cli`](https://github.com/NomaDamas/fsearch-mac) yourself (for example `brew install NomaDamas/fsearch-mac/fsearch-mac`, which installs the `fsearch-cli` binary). AutoRAG never bundles or links it; it spawns it as a separate process and builds a per-workspace database at `<workspace>/.autorag/fsearch/`. Without `fsearch-cli`, name search degrades to a slow bounded filesystem walk. Set `"fsearch": false` to turn it off.

## Updating

`autorag update-check` compares the running package against the published version on the npm registry and prints whether a newer release is available; it exits 0 either way. `autorag tui` runs the same check on launch and shows a one-line notice.

```bash
autorag update-check
# autorag v2.13.13 is up to date
```

Set `AUTORAG_NO_UPDATE_CHECK=1` to skip the lookup; `AUTORAG_UPDATE_CHECK_URL` overrides the registry endpoint. To upgrade, re-run the global install:

```bash
bun install -g @autorag/librarian
# or: npm install -g @autorag/librarian
```

See [Environment variables](../reference/environment-variables.md) for the full variable list.
