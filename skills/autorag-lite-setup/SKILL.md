---
name: autorag-lite-setup
description: Install and register the model-free AutoRAG Lite MCP server, configure approved roots and datasources, build indexes, and verify MCP discovery and search. Use when autorag-mcp is missing, MCP connection fails, indexes are stale, or the user wants document search without configuring a search model.
license: MIT
---

# AutoRAG Lite setup

Use this skill only to bootstrap, register, or repair AutoRAG Lite MCP:
initialize trusted config, connect the host, build indexes, and verify search.
Normal Lite use is MCP tool calling, not a search skill or shell command.
Use `autorag-setup` instead when the model-backed librarian must be configured.

## Safety

- Inspect only non-secret config metadata: `searchPaths`, `workspacePath`,
  `memoryPath`, `minSync`, `jikji`, `datasources`.
- Never print, copy, migrate, compare, or persist credential values. Store only
  environment-variable names such as `tokenEnv` or `apiKeyEnv`.
- Never move, rename, edit, or delete source documents. Lite commands write
  indexes only under the configured workspace `.autorag/` directory and
  Jikji's per-source `.jikji/` caches.
- Do not index system trees, app bundles, caches, credential stores,
  `node_modules`, `.git`, `dist`, `build`, `target`, `.cache`, `.autorag`, or
  `.jikji`.

## Install the package if needed

`@autorag/librarian` ships both `autorag` (bootstrap/maintenance) and
`autorag-mcp` (stdio server). The executable requires Node.js >= 24 on PATH.

```bash
command -v autorag-mcp >/dev/null || bun install -g @autorag/librarian
command -v autorag
command -v autorag-mcp
autorag lite --help
```

If Bun is unavailable, `npm install -g @autorag/librarian` is acceptable.
If only `autorag` exists, upgrade the package rather than substituting CLI
retrieval for MCP. For a source checkout, run `bun run build` and register the
absolute `dist/mcp/index.js` path instead of the installed executable.

## Initialize a model-free config

```bash
autorag lite init \
  --search-paths "/path/to/documents,/path/to/notes" \
  --workspace "/path/to/workspace" \
  --memory-path "/path/to/memory.json"
```

`lite init` writes the configured model-free lifecycle config. Use `--force`
only when intentionally replacing an existing config, and target it with an
explicit path (`--config` or `AUTORAG_CONFIG`); `--force` refuses to replace the
implicit `~/.autorag/config.json`. Explicit user paths win;
otherwise propose one to three document-dense roots and get approval before
indexing. Supported parsed formats are `md`, `markdown`, `txt`, `text`, `pdf`,
`docx`, `pptx`, `xlsx`, `xls`, `hwp`, `hwpx`, and `eml`. Legacy `.doc` is not
a supported parsed format.

Config resolution follows the usual order: `--config`, `AUTORAG_CONFIG`,
`$AUTORAG_HOME/config.json`, or `~/.autorag/config.json`. Environment overrides
include `AUTORAG_HOME`, `AUTORAG_CONFIG`, `AUTORAG_SEARCH_PATHS`,
`AUTORAG_WORKSPACE`, and `AUTORAG_MEMORY_PATH`.

## Register and verify the MCP connection

After config and datasource setup, register the server with the host. Use
absolute config, search root, workspace, and executable paths: hosts may start
the server from a different working directory or with a restricted PATH.
Resolve `command -v autorag-mcp` and substitute its absolute path below.
Inspect an existing `autorag` registration first; keep a working registration
and update only an outdated command or config path.

```bash
# Claude Code: project-local registration
claude mcp add --transport stdio --scope local autorag \
  --env AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json \
  -- /absolute/path/to/autorag-mcp

# Codex: user registration
codex mcp add autorag \
  --env AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json \
  -- /absolute/path/to/autorag-mcp
```

For other hosts, use their stdio MCP configuration with the same command,
empty arguments, and `AUTORAG_CONFIG` environment variable. The host starts
the subprocess; do not run it as a background HTTP service. Pass any required
credential environment-variable names through the host's secret mechanism;
never put credential values in registration examples or logs.

Reload/reconnect the host, then use MCP `tools/list` to discover the actual
tools and schemas. Verify `autorag.status`, `autorag.refresh`, `autorag.search`,
`autorag.search.files`, and `autorag.datasources.list` are available. Configured
integrated datasources add their own search tools; do not assume a fixed count.
Run `autorag.status` and `autorag.datasources.list` through MCP, then refresh
and search a known phrase from an approved document. Registration alone is
not proof of a connected or searchable server. After config changes, restart
the server so datasource tools are rebuilt.

`AUTORAG_MCP_READ_ONLY=1` omits refresh; build indexes with the CLI before
connecting that mode. `AUTORAG_MCP_TOOLS` is an optional comma-separated exact
tool allowlist; omitted tools cannot be called. Use unrestricted tools for
initial setup unless the user intentionally requests a restricted connection.


## Probe and configure datasources (setup wizard)

A datasource setup UI is not shipped in this build — do not recommend it for datasource
setup. Configure datasources directly in trusted config, wizard-style:

1. Probe every datasource for setup feasibility before asking the user
   anything: the backing CLI exists (`lazykatok`, `discrawl`, `slacrawl`,
   `wacrawl`, `telecrawl`, `notcrawl`, `qmd`, `mailcrawl`, `rclone`) and its
   local store or archive is present. CLI-backed datasources own their own
   archive, index, and authentication, so environment credentials (such as bot
   tokens) are never required or checked for them. Non-CLI connectors
   (such as `github`) require their credential environment variable
   (`GITHUB_TOKEN`).
2. Auto-configure every datasource that probes feasible — write its trusted
   `datasources` entries without asking. For example,
   when Slack (`slacrawl`) and Discord (`discrawl`) are installed with local
   stores present, set both up automatically. Discord uses discrawl's local
   desktop wiretap archive; no Discord bot token is configured or needed.
3. Skip every datasource that probes infeasible (for example Notion or
   Telegram when their CLIs are not installed) and always report the skipped
   list to the user, with what is missing for each.
4. Set up a skipped datasource only when the user explicitly asks for it:
   install or authenticate the backing CLI first, then configure it.
5. E-mail datasources (`mail-export`, `mailcrawl`) matter to most
   users — always probe them and report their status, even when they end up
   skipped.

Datasource skills belong in trusted config.

Config keys may be builtin template names (`kakao`, `whatsapp`, `telegram`,
`slack`, `discord`, `clawgallery`, `notion`, `github`, `cloud-drive`,
`mail-export`, `mailcrawl`, `obsidian`, `rss`, `spotlight`, `lark`) or connection
aliases with `"type": "<template>"`. Unknown names are skipped with an
`unknown-datasource-skill` warning; they do not fail config resolution.
`scope` narrows a query to a sub-path as ordinary filtering. Tags are
descriptive metadata only, not search filters. Use MCP `datasourceIds` to
select configured connections before retrieval; discover their IDs with
`autorag.datasources.list`. Store only env-var names such as `tokenEnv` or
`apiKeyEnv`, never credential values.

## Build and refresh indexes

After connecting, call MCP `autorag.refresh` with `{}` for an incremental run.
Use `{"methods":["parsed","minsync"]}` to narrow indexing, or
`{"force":true}` only when a full resync is needed. Omit `methods` to run all
configured methods; MCP does not accept `"all"` as a method value.

CLI refresh remains available for bootstrap, read-only deployments, and repair:

```bash
AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json autorag lite refresh --json
```

- Refresh syncs parsed mirrors, MinSync, Jikji, and configured datasources.
- CLI `--full` and `--force` request a full resync; MCP uses `force: true`.
- CLI `--method <csv>` accepts `parsed`, `minsync`, `datasources`, `jikji`,
  `everything` (Windows), `fsearch` (macOS/Linux), and `all`. MCP `methods`
  accepts the same individual methods, not `all`. Unknown values are rejected.
- Indexing never runs while answering: retrieval only reads what refresh built.
  MinSync and Jikji auto-install during refresh (never during a query). If they
  are missing or broken, run a full refresh or return to setup rather than
  silently degrading to lexical-only search.
- MinSync's default embedder is in-process native Qwen3 embeddings
  (`native:Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions, MinSync 0.4.5+): no
  embedder flags, no API key, and no external daemon (such as Ollama) are
  needed, and no corpus text leaves the machine. For gateway-profiled
  embeddings, prefetch with `autorag models prefetch --profile qwen3-embedding-0.6b`.
  The legacy Ollama/TEI adapter path (EmbeddingGemma, 768 dimensions) is for
  manual QA only. Override the embedder only when intentionally using a
  different provider.
- Exact duplicate exclusion during refresh is enabled by default via the
  external `dupey` CLI. Install dupey during setup when it is missing
  (`command -v dupey || cargo install dupey --locked`) and tell the user the feature is
  available; when installation is impossible, refresh continues without it
  and the user is told duplicate exclusion is off. Set
  `"excludeExactDuplicates": false` to index every copy.

Content search requires a completed refresh. MCP `autorag.search` returns
`isError: true` with `errorCode: "index-not-ready"` until refresh completes.
Always refresh first, and refresh again when roots change. Stale indexes can
still return results with `stale: true` and diagnostics; use `strict: true`
to reject stale answers, or call `autorag.refresh` before searching. Sources
deliberately skipped during refresh do not count as stale.
Jikji is a discovery/indexing preparer, not a lite retrieval method.

## Watch and scheduled freshness

```bash
autorag lite watch --once --json
autorag lite watch
```

Prefer non-daemon `autorag lite watch --once` from cron, launchd, a systemd
user timer, or Task Scheduler, hourly by default (every 1 hour; shorten only
when the user asks for fresher indexes). Use the same config as retrieval,
avoid overlapping runs, and keep logs outside source trees. `--immediate`
triggers a first refresh on start and `--debounce-ms N` tunes change
coalescing. Once the schedule is installed, tell the user right away that
hourly freshness is set up.

## Status and index maintenance

Use MCP `autorag.status` for health and `autorag.duplicates` for read-only
duplicate-family inspection. No search model is required. CLI-only repair:

```bash
AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json autorag lite index rebuild --yes --json
AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json autorag lite index reset --method parsed --yes --json
```
- `index reset` and `index rebuild` remove or rebuild only workspace
  `.autorag` indexes selected by `--method`. They never target source
  documents.
- `duplicates` scans duplicate document families read-only and never deletes
  or moves files.

## Unavailable components and failure handling

Inspect MCP `isError`, `errorCode`, `diagnostics`, and `unsearched` before
trusting results. `ok: true` with skipped surfaces is partial coverage, not
proof that the whole corpus was searched. Preserve the underlying error text
when reporting unavailable components such as `retrieval-method-failed`.
MinSync is required: when its binary is missing the call fails with an
`MinSync is required ...` error rather than returning partial results, so
install it (`cargo install minsync`) instead of working around it. Repair the
component rather than silently accepting degraded search.

## Normal MCP workflow and curation lifecycle

MCP results carry source, method, and content directly. After a search, the
optional curation lifecycle is also exposed through MCP: `autorag.report`
records a curated answer, and `autorag.evidence` returns the exact chunks behind
numbered results. These accept and return JSON matching the CLI's
report/evidence contracts; discover the exact schemas with MCP `tools/list`
rather than restating them here. Every bracketed `[n]` citation in a report
`answer` must be a `results[].number`; an unmatched citation is removed from
the persisted answer and returned as a `citation-without-result` diagnostic.
The CLI `autorag report` and `autorag evidence` commands remain available for
terminal maintenance.

## Completion condition

Setup is complete only when the package and stdio executable are installed,
roots are approved, trusted model-free config is written, every datasource has
been probed and configured/skipped lists reported, and dupey is installed or
its absence reported. The host must connect, discover tools, refresh requested
indexes, return acceptable MCP status, and retrieve known source content via
MCP. Any requested watch schedule must use the same config and be verified.
Remove any previously copied Lite search skill from the host's skill directory
so it no longer routes routine search through shell commands.
