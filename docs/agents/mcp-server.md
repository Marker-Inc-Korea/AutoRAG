---
title: "MCP server (Lite)"
description: "Run autorag-mcp over stdio or HTTP and call its fixed and per-datasource tools from any agent."
---

# MCP server (Lite)

`autorag-mcp` is the model-free AutoRAG Lite MCP server. It indexes your files and
datasources, then exposes search to any MCP host over stdio or streamable HTTP.
The calling agent's model plans the search and writes the answer; AutoRAG returns
ranked, source-labelled chunks. This page documents both transports, every flag
and environment variable, registration, and the full tool surface.

## Transports

| Transport | Command | Who uses it |
| --- | --- | --- |
| stdio (default) | `autorag-mcp` | Hosts that spawn a child process: Claude Code, Codex, Cursor, Pi, Hermes. |
| Streamable HTTP | `autorag-mcp --http` | Agents that run in a vendor cloud and reach you through a URL. See [Always-on cloud agents](remote-agents.md). |

With stdio, the host owns the process and `stdout` is the protocol channel: a
config that fails to reload only writes to `stderr`. HTTP requires an owner token
on every `/mcp` request (see [Owner token](remote-agents.md#owner-token));
`/healthz` answers `{"ok":true}` unauthenticated.

## Flags

Verified against `node src/mcp/index.ts --help`.

| Flag | Effect |
| --- | --- |
| `--http` | Serve streamable HTTP instead of stdio. |
| `--agent` | Also expose `autorag.ask` (model-backed librarian agent with `mode`: `auto` default, or `local`/`organize`/`extra`/`config`). Also `AUTORAG_MCP_AGENT=1`. |
| `--host HOST` | Bind interface for HTTP (default `127.0.0.1`). Only with `--http`. |
| `--port PORT` | HTTP port (default `8765`; `0` picks a free one). Only with `--http`. |
| `--public-url URL` | HTTPS origin the agent reaches this server at; enables the OAuth endpoints and widens the Host/Origin allowlist. Also `AUTORAG_MCP_PUBLIC_URL`. Only with `--http`. |
| `--allow-write` | Also offer `autorag.refresh`, `autorag.report`, and `autorag.convert` (HTTP is read-only by default). Only with `--http`. |
| `--check URL` | Replay a cloud agent's connect sequence against `URL` (for example your tunnel's `/mcp`) without a host. |
| `--json` | With `--check`, print one JSON document. Only with `--check`. |
| `-h`, `--help` | Print usage. |

Passing `--host`, `--port`, `--public-url`, or `--allow-write` without `--http`
is an error. `--check` exits `1` if any step fails; it needs no token to run its
unauthenticated steps, and with `AUTORAG_MCP_TOKEN` (or the stored token file)
available it also runs a real `tools/list`.

## Environment variables

| Variable | Effect |
| --- | --- |
| `AUTORAG_CONFIG` | Path to a specific `config.json`. |
| `AUTORAG_HOME` | Directory holding `config.json` and `memory.json` (default `~/.autorag`). |
| `AUTORAG_MCP_READ_ONLY` | stdio read-only mode: `1` or `true` omits `autorag.refresh`, `autorag.report`, and `autorag.convert`, and blocks mutating modes (`organize`, `config`) on `autorag.ask`. |
| `AUTORAG_MCP_AGENT` | `1` or `true` exposes `autorag.ask` alongside the Lite tools (same effect as `--agent`). |
| `AUTORAG_MCP_TOOLS` | Comma-separated exact tool allowlist; omitted tools cannot be called (applies to both transports). |
| `AUTORAG_MCP_PUBLIC_URL` | HTTP public origin; same effect as `--public-url`. |
| `AUTORAG_MCP_TOKEN` | HTTP owner token, at least 32 characters. |

## Register with a host

Use absolute paths for the config, roots, workspace, and executable: hosts may
start the server from a different working directory or with a restricted `PATH`.
Inspect an existing registration first and update only an outdated command.

```bash
claude mcp add --transport stdio --scope local autorag \
  --env AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json \
  -- /absolute/path/to/autorag-mcp
```

```bash
codex mcp add autorag \
  --env AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json \
  -- /absolute/path/to/autorag-mcp
```

For another host, use its stdio MCP configuration with the same command, empty
arguments, and the `AUTORAG_CONFIG` environment variable. The host starts the
subprocess; do not run it as a background HTTP service for a stdio host. Reload
or reconnect the host, then discover the actual tools and schemas with MCP
`tools/list`.

## Tool surface

Tools are registered in
[`src/mcp/server.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/mcp/server.ts).
Discover the live set with `tools/list`; do not assume a fixed count.

### Fixed tools

| Tool | Purpose | Input fields | Access |
| --- | --- | --- | --- |
| `autorag.status` | Index freshness and health, without modifying the corpus. | none | read-only |
| `autorag.search` | Search selected configured methods and datasources; selection narrows execution before any backend runs. | `query`, `topK`, `scope`, `since`, `until`, `strict`, `datasourceIds`, `methods`, `local` | read-only |
| `autorag.search.files` | Search file and folder names under configured roots (Everything on Windows, fsearch on macOS/Linux); never reads contents. | `query`, `root`, `matchPath`, `matchCase`, `kind`, `maxResults`, `offset` | read-only |
| `autorag.duplicates` | Scan configured roots with Dupey for exact duplicates and near-duplicate document families; never moves or deletes. | none | read-only |
| `autorag.datasources.list` | List the datasources connected under the server's configuration. | none | read-only |
| `autorag.datasources.get` | Return one configured datasource descriptor without credentials or private metadata. | `datasourceId` | read-only |
| `autorag.convert` | Convert local files to another format offline with Filee; writes new files only and never replaces one. | `inputs`, `to`, `outputDir`, `recursive`, `quality`, `keepDates` | mutating |
| `autorag.convert.formats` | List the formats Filee knows, or what one source format can become. | `from` (optional) | read-only |
| `autorag.report` | Persist an externally curated Lite search report with opaque source mappings. | `query`, `report` | mutating |
| `autorag.evidence` | Return persisted opaque sources and evidence chunks for a curated report session. | `sessionId`, `resultNumber` | read-only |
| `autorag.refresh` | Refresh configured indexes; never modifies source documents. | `force`, `methods` | mutating |
| `autorag.ask` | Run the model-backed librarian agent (opt-in via `--agent` / `AUTORAG_MCP_AGENT=1` or `ask` option). `mode` defaults to `auto` (Jev routing); pass `local`, `organize`, `extra`, or `config` (or aliases `find`/`summarize`/`automate`) to skip Jev routing. | `query`, `mode`, `topK`, `scope`, `since`, `until` | depends on `mode` (`organize`/`config` blocked when read-only) |

`autorag.convert` is also on unless `"filee": { "enabled": false }` in the
config; `autorag.convert.formats` is offered whenever Filee is on. `autorag.report`
and `autorag.refresh` are omitted in read-only mode. `autorag.convert` is omitted
in read-only mode too.

### Per-datasource tools

Every integrated datasource connection gets its own search tool, named from the
connection id: lowercased, with non-alphanumeric characters collapsing to `_` and
edge underscores trimmed, and `search_datasource_` prepended (the server's `autorag`
namespace holds the rest). The `slack` connection exposes
`autorag.search_datasource_slack`; other real examples are
`autorag.search_datasource_kakao` and `autorag.search_datasource_telegram`. Each
takes `query`, `topK`, `scope`, `since`, and `until`. The datasource is
server-configured: the inputs only narrow the search. Discover the ids with
`autorag.datasources.list`. The set changes with configuration, so restart the
server after config changes. The naming rule lives in
[`src/datasource/tool-naming.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/datasource/tool-naming.ts).

### Read-only mode

`AUTORAG_MCP_READ_ONLY=1` (stdio) omits `autorag.refresh`, `autorag.report`, and
`autorag.convert`. HTTP is read-only by default and exposes those three only with
`--allow-write`. A read-only server gets search, evidence, and the read-only
inspection tools. Build indexes with the CLI before connecting a read-only
server.

### Time windows

Every search tool accepts optional ISO-8601 `since` (inclusive) and `until`
(exclusive). A date-only value is local midnight: "all of September 2026" is
`since: "2026-09-01"`, `until: "2026-10-01"`. A malformed value is returned
verbatim as a tool error, not a schema error. Results with no date are excluded
and the method is reported in `unsearched`.

### Evidence and report flow

Search returns ranked results with `source`, `method`, and `content` directly, so
most agents do not need more. The optional curation lifecycle persists a report
and re-opens its evidence:

1. `autorag.report` validates a report (`query` plus a `report` document whose
   citations must match `results[].number`) and returns a `sessionId` and the
   stored results.
2. `autorag.evidence` with that `sessionId` and an optional `resultNumber`
   returns the exact chunks behind numbered results.

An unmatched `[n]` citation is removed from the persisted answer and returned as a
`citation-without-result` diagnostic. The CLI equivalents are `autorag lite report` and
`autorag evidence`.

### Tool error codes

Handlers return `isError: true` with an `errorCode` for the model to act on:
`index-not-ready`, `stale-index`, `invalid-selection`, `search-failed`,
`datasource-not-found`, `duplicates-disabled`, `duplicates-failed`,
`convert-failed`, `invalid-report`, `report-failed`, `evidence-not-found`, and
`refresh-failed`. `index-not-ready` means refresh has not completed; run
`autorag.refresh` first.

## Lite CLI fallback

The CLI is the bootstrap and repair interface when the MCP server is unavailable
or the deployment is read-only.

| Command | Purpose | Key flags |
| --- | --- | --- |
| `autorag lite init` | Write the model-free lifecycle config. | `--search-paths`, `--exclude`, `--workspace`, `--memory-path`, `--force` |
| `autorag lite refresh` | Incremental index refresh, no model required. | `--full`, `--force`, `--method` |
| `autorag lite watch` | Watch configured roots, or one tick. | `--once`, `--immediate`, `--debounce-ms`, `--force` |
| `autorag lite status` | Path-opaque corpus freshness and index health. | `--json` |
| `autorag lite index` | Reset or rebuild parsed/MinSync indexes. | `reset\|rebuild`, `--yes`, `--method` |
| `autorag lite duplicates` | Scan duplicate document families read-only. | `--json` |
| `autorag lite health` | Alias of `lite status`. | `--json` |
| `autorag lite retrieve <query>` | Retrieve documents without model curation. | `--top-k`, `--scope`, `--since`, `--until`, `--refresh`, `--strict`, `--json`, `--debug` |
| `autorag lite report <query>` | Persist a structured report from external curation. | `--input`, `--json`, `--debug` |
| `autorag lite find <query>` | Whole search without a model; streams NDJSON one group per surface. | `--top-k`, `--groups`, `--json`, `--debug` |

The `--method` value accepts `parsed`, `minsync`, `datasources`, `jikji`,
`everything`, `fsearch`, and `all`; MCP `methods` accepts the same individual
values but not `all`. For setup and index-building steps, see
[Installation](../guide/installation.md) and
[Scheduling and exclusions](../indexing/scheduling-and-exclusions.md).
