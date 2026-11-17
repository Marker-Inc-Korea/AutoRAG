---
title: "Coding-agent sessions (wombat)"
description: "Search past coding-agent conversations from twenty CLI agents, indexed by wombat and MinSync, with one shared embedding."
---

# Coding-agent sessions (wombat)

The `wombat` datasource searches your past conversations with coding agents: what was asked, decided, or built in earlier sessions, in any project. The external `wombat` CLI parses each agent's own log format, keeps only the user and assistant conversation, and lets MinSync index it incrementally with SPLADE and dense vectors. AutoRAG never opens the agents' logs or wombat's catalog directly.

## Prerequisites

- The `wombat` CLI installed. It lands in `~/.cargo/bin`, which a desktop app's `PATH` often omits; setup finds it there anyway.

```bash
cargo install wombat-sessions
```

- At least one supported agent with session logs on this machine. wombat parses: `codex`, `claude-code`, `opencode`, `pi`, `omp`, `hermes`, `aside`, `kiro`, `droid`, `goose`, `vibe`, `aider`, `codebuddy`, `iflow`, `continue`, `copilot`, `crush`, `qwen`, `kilo`, `gemini`.
- MinSync, resolved exactly as for the local-file index (see the shared setup below).

## Connect

Add a `datasources` entry to `~/.autorag/config.json`:

```json
{
  "datasources": {
    "wombat": {
      "connector": {
        "binaryPath": "wombat",
        "agents": ["codex", "claude-code", "opencode", "pi", "omp"]
      }
    }
  }
}
```

| Connector option | Purpose |
| --- | --- |
| `binaryPath` | The `wombat` executable, when it is not on `PATH`. |
| `agents` | Restrict indexing and search to these agents (an unknown name is a config error, not a silently empty datasource). |
| `sources` | Pin session roots, `{"claude-code": ["/path"]}`, passed as `--source <agent>=<root>`; otherwise wombat reads every agent's default location under the home directory. |
| `includeSubagents` | Also index subagent / sidechain transcripts (skipped by default). |
| `includeEphemeral` | Also index headless, benchmark, and temp-directory sessions (skipped by default). |
| `dataDir` | Move wombat's data directory (default `~/.wombat`); holds the mirror, the MinSync index, and the session catalog. |
| `configPath` | A wombat `config.toml`. |
| `indexTimeoutMs` | `sync --index` budget (default one hour); search keeps its 60-second default. |

Then index it:

```bash
autorag refresh --method datasources
```

## What is indexed

- Refresh runs `wombat doctor --json`, then `wombat sync --index --json`: parse new and changed sessions into wombat's mirror, then let MinSync index it incrementally. Only changed sessions are parsed and only new chunks are embedded; an unchanged store embeds nothing.
- Only the user and assistant conversation is indexed. Thinking blocks, tool calls and outputs, and file diffs are not; search for what was said, not for code that was run.
- Sources are `/wombat/<instance>/chunks/<chunk-id>`. Each hit's first line is a label, `[<sessionKey> | turns A-B | date | title | project]`, with missing parts omitted. `sessionKey` is `<agent>:<session id>`.

## Search behavior

- Retrieval is hybrid-first: one `wombat search --mode hybrid` per query, falling back to SPLADE keyword search when the dense side cannot run (for example, before an embedder exists).
- A scope narrows within the connection, for example `/wombat/**` (see [Scope narrowing](search-semantics.md#scope-narrowing)). When `connector.agents` is set, an agent allow-list is applied after the CLI answers.
- A `since`/`until` window uses the session's `startedAt` (see [Time windows](search-semantics.md#time-windows)).
- To go from a result to the full conversation, the agent calls `wombat` through `bash` with the `sessionKey` from the label: `wombat show <sessionKey>` prints the cleaned conversation; `wombat trace <sessionKey> --turn N --context K` prints the original log around that turn, including the tool calls and outputs the index leaves out.

## One MinSync and one embedding

wombat is built on MinSync, so it shares AutoRAG's MinSync instead of bringing its own:

- **Binary.** Refresh resolves `minsync` exactly as the local-file index does and hands it to wombat with `--minsync-bin`; wombat never runs a second or older copy. A query only looks for the binary and never installs; a refresh that cannot resolve MinSync fails as `datasource-unavailable` with the resolver's message.
- **Model and daemon.** The default is MinSync's local model in the same cache as everything else. A running `minsync serve` daemon is adopted automatically, so the model is loaded once. A sparse-only or `native:` `minSync.embedder` id is passed to wombat as `--embedder` when it first creates its workspace; a remote `minSync.embedder` is never handed over, because conversations are private.
- **Query instruction.** After a sync, AutoRAG sets `query_prefix` in wombat's `.minsync/config.toml` to the model's query instruction when that key is absent. MinSync prepends it to queries only, so no passage is re-embedded. An existing value, another model, a remote `base_url`, or a sparse-only workspace is left alone.
- **No second copy in the local index.** wombat's mirror (`<data dir>/mirror`, default `~/.wombat/mirror`) is added to `excludePaths` automatically, so a `searchPaths` entry that covers the home directory does not parse and embed every conversation again as a file.

::: warning **Privacy:** AutoRAG refuses to spawn the CLI when `connector.embedder` or `WOMBAT_EMBEDDER` selects a remote (`openai:` or `tei:`) embedder, so session text never leaves the machine for embeddings.
:::

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| `unknown wombat agent "<name>"` | thrown at config load | Use a name from the supported list. |
| `wombat binary-missing` / `spawn-error` / `timeout` | `datasource-unavailable` | Install wombat; check `connector.binaryPath`. |
| `wombat minsync-unavailable` | `datasource-unavailable` | A refresh could not resolve MinSync; see [Embedding backends](../indexing/embedding-backends.md). |
| `remote-embedding-rejected` | `datasource-embedding-egress-rejected` | Remove the remote embedder; wombat keeps its local default. |
| `wombat could not parse or index N session(s)` | warning `datasource-index-partial` | The next refresh retries them. |
| `wombat found no coding-agent sessions to index` | info `datasource-empty` | No session logs were discovered; check `agents`/`sources`. |
| `autorag setup` reports `skipped: not configured` | — | Add the `datasources.wombat` entry; setup never writes it for you. |
