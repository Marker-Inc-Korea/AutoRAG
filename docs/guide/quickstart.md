---
title: "Quickstart"
description: "Run AutoRAG end to end on a temporary home: initialize, index, retrieve or search, and connect an MCP client."
---

# Quickstart

This page runs AutoRAG from nothing to a first hit, first in model-free (Lite) mode and then with a chat model, on an isolated temporary home so nothing touches your real `~/.autorag`. Every command below is copy-pasteable.

## 1. Isolate a temporary home

AutoRAG reads its config from `--config`, then `AUTORAG_CONFIG`, then `$AUTORAG_HOME/config.json`, then `~/.autorag/config.json`. Point both variables at a temp directory first:

```bash
export QA_ROOT="$(mktemp -d)"
export AUTORAG_HOME="$QA_ROOT/.autorag-home"
export AUTORAG_CONFIG="$AUTORAG_HOME/config.json"
mkdir -p "$AUTORAG_HOME" "$QA_ROOT/workspace" "$QA_ROOT/autorag-demo"
printf '# Refund policy\n\nRefund exceptions require director approval before payout.\n' > "$QA_ROOT/autorag-demo/refund-policy.md"
echo "Configuration will be written under $AUTORAG_HOME"
```

Every command in this shell then writes only inside `$QA_ROOT`. `--workspace` is **not** isolation: it moves only the workspace (the folder that owns `.autorag` index state), not the config path. Without `--workspace` the workspace is the current directory, so always pass it.

## 2. Model-free path (Lite)

Lite needs no chat model. It initializes a config, builds indexes, and retrieves raw chunks that a caller curates.

### Initialize

```bash
autorag lite init --search-paths "$QA_ROOT/autorag-demo" --workspace "$QA_ROOT/workspace"
```

`lite init` writes the model-free config and exits 0. Add `--force` (with an explicit `--config`/`AUTORAG_CONFIG`) to replace an existing config.

### Build the indexes

```bash
autorag refresh --json      # parses sources, then syncs MinSync, datasources, Jikji
autorag status --json       # freshness and per-component health (model-free, path-opaque)
```

`autorag refresh` is model-free and exits 0 on success. A never-refreshed workspace still answers, but with no MinSync or Jikji evidence. `autorag status` reports `stale`, `state`, `diagnostics`, and per-component state for `minsync`, `jikji`, and `datasources`.

### Retrieve

```bash
autorag lite retrieve "what does the refund exception require" --top-k 5 --json
```

A successful run exits 0 and prints a JSON envelope (values illustrative):

```json
{
  "ok": true,
  "query": "what does the refund exception require",
  "stale": false,
  "results": [
    { "number": 1, "source": "/tmp/tmp.XXXX/autorag-demo/refund-policy.md", "method": "minsync-hybrid", "score": 1, "metadata": {}, "content": "Refund exceptions require director approval before payout." }
  ],
  "unsearched": [],
  "diagnostics": []
}
```

`--scope SCOPE` narrows retrieval, `--since`/`--until` accept ISO-8601 and keep only dated evidence in `[since, until)`, and `--refresh` rebuilds the indexes incrementally first. `--strict` fails with exit 2 when the index is stale instead of answering from it.

**Exit codes for `autorag lite retrieve`:** `0` retrieved (results may be empty), `2` empty query, config error, index not ready, or `--strict` with a stale index, `1` runtime error.

## 3. Model-backed path

The full librarian curates an answer with one configured chat model. See [Models and providers](models.md) for provider setup.

### Initialize and pick a model

```bash
autorag init --search-paths "$QA_ROOT/autorag-demo" --workspace "$QA_ROOT/workspace"
```

`autorag init` writes the config but does **not** pick a model. Sign in and choose one interactively:

```bash
autorag tui      # then /login and /model
```

or record a model in the config (`model: { provider, id }`), preserving the other settings. Never re-run `autorag init --force` just to change a model — it replaces the whole config. Find provider/model ids with `autorag models list --available`.

### Verify and search

```bash
autorag health --json                                  # resolves the model, then one live completion probe
autorag refresh --json                                 # build indexes (model-free)
autorag search "What are our primary Q3 deliverables?" --top-k 5 --json
```

`autorag health` exits 0 when the model resolves, the credential is present, and the probe succeeds; 2 for a config or model-resolution failure; 1 for a missing credential, unreachable provider, completion failure, or timeout. `--skip-probes` proves config shape only and never live access.

`autorag search` exits `0` answered, `1` search failed, `2` usage or model configuration, `3` decision model unavailable (`decision-model-unavailable`). The answer is the model's final assistant message with inline citations resolved to numbered results.

## 4. Connect a datasource

Datasources are declared under `datasources` in the trusted config; each gets its own `search_datasource_<id>` tool and MCP search tool. A Datasource must be indexed and actually return hits — indexing alone is not proof.

```json
{
  "datasources": {
    "github": { "connector": { "repos": ["owner/repo"], "tokenEnv": "GITHUB_TOKEN" } },
    "obsidian": { "connector": { "vaultPath": "/path/to/vault" } }
  }
}
```

Probe readiness with `autorag setup --format json`, then `autorag refresh --method datasources --json`. Store only environment-variable **names** such as `tokenEnv` — never credential values. Full per-source setup is in [Connecting datasources](../datasources/connecting.md).

## 5. Register the MCP server

To let a coding agent search model-free, register `autorag-mcp` as a stdio server pointing at the same absolute config path:

```bash
claude mcp add --transport stdio --scope local autorag \
  --env AUTORAG_CONFIG="$AUTORAG_CONFIG" \
  -- "$(command -v autorag-mcp)"
```

The server exposes `autorag.status`, `autorag.refresh`, `autorag.search`, `autorag.search.files`, `autorag.datasources.list`, and `autorag.evidence`, plus one search tool per integrated datasource. `AUTORAG_MCP_READ_ONLY=1` omits the write tools. Registration alone is not proof — reload the host and run a known-source `autorag.search` through it. See [MCP server](../agents/mcp-server.md) and [Agent skills](../agents/skills.md).

## 6. Keep it fresh

A query only reads indexes that a refresh built. For continuous freshness, schedule a refresh tick — hourly by default — with cron, launchd, a systemd user timer, or Task Scheduler:

```bash
autorag watch --once --json
```

A skipped tick (outside its `--window`/`--days`, or on battery with `--power ac`) exits 0 with `skipped=true`. See [Refresh operations](../indexing/refresh-operations.md).
