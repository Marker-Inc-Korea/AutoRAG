---
title: "Slack"
description: "Connect the local Slack archive through slacrawl — prerequisites, channel selection, what is indexed, and how searches behave."
---

# Slack

This page shows how to connect Slack through the `slacrawl` CLI, which channels to expose, and what AutoRAG indexes and searches. AutoRAG never talks to the Slack API itself and never reads the Slack Desktop cache directly.

## Prerequisites

- The upstream `slacrawl` CLI installed and available on `PATH`, or pointed at by `connector.binaryPath`.
- Slack credentials in slacrawl's own config, or a local Slack Desktop cache that slacrawl can import.
- On macOS, the Full Disk Access the crawler documents, so it can read the desktop cache.
- A Slack team id for `connector.workspace`. slacrawl's read paths return nothing unless a workspace is named explicitly, even when the archive and its FTS index hold matching rows — an unscoped search looks exactly like an empty archive. AutoRAG passes it as `-workspace <id>`.

## Connect

`connector.syncSource` is passed to slacrawl's `sync` as `--source` (for example `wiretap`); set it for the import you want. `connector.indexTimeoutMs` bounds the first import, which can take minutes, while `connector.timeoutMs` bounds interactive reads.

```json
{
  "datasources": {
    "slack-local": {
      "type": "slack",
      "instanceId": "local",
      "description": "Recent work conversations available in this Mac's Slack Desktop cache.",
      "connector": {
        "configPath": "~/.slacrawl/config.toml",
        "syncSource": "wiretap",
        "workspace": "T0123456789",
        "timeoutMs": 120000
      }
    }
  }
}
```

Initialize and refresh the local mirror with slacrawl, then refresh AutoRAG:

```bash
slacrawl init -db ~/.slacrawl/slacrawl.db -workspace local
slacrawl sync --source wiretap

# `init -workspace local` only names the local config; it is not a Slack team
# id. List the real ids the import produced and use one as `workspace`:
slacrawl sql 'select id, count(*) from workspaces join messages on messages.workspace_id = workspaces.id group by 1;'
slacrawl search -workspace T0123456789 <term>   # must print rows before wiring it up

autorag refresh --method datasources
```

`connector.configPath` is passed as `--config`; the sidecar reads the archive that slacrawl's config `db_path` names, resolving relative paths against that file.

## What is indexed

Slack is a local workspace/channel/thread archive plus FTS5, owned by slacrawl. Refresh runs the CLI sync first, then embeds new or edited messages into the AutoRAG sidecar through the shared embedder. Documents are exposed as opaque `/slack/<instance>/chunks/<id>` sources.

## Search behavior

- One method per connection. With a sidecar (the default) it is `slack-hybrid` (BM25 over the read-only sidecar plus semantic, fused with reciprocal rank fusion), falling back to the sidecar's BM25 and only then to the CLI-backed `slack-fts`. Without the sidecar, only `slack-fts` is registered.
- `channels.ids` / `channels.names` filter in the sidecar's SQL, so excluded chats are never searched, embedded, or stored. See [Hybrid search for chat archives](./chat-hybrid-search.md).
- `scope` narrows results; a time window is applied natively in the sidecar's SQL before the `topK` cut. See [Search semantics](./search-semantics.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Search returns nothing with a populated archive | `connector.workspace` is missing or wrong; named workspaces are required. |
| `sync` fails as `datasource-index-failed` | No `connector.syncSource`, so slacrawl's `sync` has no `--source`. |
| First refresh times out | Raise `connector.indexTimeoutMs`. |
| `missing slacrawl binary` from setup | Install slacrawl or set `connector.binaryPath`. |
| Empty result is JSON `null`, not `[]` | slacrawl reports no hits as `null`; treat it as empty, not broken. |

Diagnostics use the datasource codes in [Search semantics](./search-semantics.md); the CLI's stderr reaches them verbatim.
