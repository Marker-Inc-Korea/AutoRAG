---
title: "KakaoTalk"
description: "Connect KakaoTalk on macOS through lazykatok — prerequisites, sync, sidecar hybrid search, and the platform restriction."
---

# KakaoTalk

This page shows how to connect KakaoTalk through the external `lazykatok` CLI. AutoRAG never reads KakaoTalk databases directly, and KakaoTalk is macOS-only.

## Prerequisites

- macOS. On Linux and Windows the skill is not built: a `kakao` entry is reported as unknown by the agent, `autorag setup` lists it as `skipped` (`not supported on <platform>`), and setup never offers it unprompted.
- The upstream `lazykatok` CLI installed and available on `PATH`, or pointed at by `connector.binaryPath`.
- KakaoTalk access already granted to `lazykatok` outside AutoRAG. The default store is `~/Library/Application Support/katok`.
- `lazykatok` 0.3.3 or newer to embed through the shared backend. An archive built with the old native model rebuilds on the next `index`.

## Connect

Refresh runs `lazykatok doctor`, then `sync`, then `index` (or, when the sidecar is enabled, `doctor` and `sync` only, with the sidecar embedding).

```json
{
  "datasources": {
    "family-kakao": {
      "type": "kakao",
      "instanceId": "personal",
      "description": "Family group chats and a few work rooms.",
      "channels": { "names": ["Family"] },
      "connector": { "binaryPath": "lazykatok" }
    }
  }
}
```

| Connector key | Meaning |
|---|---|
| `binaryPath` | Explicit lazykatok path; default is a bare `lazykatok` lookup. |
| `workspacePath` | Native data directory passed as `--data-dir`. |
| `configPath` | Operator-owned config/workspace transport. |
| `source` | Adapter passed to `sync --source`. AutoRAG names the live adapter (`macos`) on macOS; a bare `sync` would otherwise fall back to the CLI's `fixture` default and fail. |
| `timeoutMs` | Interactive budget (`doctor`, `search`, chunk lookups); default 60 s. |
| `indexTimeoutMs` | `sync`/`index` budget; default one hour. A first semantic index over a large archive was measured at ~11 minutes. |
| `channels.ids` / `channels.names` | Chat allowlist; absent means every chat. |

Remote embedding egress configuration is rejected before spawning `lazykatok`.

## What is indexed

`lazykatok` owns the archive, the lexical index, and the message vectors. Results map to opaque `/kakao/<instance>/chunks/<chunk-id>` sources. AutoRAG never runs the native fastembed pipeline when the sidecar is enabled; the sidecar embeds through the shared embedder instead.

## Search behavior

- With the sidecar (the default), the connection registers one method: `kakao-hybrid`, falling back to the sidecar's BM25 and then the CLI-backed BM25. Without the sidecar, the CLI's hybrid method is used with BM25 as fallback. See [Hybrid search for chat archives](./chat-hybrid-search.md).
- Chat/channel filtering is owned by lazykatok; the skill does not expose per-source `scope` narrowing. A hit's chat identity stays in its metadata.
- A time window is applied over the over-fetch (`publishedAt` from the chunk's `startedAt`). See [Search semantics](./search-semantics.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Setup lists `skipped` "not supported on linux/win32" | KakaoTalk is macOS-only. |
| `missing lazykatok binary` from setup | Install lazykatok or set `connector.binaryPath`. |
| `sync` fails | `connector.source` is unset on a non-macOS adapter or the adapter's input is missing. |
| `datasource-embedding-egress-rejected` | A remote embedding endpoint was configured for a path that must stay local. |
| First `index` times out | Raise `connector.indexTimeoutMs`; a timed-out build restarts from scratch. |

Diagnostics use the datasource codes in [Search semantics](./search-semantics.md); lazykatok's stdout, stderr, and error text reach them verbatim.
