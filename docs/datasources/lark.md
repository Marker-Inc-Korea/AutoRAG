---
title: "Lark / Feishu"
description: "Search a signed-in Lark or Feishu tenant through lark-cli — auth scopes, message and document search, and coverage limits."
---

# Lark / Feishu

This page shows how to search a Lark or Feishu tenant in place through the official `lark-cli`. Lark (larksuite.com) and Feishu (feishu.cn) are the same product on separate hosts. There is no local archive: the tenant keeps the index, AutoRAG stores nothing, and credentials stay in the CLI keychain.

## Prerequisites

- `lark-cli` installed (`npm install -g @larksuite/cli`) and available on `PATH`, or pointed at by `connector.binaryPath`.
- Login with explicit scopes, then confirm:

```bash
lark-cli auth login --scope "search:message,search:docs:read"
lark-cli auth status
```

`auth login --recommend` does not grant those scopes by itself. AutoRAG never reads the desktop client's private databases and never accepts an app secret.

## Connect

```json
{
  "datasources": {
    "lark": {
      "type": "lark",
      "instanceId": "default",
      "channels": { "ids": ["oc_xxx"] }
    }
  }
}
```

| Key | Meaning |
|---|---|
| `connector.binaryPath` | Explicit lark-cli path; default is a bare `lark-cli` lookup. |
| `channels.ids` | Chat ids applied to message search with `--chat-id`. It does not hide documents. |
| `instanceId` | Instance id; must be a safe single path segment. |

`connector.embedderBaseUrl` exists only to reject remote embedding egress: if it names a remote endpoint, `index()` fails with `datasource-embedding-egress-rejected`.

## What is indexed

Nothing. Lark is a remote-search datasource: `index()` probes `lark-cli --version` and `lark-cli auth status --json` and reports `chunkCount: 0`. Retrieval reads the tenant live on each query, so a hit is a live pointer and later edits or recalls change the text.

## Search behavior

- Two retrieval methods: one over messages and one over documents. A `scope` under `/lark/<instance>/messages` selects chats; a scope under `/lark/<instance>/docs` selects documents.
- Sources are `/lark/<instance>/messages/<message_id>` and `/lark/<instance>/docs/<token>`.
- Server ranking is not BM25, vector, or hybrid. Coverage of older messages, retention, and chat types is whatever the tenant indexed for this token, and is not guaranteed; a hit carries `coverage: "server-side-unverified"`.
- Document search rejects a query longer than 30 Unicode characters.
- Rate-limit responses (error code `99991400`) are retried with exponential backoff, up to 7 attempts.

To read a hit in full, use the CLI on the ids from the result, not `bash` on the `/lark/...` identity:

```bash
lark-cli im +messages-mget --message-ids <message_id> --format json
lark-cli docs +fetch --doc <token> --doc-format markdown
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `datasource-auth-error` | `lark-cli auth status` fails; log in with the two scopes above. |
| `datasource-unavailable` | `lark-cli --version` fails or the binary is missing. |
| Search fails after a brief wait | The tenant rate-limited search; AutoRAG retries with backoff (up to 7 attempts), then surfaces the CLI failure if still limited. |
| `datasource-embedding-egress-rejected` | A remote embedding endpoint was configured for Lark. |
| A message search misses old content | Server-side coverage is not guaranteed; this is a limit, not an error. |

Diagnostics use the datasource codes in [Search semantics](./search-semantics.md).
