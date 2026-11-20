---
title: "GitHub Gists"
description: "Index your GitHub Gists with an incremental cursor and search them lexically or semantically from AutoRAG."
---

# GitHub Gists

The `github-gist` datasource mirrors the authenticated account's own Gists — public and secret ones the token permits — into a local index and searches them. Use it for code snippets, notes, and design memos captured as gists.

## Prerequisites

- A GitHub token reachable in one of two ways: an environment variable (`connector.tokenEnv`, default `GITHUB_TOKEN`) or the `gh` CLI login (`gh auth token`), which is the fallback. Set `connector.ghCliFallback: false` to disable the fallback.
- No repository list is required: the datasource indexes the account the token belongs to.

## Connect

Add a `datasources` entry to `~/.autorag/config.json`:

```json
{
  "datasources": {
    "github-gist": {
      "connector": { "tokenEnv": "GITHUB_TOKEN" }
    }
  }
}
```

| Connector option | Default | Purpose |
| --- | --- | --- |
| `tokenEnv` | `GITHUB_TOKEN` | Name of the environment variable holding the token. |
| `token` | unset | Literal token; wins over `tokenEnv`. For tests only. |
| `ghCliFallback` | `true` | Fall back to `gh auth token` when no token is in the environment. |
| `maxPages` | `10` | Gist-list pages (100 gists each). |
| `maxGists` | `500` | Cap on gists tracked. |
| `statePath` | `<workspace>/.autorag/datasources/github-gist/<instance>/state.json` | Incremental-cursor file. Without a workspace, every fetch is a full sync. |
| `baseUrl` | `https://api.github.com` | REST base; override only for a mock server. |

Semantic retrieval is a skill option, written under `connector` and taken out before the connector sees the rest. `connector.semantic: false` (or `{ "enabled": false }`) keeps the datasource lexical-only: no embedding calls, no vector index, no `<name>-semantic` method.

```json
{
  "datasources": {
    "github-gist": {
      "connector": { "tokenEnv": "GITHUB_TOKEN", "semantic": false }
    }
  }
}
```

Then index it:

```bash
autorag refresh --method datasources
```

## What is indexed

- The authenticated account's gists. Visibility is bounded by the token's scopes; secret gists are included when the token allows.
- Refresh is incremental: the gist-list endpoint returns `id` and `updated_at` cheaply, and a persisted cursor decides which gists need a full content fetch and which disappeared. Gists whose full fetch failed stay out of the cursor so the next sync retries them.
- Document content is the description plus each file as `## <filename>` followed by its text; metadata carries `gistId`, `public`, `files`, and `htmlUrl`. The time is `updated_at`.
- Sources are `/github-gist/<instance>/chunks/<chunk-id>`; the chunk id is the sanitized gist id (with a `-p<n>` suffix when a gist splits into several chunks).
- Semantic vectors live beside the cursor in `<workspace>/.autorag/datasources/github-gist/<instance>/vectors.json`.

## Search behavior

- Lexical BM25 (`github-gist-lexical`) and, when an embedder is available, semantic retrieval (`github-gist-semantic`) through AutoRAG's shared embedder. Gist content stays on the machine for local embeddings.
- The vector sidecar is keyed by chunk id with a content hash: unchanged chunks are never re-embedded, removed chunks are pruned, and an embedding-identity change (provider, model, or dimension) forces a full re-embed. A changed gist's previous vector is excluded from semantic results until re-embedding succeeds; the next refresh retries pending embeddings even when the gist content is unchanged.
- A `scope` narrows within the connection, for example `/github-gist/**` (see [Scope narrowing](search-semantics.md#scope-narrowing)).

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| `no GitHub token: set GITHUB_TOKEN … or authenticate the gh CLI` | `datasource-unavailable` (`not-configured`) | Export the token variable or run `gh auth login`. |
| `github: unauthorized` | `datasource-auth-error` | The token is invalid or lacks access. |
| `github: rate limited or forbidden` | `datasource-rate-limited` | HTTP 403/429. |
| `semantic-unavailable: …` | warning `datasource-index-failed` | The embedder failed; indexing and lexical search continue. Start `inl serve --idle-timeout 0` or configure `minSync.embedder` (see [Embedding backends](../indexing/embedding-backends.md)). |
| Semantic results miss a just-edited gist | — | The previous vector is excluded until the next refresh re-embeds it. |
