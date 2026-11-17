---
title: "GitHub"
description: "Index GitHub issues and pull requests for configured repositories through the REST API and search them from AutoRAG."
---

# GitHub

The `github` datasource fetches issues and pull requests for a trusted list of `owner/repo` targets and indexes them for lexical search. Use it to answer questions about bug reports, feature requests, and review discussion in the repositories you configure.

## Prerequisites

- A GitHub token in an environment variable. Public repositories work unauthenticated, but `autorag setup` reports `github` as `skipped: credential GITHUB_TOKEN is unavailable` unless that variable is set, so export it before expecting the connection to probe ready.
- At least one `owner/repo` target. Without `connector.repos` the connector returns `not-configured` and indexes nothing.

::: tip
The token lives only in the environment. `connector.tokenEnv` names the variable; its value is never written to `~/.autorag/config.json`.
:::

## Connect

Add a `datasources` entry to `~/.autorag/config.json`:

```json
{
  "datasources": {
    "github": {
      "connector": {
        "repos": ["Marker-Inc-Korea/AutoRAG"],
        "tokenEnv": "GITHUB_TOKEN"
      }
    }
  }
}
```

| Connector option | Default | Purpose |
| --- | --- | --- |
| `repos` | none (required) | `owner/repo` targets to index. |
| `tokenEnv` | `GITHUB_TOKEN` | Name of the environment variable holding the token. |
| `token` | unset | Literal token; an explicit `token` wins over `tokenEnv`. For tests only. |
| `maxPages` | `10` | Issue pages per repository (100 items each). |
| `maxDocuments` | `500` | Cap on documents indexed across all repositories. |
| `timeoutMs` | HTTP default | Per-request timeout. |
| `baseUrl` | `https://api.github.com` | REST base; override only to point at a mock server. |

Index several repositories under separate identities with a connection alias:

```json
{
  "datasources": {
    "work-github": {
      "type": "github",
      "connector": { "repos": ["acme/api", "acme/web"] }
    }
  }
}
```

Then index it:

```bash
autorag refresh --method datasources
```

## What is indexed

- One document per issue and pull request, open and closed, from `/repos/<owner>/<repo>/issues?state=all`. That endpoint also returns pull requests, so the result metadata distinguishes them.
- Document title is `#<number> <title>`; content is the title plus the body.
- Metadata carries `state`, `number`, `kind` (`issue` or `pull`), and `labels`.
- The document's time is GitHub's `updated_at`; it drives `since`/`until` windows.
- Chunk identity is the opaque source `/github/<instance>/chunks/<chunk-id>`, where the chunk id is the sanitized `<owner>-<repo>-<number>`. `describeSources` also lists hierarchy paths such as `/github/<instance>/<owner>/<repo>/issues`.
- The chunk store persists to `<workspace>/.autorag/datasources/github/<instance>/chunks.json`; without a workspace it is in-memory only.

## Search behavior

- Retrieval is lexical BM25 over the chunk store (method `github-lexical`). GitHub has no semantic method.
- A `scope` narrows within the connection, for example `/github/**` or `/github/work-github` (see [Scope narrowing](search-semantics.md#scope-narrowing)). The chunk source does not carry the repository, so per-repository narrowing is not expressible as a scope; each result's repository is in its `hierarchy` metadata.
- Default poll cadence is 15 minutes while auto-refresh runs.
- The agent may also inspect repositories directly with the `gh` CLI through `bash`, for example `gh search issues "<query>" --repo <owner>/<repo>`. The dedicated `search_datasource_github` tool remains the indexed path.

## Troubleshooting

| Symptom | Diagnostic | Fix |
| --- | --- | --- |
| Nothing indexed | `datasource-unavailable` (`not-configured: no repositories configured`) | Set `connector.repos`. |
| `github: unauthorized` | `datasource-auth-error` | The token is missing or invalid; the whole fetch fails. |
| `github: rate limited or forbidden` | `datasource-rate-limited` | HTTP 403/429, including primary rate limits; the whole fetch fails. |
| One repository failed | warning `datasource-index-failed` | A per-repo 404 or transient error degrades to a warning; other repositories still index. |
| Fetch returned no documents | info `datasource-empty` | The repositories have no issues, or the token cannot see them. |
| `autorag setup` reports `skipped: credential GITHUB_TOKEN is unavailable` | — | Export the variable named by `connector.tokenEnv`. |
