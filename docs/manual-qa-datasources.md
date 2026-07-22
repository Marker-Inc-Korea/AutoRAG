# Manual QA — Connector-Backed Datasource Skills

Covers issues #1300 (Slack), #1301 (Google Drive), #1302 (Notion), #1303
(GitHub Issues/PRs), #1304 (Gmail), #1305 (Discord), #1311 (local mail
export), #1314 (Obsidian vault), #1316 (RSS/news).

## Harnesses

| Harness | Target systems | Command |
|---|---|---|
| `scripts/manual-qa/run-qa.ts` | Protocol-accurate local mocks of Slack/Discord/Notion/GitHub/Drive/Gmail APIs + real filesystem fixtures (Obsidian vault, mbox/eml exports) + local RSS feed | `bun scripts/manual-qa/run-qa.ts` |
| `scripts/manual-qa/run-qa-live.ts` | Real public GitHub REST API (this repo's issues) and a real RSS feed (hnrss.org), credential-free | `bun scripts/manual-qa/run-qa-live.ts` |

Skills that need tenant credentials (Slack bot token, Discord bot token,
Notion integration token, Drive/Gmail OAuth tokens) are QA'd against the
mock services, which reproduce each API's envelope shapes and native auth
failures (`invalid_auth`, HTTP 401/403/429). To QA against a real tenant,
point `connector.baseUrl` at the real API base and supply the token via the
default env var (`SLACK_BOT_TOKEN`, `DISCORD_BOT_TOKEN`, `NOTION_TOKEN`,
`GITHUB_TOKEN`, `GDRIVE_ACCESS_TOKEN`, `GMAIL_ACCESS_TOKEN`).

## Checklist (all automated by the harnesses)

### Setup & wiring
- [x] `buildDatasourceSkills` materializes every configured skill from the
      trusted `datasources` config section; unknown names are rejected at
      `buildAgentOptions` with a `ConfigError`.
- [x] Skills register their retrieval methods through the existing
      `RetrievalMethodRegistry` pipeline on agent construction.

### Indexing
- [x] `agent.refresh(true, { methods: ["datasources"] })` indexes all nine
      skills; each returns an ok result with a chunk count.
- [x] Chunk stores persist under `<workspace>/.autorag/datasources/<skill>/<instance>/`
      and reload lazily in fresh agent processes.
- [x] Polling metadata (`mode`, `intervalMs`, `lastIndexedAt`, `lastPolledAt`,
      `lastError`) tracks success and failure; RSS applies a dedupe window.

### Progressive disclosure & search
- [x] Authorized skills appear as `datasource-<name>` in the system prompt;
      unauthorized skills are omitted entirely.
- [x] `load_datasource_skill` returns full path-opaque instructions for
      authorized names and not-available for denied/unknown names.
- [x] `search_datasource_documents` returns hits for each skill with opaque
      slash-hierarchical sources (`/<skill>/<instance>/chunks/<id>`); no `#`
      fragments, no real filesystem paths.
- [x] `scope` narrows results (e.g. `/gmail/**` excludes Slack hits) and can
      never widen access.

### Security
- [x] Default-deny: without trusted `allowedTags`, searches return nothing
      and skills are absent from the prompt.
- [x] Tool arguments carrying `allowedTags`/`allowedScopes` are ignored —
      they cannot grant permissions.
- [x] User scope intersects trusted scopes before merge
      (`DatasourceResultFilter`).

### Diagnostics
- [x] Wrong tokens map to `datasource-auth-error`; permission problems to
      `datasource-permission-denied`; HTTP 429 to `datasource-rate-limited`;
      unreachable services to `datasource-unavailable`.
- [x] Failure payloads and warnings never contain tokens, URLs, e-mail
      addresses, or absolute filesystem paths (`sanitizeOpaqueText`).
- [x] Per-item failures (a denied Slack channel, a 404 repo, one bad feed,
      an unparseable mbox message) degrade to warnings without failing the
      whole index run.

## Last run

- `run-qa.ts`: 27/27 checks passed (mock APIs + real filesystem fixtures).
- `run-qa-live.ts`: 4/4 checks passed (live GitHub REST, live hnrss.org feed).
