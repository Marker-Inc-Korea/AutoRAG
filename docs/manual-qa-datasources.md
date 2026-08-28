# Manual QA — Datasource Skills

Covers issues #1300 (Slack), #1302 (Notion), #1303
(GitHub Issues/PRs), #1304 (Gmail), #1311 (local mail
export), #1314 (Obsidian vault), #1316 (RSS/news), #1350 (macOS Spotlight).
Issue #1416 adds external-crawler process coverage for WhatsApp through
wacrawl, Telegram through telecrawl, Slack through slacrawl, and Notion
through notcrawl.
Issue #1477 adds the live ClawGallery CLI path.

## Harnesses

| Harness | Target systems | Command |
|---|---|---|
| `scripts/manual-qa/run-qa.ts` | Protocol-accurate local mocks of GitHub/Gmail APIs + real filesystem fixtures (Obsidian vault, mbox/eml exports) + local RSS feed | `bun scripts/manual-qa/run-qa.ts` |
| `scripts/manual-qa/run-qa-discrawl-live.ts` | Real Discord archive through the external `discrawl` CLI (FTS + semantic + hybrid, incremental re-sync) | `bun scripts/manual-qa/run-qa-discrawl-live.ts` |
| `scripts/manual-qa/run-qa-clawgallery-live.ts` | Real ClawGallery CLI plus a local image folder (incremental bootstrap + hybrid search) | `bun scripts/manual-qa/run-qa-clawgallery-live.ts /path/to/images "query"` |
| `scripts/manual-qa/run-qa-live.ts` | Real public GitHub REST API (this repo's issues) and a real RSS feed (hnrss.org), credential-free | `bun scripts/manual-qa/run-qa-live.ts` |
| `scripts/manual-qa/run-qa-spotlight-live.ts` | Real macOS Spotlight (`mdfind`/`mdimport`) end-to-end; macOS only, no credentials | `bun scripts/manual-qa/run-qa-spotlight-live.ts` |
| `scripts/manual-qa/run-qa-rclone.ts` | Deterministic `cloud-drive`/rclone process seam covering initial/no-op/update/delete/rename/interrupted recovery and scoped search | `bun scripts/manual-qa/run-qa-rclone.ts` |
| `scripts/manual-qa/run-qa-datasource-aliases.ts` | Universal alias registration plus all-channel and channel-allowlisted chat retrieval | `bun scripts/manual-qa/run-qa-datasource-aliases.ts` |
| `test/datasource/skills/wacrawl.test.ts` | Real child-process boundary with a deterministic fake wacrawl executable: argv, JSON parsing, env isolation, missing binary, malformed output, indexing, retrieval | `bunx vitest run test/datasource/skills/wacrawl.test.ts` |
| `test/datasource/skills/telecrawl.test.ts` | Real child-process boundary with a deterministic fake telecrawl executable: argv, JSON parsing, env isolation, missing binary, malformed output, indexing, retrieval | `bunx vitest run test/datasource/skills/telecrawl.test.ts` |
| `test/datasource/skills/slacrawl.test.ts` | Real child-process boundary with a deterministic fake slacrawl executable: argv, JSON parsing, env isolation, missing binary, malformed output, indexing, retrieval | `bunx vitest run test/datasource/skills/slacrawl.test.ts` |
| `test/datasource/skills/notcrawl.test.ts` | Real child-process boundary with a deterministic fake notcrawl executable: argv, JSON parsing, env isolation, missing binary, malformed output, indexing, retrieval | `bunx vitest run test/datasource/skills/notcrawl.test.ts` |

Skills that need tenant credentials (Gmail OAuth tokens) are QA'd against the
mock services, which reproduce each API's envelope shapes and native auth
failures (`invalid_auth`, HTTP 401/403/429). Google Drive is tested through the
provider-neutral `cloud-drive` skill and the external `rclone` CLI; configure
and authenticate the remote with `rclone config`, then run
`bun scripts/manual-qa/run-qa-rclone.ts`. To QA Gmail against a real tenant,
point `connector.baseUrl` at the real API base and supply the token via the
default env var (`GITHUB_TOKEN`, `GMAIL_ACCESS_TOKEN`).

WhatsApp uses the external wacrawl CLI instead of an HTTP mock. The deterministic
test executable exercises the actual spawn/stdout/stderr contract without
requiring access to a private WhatsApp archive. For a live macOS check, install
wacrawl, grant its documented Full Disk Access, configure
`datasources.whatsapp`, then run `autorag refresh --method datasources`.

Telegram uses the external telecrawl CLI instead of an HTTP mock. Its
deterministic executable covers the same process boundary without private
Telegram data. For a live macOS check, install telecrawl, grant its documented
Full Disk Access, configure `datasources.telegram`, then run
`autorag refresh --method datasources`.

Slack uses the external slacrawl CLI instead of AutoRAG's former Web API
connector. Its deterministic executable verifies the sync/search argv,
structured results, update-check suppression, environment isolation, and
diagnostic mapping. Configure Slack credentials in slacrawl itself, then set
`datasources.slack.connector.configPath` and optional `syncSource`.

Notion uses the external notcrawl CLI instead of AutoRAG's former Notion API
connector. Its deterministic executable verifies sync/search argv, structured
page results, update-check suppression, environment isolation, and diagnostic
mapping. Configure credentials in notcrawl itself, then set
`datasources.notion.connector.configPath`.

## Checklist (all automated by the harnesses)

### Setup & wiring
- [x] `buildDatasourceSkills` materializes every configured skill from the
      trusted `datasources` config section; unknown names are skipped at
      `buildAgentOptions` with an `unknown-datasource-skill` diagnostic so
      unrelated search/status commands still run.
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

### Rclone cloud-drive checklist

- [x] `rclone lsjson --recursive --files-only --hash` is the only inventory
      seed; provider credentials remain in rclone's own config.
- [x] The workspace manifest stores stable virtual path, size, modtime,
      hashes, and provider id when available.
- [x] Initial sync copies indexable files; a no-op sync copies zero bodies and
      does not rewrite the chunk store.
- [x] Updating one file copies/reindexes only that file; delete and rename
      remove stale mirror entries.
- [x] A failed copy leaves the previous completed snapshot searchable.
- [x] Google Drive is Tier-1; OneDrive and other remotes are provider-neutral;
      iCloud Drive is documented as experimental/Tier 4.
- [x] Include/exclude, maximum size, concurrency, bandwidth limit, and
      dry-run are trusted CLI datasource configuration.

## Last run

- `run-qa.ts`: 27/27 checks passed (mock APIs + real filesystem fixtures).
- `run-qa-live.ts`: 4/4 checks passed (live GitHub REST, live hnrss.org feed).
