---
title: "Adding a datasource"
description: "The verified checklist for adding a new datasource skill to AutoRAG, from the skill directory to tests, docs, and manual QA."
---

# Adding a datasource

This page is the procedure for adding a built-in datasource skill to AutoRAG 2.0. Follow it in order; each step names the file it touches and the check that proves it.

## Prerequisites

- Read [Architecture](architecture.md) and the datasource contract in [`src/datasource/types.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/datasource/types.ts).
- Choose the family your backend belongs to:
  - Connector-backed (`ConnectorDatasourceSkill`, [`src/datasource/connector-skill.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/datasource/connector-skill.ts)): AutoRAG fetches documents and owns the chunk store — see `github` and `rss`.
  - Crawler-backed (`CrawlerDatasourceSkill`, [`src/datasource/crawler-skill.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/datasource/crawler-skill.ts)): an external crawler CLI owns the archive — see `notion`.
  - A custom `DatasourceSkill` when neither base fits — see `wombat`, `hister`, `clawgallery`, and `obsidian`.
- Keep the skill constructible only from trusted config. Model and tool arguments must never reach it, and a query `scope` may only narrow, never widen.

## 1. Create the skill directory

Put the implementation under `src/datasource/skills/<name>/`, with `index.ts` re-exporting the skill, its options, and any helpers. Split modules the way the family you chose does:

| File | Role |
| --- | --- |
| `skill.ts` | The `DatasourceSkill` (or a subclass of a base skill). |
| `connector.ts` | A `DatasourceConnector` for a connector-backed skill. |
| `client.ts` | A process/HTTP client for a CLI- or server-backed skill. |
| `methods.ts` | `RetrievalMethod` implementations when a base skill does not provide them. |
| `index.ts` | The package surface: skill, options types, and any test seams. |

Keep one module focused: the connector/client must never throw, and returns an `ok: false` union with a coarse reason instead.

## 2. Implement the contract

`DatasourceSkill` (in [`src/datasource/types.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/datasource/types.ts)) requires `describe()`, `polling()`, `index()`, `retrievalMethods()`, `describeSources()`, and `skillManifest()`. Hold to these rules:

- `index()` never throws; report failures as `DatasourceDiagnostic` entries whose `code` is one of `DatasourceDiagnosticCode` (for example `datasource-auth-error`, `datasource-unavailable`, `datasource-empty`).
- Descriptor `capabilities` and `tags` are descriptive metadata. A method that accepts a `scope` must advertise `scoped`; a method that cannot answer must throw `datasourceCliError(...)` so the pipeline reports the surface as unsearched instead of returning an empty set.
- Sources are opaque, slash-hierarchical paths (`/<name>/<instance>/chunks/<id>`). Never emit an OS path for the agent to `cat`, and never emit a `#` fragment.
- `describeSources()` explains the data without leaking storage paths; the `skillManifest()` body stays path- and PII-opaque.

## 3. Register in the factory

[`src/datasource/skills/factory.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/datasource/skills/factory.ts) materializes skills from the trusted `datasources` config section:

1. Add the import for your skill.
2. Add a `BUILDERS` entry keyed by the template name; the key becomes the config `type` and the default connection id.
3. If the backend exists on one platform only, add it to `PLATFORM_ONLY_SKILLS` so it is neither offered nor built elsewhere.
4. If you retire a template, add it to `RETIRED_TEMPLATES` so a config entry using it is collected as `retired` and skipped, never thrown.

`builtinDatasourceSkillNames()` then includes your name automatically, and `buildDatasourceSkills` reports unknown names without failing agent construction.

## 4. Wire the setup probe

[`src/cli/setup.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/cli/setup.ts) decides what `autorag setup` reports:

- `BUILTIN_BINARIES` maps a CLI-backed template to its binary; a missing binary reports `skipped: missing <binary> binary`.
- `DEFAULT_CREDENTIALS` lists the environment variables a non-CLI connector requires (for example `github` → `GITHUB_TOKEN`); a CLI-backed datasource is never required to hold one.
- `datasourceNames()` is the set of built-ins probed even when not configured; add names that should always be offered.
- A datasource with its own health story models it on `setupHister` and surfaces it on `SetupReport`.

Verify the wiring:

```bash
autorag setup --format json
```

The datasource must appear with `configured`, `skipped`, or `blocked` and an actionable reason.

## 5. Tests

- Add `test/datasource/skills/<name>.test.ts`. Cover argv and environment isolation, output parsing, a missing binary, malformed output, indexing, retrieval, auth and rate-limit diagnostics, and `#`-fragment rejection.
- A scope-capable datasource also needs multi-scope and user-scope narrowing tests.
- Run the focused files with vitest, for example `bunx vitest run test/datasource/skills/<name>.test.ts`.
- [`test/docs/site-contract.test.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/test/docs/site-contract.test.ts) fails when a built-in has no `docs/datasources/<name>.md`, so land step 6 in the same change.

## 6. Documentation

- Write `docs/datasources/<name>.md`. The site contract test requires this exact path for every built-in datasource.
- Add exactly one entry to [`docs/.vitepress/sidebar.mts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/docs/.vitepress/sidebar.mts) under Datasources → Knowledge and code, with `text` equal to the page's `title` and `link` `/datasources/<name>`.
- Follow the page format the docs tests enforce: YAML `title` and `description`, one `# H1` equal to the title, relative `.md` links to other pages, absolute GitHub URLs for code, and a language on every fenced code block. See [Page format](development.md#page-format) and [Contract tests](development.md#contract-tests).

## 7. Manual QA

- Add a harness under `scripts/manual-qa/run-qa-<name>-live.ts` when a local store exists, and record it in [Manual QA and live E2E](manual-qa.md#per-datasource-harnesses).
- Run it inside the repository's isolated QA shell (`make qa-shell`); never mount a real host home to make a native lane pass.

## 8. Issue template and labels

- Track the work with [`.github/ISSUE_TEMPLATE/datasource-skill.md`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/.github/ISSUE_TEMPLATE/datasource-skill.md).
- Labels: `datasource-skill` and `integration`, plus the `데이터 소스 추가` label; add `manual-qa-required` when real credentials or local-app permissions are needed.

## Acceptance criteria

- Every configured connection is searchable with no authorization step, and tool arguments cannot change trusted connector configuration.
- A query `scope` narrows and never widens; results are narrowed before merge and use slash-hierarchical sources with no `#` fragment.
- Expected auth and API failures return non-throwing diagnostics; a retrieval method that cannot answer surfaces the CLI's own words.
- The datasource appears in `autorag setup`, in `autorag status --json`, and as `datasource-<name>` in the agent's skill list.
- The docs page, sidebar entry, tests, and manual QA land in the same change.
