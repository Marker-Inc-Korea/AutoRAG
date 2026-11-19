---
title: "TypeScript API"
description: "Embed AutoRAG in your own app with AutoRAGAgent, AutoRAGLite, config loaders, streaming search, and the report schema."
---

# TypeScript API

AutoRAG is published as the `@autorag/librarian` package with three entry points, so you can drive the same runtime the CLI uses from your own code. This page covers the package exports map, the `AutoRAGAgent` and `AutoRAGLite` classes, the config loaders, streaming search, result types, and the report schema. All snippets import real, exported names; verify them against `src/index.ts`, `src/core.ts`, and `src/agent/`.

## Package exports

`package.json` declares the module and its subpaths:

| Import specifier | Source | Contents |
|---|---|---|
| `@autorag/librarian` | `dist/index.js` | Everything below plus all node/backend exports (`cloud`, `convert`, `datasource`, `dupey`, `everything`, `fsearch`, `jikji`, `memory`, `minsync`, `mirror`, `parser`, `retrieval`) |
| `@autorag/librarian/core` | `dist/core.js` | The model-free runtime facade, config loaders, and model resolution |
| `@autorag/librarian/profiles` | `dist/profiles.js` | Agent profiles for helper features (smart indexing, file-triggered automations) |

The package also installs two executables: `autorag` (`dist/cli/index.js`) and `autorag-mcp` (`dist/mcp/index.js`). Types ship as `.d.ts` beside each entry.

## AutoRAGAgent

`AutoRAGAgent` hosts the model-backed librarian loop. It takes `AutoRAGAgentOptions` (from `src/agent/agent.ts`), whose only required field is `searchPaths`.

```ts
import { AutoRAGAgent, buildAgentOptions, resolveConfig } from "@autorag/librarian";
import { resolveAgentModel } from "@autorag/librarian/core";

const config = resolveConfig({ flags: {} });
const resolved = await resolveAgentModel(config);
const agent = new AutoRAGAgent({
  ...buildAgentOptions(config),
  model: resolved.model,
  ...(resolved.apiKey === undefined ? {} : { apiKey: resolved.apiKey }),
});
```

`buildAgentOptions(config)` (exported from both entry points) maps a resolved `CliConfig` onto `AutoRAGAgentOptions`; `resolveAgentModel(config)` resolves the model, its `apiKey`, and any `providerApiKeys` from the config, environment, or signed-in credential.

### Selected methods

| Method | Returns | Purpose |
|---|---|---|
| `searchDocuments(query, options?)` | `Promise<SearchDocumentsResponse>` | Run the full two-phase curation and return the final answer with numbered results |
| `searchDocumentsStream(query, options?)` | `AsyncIterable<SearchDocumentsStreamEvent>` | Stream progress, the preliminary answer, and the final answer |
| `steer(text, images?)` | `boolean` | Correct the running search before its next model request; `false` when nothing is running |
| `abort()` | `void` | Stop the whole search, including later phases |
| `waitForIdle()` | `Promise<void>` | Resolve when the current run settles |
| `refresh(force?, options?)` | `Promise<AutoRAGRefreshResult>` | Parse and resync every configured index; model-free |
| `getRefreshStatus()` | `Promise<AutoRAGRefreshStatus>` | Path-opaque freshness/health snapshot |
| `listDatasources()` | `DatasourceCatalogEntry[]` | Identity and scope strings of configured datasources |
| `retrieve(query, options?)` | `Promise<RetrievalResult[]>` | Merged retrieval without the model loop |
| `startWatchRefresh(options?)` | `AutoRAGWatchRefreshHandle` | Start an incremental filesystem watcher |
| `convertFiles(request, signal?)` | `Promise<ConvertFilesDetails>` | Convert files with Filee |
| `createPiInteractiveRuntime()` | `Promise<AutoRAGPiInteractiveRuntime>` | Build the Pi session the TUI drives |

## Streaming search

`searchDocumentsStream` yields `progress`, `preliminary`, `revision`, and `complete` events. The `preliminary` response is the fast first answer; the `complete` response is the verified final one.

```ts
const run = (async () => {
  for await (const event of agent.searchDocumentsStream("When is the launch?")) {
    if (event.type === "progress") console.log(event.text);
    if (event.type === "preliminary") console.log("Quick:", event.response.answer);
    if (event.type === "revision") console.log(`Corrected ${event.revision.claimId}:`, event.revision.reason);
    if (event.type === "complete") console.log("Final:", event.response.answer);
  }
})();

// A correction of the running search, not a new question. false = no search to steer.
agent.steer("Use the June plan, not the April one.");
// agent.abort() stops the whole search, including every later phase.
await run;
```

**Corrections of the first answer.** A search that verifies its first answer corrects it in place instead of writing a second answer next to it. The `preliminary` answer is split into claims (`c1`, `c2`, … one per list item, sentence, table, or code block); the verification phase names the claims it changes through the `revise_answer` tool, and each applied change is a `revision` event between `preliminary` and `complete`:

| `revision.op` | Meaning | Fields |
|---|---|---|
| `replace` | The claim was wrong; this is the corrected claim. | `claimId`, `original`, `replacement`, `reason` |
| `retract` | The claim was wrong or unsupported and is removed. | `claimId`, `original`, `reason` |
| `add` | A new verified finding the first answer lacked. | `replacement`, `reason` |
| `confirm` | The claim was checked and holds. | `claimId`, `original`, `reason` |

`original` is the claim exactly as the caller read it in `preliminary.response.answer`, and citations in both `original` and `replacement` use the `[n]` numbers of the responses' `results`. `reason` is one line in the user's language. A claim is revised once. A correction that cites evidence must cite sources the run retrieved and that Jev finds relevant to the text: otherwise the tool call fails and the model fixes it in the same turn, so a `revision` is only emitted for an applied change.

`complete.response.answer` depends on what the verification phase did. When it applied at least one `replace`, `retract` or `add`, the answer is the preliminary answer with every correction applied, so a caller that swaps its first answer for it shows the corrected answer; a closing reply of the model that cites evidence is appended after it, and a plain closing note is not part of the answer. When it applied no correction (it confirmed claims, or changed nothing), the answer is the model's own reply, a short note as before, and the caller keeps its first answer. `autorag search` prints each revision as it arrives: `{"type": "revision", "revision": {…}}` with `--json`, and a `Corrected c1 — reason` headline with the original as `− ` lines (U+2212, so a claim that is itself a bullet does not read `- - `) and the replacement as `+ ` lines otherwise.


`steer()` reaches the search's next model request (including between phases), cuts in-flight retrieval and the baseline prefetch short, and always runs verification, whose prompt restates the correction. It returns `false` when no search is running, the search was stopped, or it is past its last model request — ask the correction as a new question then.

## Refresh and cancellation

`refresh` accepts an `AbortSignal` and an `onProgress` callback. The signal is checked between files and between phases, kills an in-flight parse worker, and terminates the MinSync, Jikji, and datasource child processes (and their descendants); the refresh then rejects with the signal's reason and is recorded as `failed`.

```ts
const controller = new AbortController();
try {
  await agent.refresh(false, {
    signal: controller.signal,
    onProgress: ({ phase, processed, total, currentVirtualPath }) => {
      console.log(`${phase} ${processed}/${total}`, currentVirtualPath ?? "");
    },
  });
} catch (error) {
  if (controller.signal.aborted) console.log("refresh cancelled");
  else throw error;
}
```

`onProgress` reports `parsed` (processed/total are source files), `minsync` (files processed by MinSync), and `datasources` (datasource skills).

## AutoRAGLite

`AutoRAGLite` is the model-free runtime facade: it resolves the same CLI config, builds an agent, and exposes indexing, status, retrieval, and reporting without constructing an LLM. `createAutoRAGLite(options)` resolves the config and returns one.

```ts
import { createAutoRAGLite } from "@autorag/librarian";

const lite = createAutoRAGLite({
  flags: { "search-paths": "/path/to/docs" }, // CLI flags keep precedence intact
});

await lite.refresh();
const { results, unsearched, diagnostics } = await lite.retrieve("quarterly plan", { topK: 5 });
for (const result of results) console.log(result.source, result.score, result.content.slice(0, 80));
```

Key members:

| Member | Purpose |
|---|---|
| `config` | The resolved `CliConfig` |
| `refresh(force?, options?)` / `index(force?, options?)` | Refresh configured indexes; `index` is an alias |
| `getRefreshStatus()` / `status()` | Freshness/health snapshot |
| `syncParsedMirrors(force?)`, `syncMinSync()`, `prepareJikji(options?)` | Run a single indexing method |
| `startWatchRefresh(options?)` | Incremental filesystem watcher |
| `listDatasources()` | Configured datasources |
| `retrieve(query, options?)` / `retrieveByMethod(query, options?)` | Merged, or per-method, retrieval |
| `searchSelected(query, selection, options?)` | Retrieve only the selected surfaces; unknown ids throw `RetrievalSelectionError` |
| `recordReport(query, details)` / `recordStructuredResultsSession(sessionId, query, details)` | Persist a typed report to retrieval memory |
| `getMemorySchema()` | Detached snapshot of persisted evidence state |
| `convertFiles(request, signal?)` / `listFileConversions(from?, signal?)` | Filee conversions |

`retrieve` and `searchSelected` return `{ results, diagnostics, unsearched, window? }`; `unsearched` lists any surface that did not run, so a partial answer is detectable.

## Config loaders

These are exported from `@autorag/librarian` and `@autorag/librarian/core` (`src/cli/config.ts`):

| Function | Signature | Purpose |
|---|---|---|
| `resolveConfig(input)` | `(ResolveConfigInput) => CliConfig` | Resolve flag/env/file precedence into a `CliConfig` |
| `resolveConfigReadOnly(input)` | same | Resolve without migrating, writing, or locking |
| `resolveConfigPath(input)` | `(ResolveConfigInput) => ResolvedConfigPath` | Resolve `{ configPath, explicit, legacyPath? }` |
| `buildAgentOptions(config, agentDir?)` | `(CliConfig, string?) => Omit<AutoRAGAgentOptions, "model">` | Map a config to agent options |
| `writeDefaultConfig(path, partial, opts?)` | `(string, Partial<CliConfig>, opts?) => void` | Write a config atomically |
| `writeConfigObject(path, config)` | `(string, unknown) => void` | Atomically replace a config file |
| `readRawConfigObject(path)` | `(string) => Record<string, unknown>` | Read a config as a plain object |
| `ConfigError` | class | Validation error raised by the loaders |

```ts
import { resolveConfig, resolveConfigReadOnly } from "@autorag/librarian";

const config = resolveConfig({ flags: { workspace: "/path/to/workspace" } });
// Read-only variant: never migrates a legacy config or takes the write lock.
const snapshot = resolveConfigReadOnly({ flags: {} });
```

`resolveAgentModel` and `resolveAgentModelDetailed` (in `@autorag/librarian/core`) return the resolved `Model`, its `apiKey`, and `providerApiKeys`; the detailed variant also returns a `role` describing provider, model id, capabilities, and auth source.

## Result types

`SearchDocumentsResponse` is what `searchDocuments` (and the `complete` stream event) returns:

| Field | Type | Meaning |
|---|---|---|
| `sessionId` | `string` | Session id used by `autorag evidence` |
| `query` | `string` | The question |
| `results` | `SearchDocumentResult[]` | Numbered curated results |
| `answer` | `string` | The final answer text |
| `searched` | `number` | Candidates gathered |
| `warnings` | `SearchDocumentWarning[]` | e.g. `empty-query` |
| `diagnostics` | `SearchDocumentDiagnostic[]` | Degraded-mode diagnostics (always populated at runtime) |
| `retrievalTrace` | `SearchDocumentRetrievalTraceEntry[]` | Candidates gathered on the degraded fallback path |

`SearchDocumentResult` has `number`, `title`, `summary`, `evidence` (`SearchDocumentEvidence[]` with `excerpt` plus optional `lineNumber`/`lineEnd`/`pageStart`/`pageEnd`/`section`), and optional `confidence` and `source`. `SearchDocumentDiagnostic` carries a stable `code`, a `severity` (`info` | `warning` | `error`), a `message`, and an optional path-opaque `source`. The full `code` union lives in `src/agent/search-documents.ts`.

## Report schema

`reportSchema` (TypeBox, exported from `@autorag/librarian`) is the persisted report shape and the input contract of `autorag lite report` and the MCP report tool, where an external curator produced the number → source mapping. It validates with `typebox`:

```ts
import { Value } from "typebox/value";
import { createAutoRAGLite, reportSchema, type AutoRAGResultsDetails } from "@autorag/librarian";

const candidate: unknown = JSON.parse(payload);
if (!Value.Check(reportSchema, candidate)) throw new Error("invalid report");
const details = Value.Parse(reportSchema, candidate) as AutoRAGResultsDetails;

const lite = createAutoRAGLite();
const response = lite.recordReport("quarterly plan", details);
console.log(response.sessionId, response.results.length);
```

The schema requires `answer`, `results` (each `number`, `title`, `summary`, `evidence[]`, `confidence` in `[0,1]`), and `mapping` (one entry per result number with `number`, `source`, `method`, `content`, and optional `evidenceRefs`). `autorag lite report` additionally enforces that result and mapping numbers are one-to-one positive integers.

## MCP server, in-process

`createAutoRAGMcpServer(lite, options)` builds the same MCP server `autorag-mcp` runs, so a host can embed it:

```ts
import { createAutoRAGLite, createAutoRAGMcpServer } from "@autorag/librarian";

const lite = createAutoRAGLite({ readOnly: true });
const server = createAutoRAGMcpServer(lite, { readOnly: true });
```

## Profiles entry point

`@autorag/librarian/profiles` exposes agent profiles — a system prompt, a tool set, and a thinking level turned into an in-memory Pi session on the agent's own model. `createProfileSession`, `listAutomationProfiles`, `getAutomationProfile`, `registerAutomationProfile`, `createRetrievalToolkit`, and `createWebTools` are the primary exports.

`createWebTools` opts a profile into the librarian's own `web_search` / `web_fetch` tools (in that order), sharing one fresh evidence ledger and never touching retrieval memory. `search`/`fetch` accept the corresponding `AutoRAGAgent` options, or `false` to leave that tool out.

## See also

- [CLI commands](cli.md) for the command-line equivalent of every method.
- [Configuration file](configuration.md) for `CliConfig` fields.
- [Environment variables](environment-variables.md) for credential resolution.
- [Retrieval overview](../retrieval/overview.md) for the retrieval result model.
