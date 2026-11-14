# TypeScript API

Embedding the librarian in your own app.

### 3. Programmatic Usage (TypeScript API)

```typescript
import { AutoRAGAgent } from "@autorag/librarian";

// Initialize librarian agent
const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
});

// Run curated search loop
const response = await agent.searchDocuments("Summarize recent compliance updates");

console.log("Answer:", response.answer);
for (const result of response.results) {
  console.log(`[${result.number}] ${result.title} (${result.source})`);
  console.log(`    ${result.summary}`);
}
```

A caller can steer a search while it runs, the way Enter steers a TUI search, and stop it:

```typescript
const run = (async () => {
  for await (const event of agent.searchDocumentsStream("When is the launch?")) {
    if (event.type === "preliminary") console.log("Quick:", event.response.answer);
    if (event.type === "complete") console.log("Final:", event.response.answer);
  }
})();

// A correction of the running search, not a new question. false = no search to steer.
agent.steer("Use the June plan, not the April one.");
// agent.abort() stops the whole search, including every later phase.
await run;
```

`steer()` reaches the search's next model request (also between phases), cuts in-flight retrieval and the baseline prefetch short, and always runs verification, whose prompt restates the correction. It returns `false` when no search is running, the search was stopped, or it is past its last model request; ask the text as a new question then.

A refresh can be cancelled and observed from code:

```typescript
const controller = new AbortController();
try {
  await agent.refresh(false, {
    signal: controller.signal, // controller.abort() stops it
    onProgress: ({ phase, processed, total, currentVirtualPath }) => {
      console.log(`${phase} ${processed}/${total}`, currentVirtualPath ?? "");
    },
  });
} catch (error) {
  if (controller.signal.aborted) console.log("refresh cancelled");
  else throw error;
}
```

`signal` is checked between files and between phases, kills an in-flight parse worker, and
terminates the MinSync, Jikji and datasource child processes (and their descendants). The
refresh then rejects with the signal's reason and is recorded as `failed`; files already
mirrored are kept and not re-parsed on the next run. `onProgress` reports `parsed`
(`processed`/`total` are source files, `currentVirtualPath` is the file about to be processed),
`minsync` (`processed`/`total` are files processed by MinSync), and `datasources`
(`processed`/`total` are datasource skills). `.autorag/refresh-progress.json` carries the running
`sourceFiles.total` and `parsedCounts` during parsing (refreshed every 25 files rather than per
file), and `minsync: { synced, total }` on every MinSync batch or heartbeat. Each progress write
also advances `updatedAt`; between events a stale timestamp means "phase still running", not
"stuck". Use the marker's pid through `getRefreshStatus().progress.ownerAlive` (or
`isRefreshOwnerAlive`) for liveness. The marker is replaced with a retried rename (an in-place
overwrite if a reader keeps the destination held, as on Windows). Without these options a refresh
behaves as before.

---
