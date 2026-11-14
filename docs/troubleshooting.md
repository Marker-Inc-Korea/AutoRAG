# Troubleshooting

Diagnostic codes and fixes.

## Troubleshooting

When search returns nothing, a datasource disappears from results, refresh hangs, or the embedder is unavailable, run the **`autorag-doctor`** agent skill (`skills/autorag-doctor/SKILL.md`). Point your coding agent at it and say *"check AutoRAG"* — it walks the full diagnose-and-repair procedure and ends with a per-datasource status table showing what is indexed and what actually returns hits.

The first three commands cover most of it:

```bash
autorag status --json          # freshness, per-component state, diagnostics
autorag health --json          # model resolution + one live completion probe
autorag setup                   # resolves the embedding backend and reports it
```

Indexing is not the same thing as searchability, so always confirm retrieval
itself through the connected Lite MCP server:

```text
autorag.status {}
autorag.refresh {}
autorag.search {"query":"a word that certainly appears","topK":3}
autorag.search {"query":"recent topic","scope":"/discord/local/**","topK":3}
```

The CLI equivalents remain available for terminal repair, but do not use them
as the agent's normal Lite search path.

Common failures and their fix:

| Symptom | Diagnostic code | Fix |
|---|---|---|
| Results are missing recent files | `stale-index` | `autorag refresh --method parsed,minsync --json` |
| Semantic search returns nothing after changing the embedder, or after moving off the old embedding gateway | `embedding-identity-mismatch` | `autorag index rebuild --method minsync` |
| No remote endpoint answers, no `minsync serve` daemon could start, and the in-process addon is missing or failed | `embedder-unavailable` | run `minsync serve` by hand to see its own error, install MinSync, install the optional `@nomadamas/minsync-native` addon, or configure a remote `minSync.embedder` |
| A datasource is healthy in its own CLI but absent from results | — | confirm it is configured under `datasources` and connected through its native CLI |
| A datasource errors during refresh | `datasource-index-failed` | run that CLI's own `doctor` |
| MinSync binary missing (search and refresh fail with `MinSync is required ...`) | — | `cargo install minsync`, or leave `minSync.autoInstall` on and re-run refresh |
| Jikji missing | `jikji-unavailable` | check the Rust toolchain, re-run refresh |
| Jikji older than 0.2.0 (the message names the version and the binary path) | `jikji-unsupported-version` | re-run `autorag refresh --method jikji` to install the pinned build, or `cargo install jikji-cli --version 0.2.1 --locked --force`; with `jikji.autoInstall: false` or `jikji.binaryPath`, upgrade it yourself |
| Jikji prepare of a large root times out (`the Jikji prepare run timed out`) | `jikji-prepare-failed` | the refresh still finishes and the other roots are indexed. There is no fixed total wall: the per-root hang guard is `jikji.timeoutMs` (60 s base) plus `jikji.prepareTimeoutMsPerFile` ms per file under the root (default: `jikji.parseTimeout`, 5 s), uncapped, so a huge root is not cut by its size and only a run stuck beyond every file hitting its parse limit is killed. Set `prepareTimeoutMsPerFile: 0` for a flat `timeoutMs` wall. `jikji.parseTimeout` (seconds) is the per-file limit, forwarded to `jikji prepare --parse-timeout`; jikji skips a file that exceeds it. A timeout kills jikji's whole process group |
| `jikji_find` miss: the `discover` retry step fails (`Jikji discover retry failed: ... (exit status N): <stderr>`) | `jikji-discover-failed` | a miss makes AutoRAG run the `jikji discover ... --after-jikji-retry --retry-proof <proof>` command jikji named (the proof is bound to that root and rewritten query; `--no-background-refresh --no-auto-prepare` keep it from reindexing or rewriting `AGENTS.md`/`CLAUDE.md`/`.cursorrules`). The message carries the CLI exit status and stderr verbatim, real paths included; the failed root is dropped and healthy roots still answer. A `next_commands` line AutoRAG cannot parse as a plain `discover` step is quoted in the same diagnostic and is never executed |
| Windows file-name search fails during refresh | `everything-index-failed` | read the ES exit code and stderr in the message, then `autorag refresh --method everything --json` |
| A search stops before answering with `[decision-model-unavailable] ... (missing-credential)` | `decision-model-unavailable` (exit 3) | sign in to Dazzi Models with `/login`, or set `OPENROUTER_API_KEY`; the error's `reason` names what failed (Jev is required) |

Native datasource stores stay owned by their CLIs — AutoRAG never rebuilds them. Fix a broken archive with `lazykatok doctor`, `discrawl --json metadata`, `slacrawl --json doctor`, `wacrawl --json doctor`, `telecrawl --json doctor`, `notcrawl doctor`, or `mailcrawl doctor`, then re-run `autorag refresh --method datasources --json`.

---
