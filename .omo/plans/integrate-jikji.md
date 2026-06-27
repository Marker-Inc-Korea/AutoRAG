# integrate-jikji - Work Plan

## TL;DR (For humans)
**What you'll get:** AutoRAG will be able to use Jikji as an optional local file-discovery retrieval source. When configured, Jikji results will flow through the same numbered answer, feedback, memory, and merge paths as the existing retrieval methods.

**Why this approach:** Jikji's stable public contract is its local CLI JSON output, while AutoRAG already has a retrieval-method registry built for pluggable backends. Keeping Jikji behind an optional CLI adapter avoids coupling AutoRAG to Python internals and keeps existing users on the current Posix/MinSync behavior unless they opt in.

**What it will NOT do:** It will not vendor Jikji, auto-install Python packages, replace Posix/MinSync, expose raw paths in curated answers, enable media OCR/ASR by default, or install Jikji skills into user agent homes.

**Effort:** Medium
**Risk:** Medium - the main risk is subprocess/config hardening around a young external CLI contract.
**Decisions I made for you:** Jikji is opt-in; adapter uses the CLI only; descriptor type is `hybrid`; extension query-time retrieval is deferred because the current extension only manages Pi tools/prompts, not the library retrieval pipeline; extension work is limited to explicit config loading and refresh/prepare support.

Your next move: run `$start-work` to implement this plan. Full execution detail follows below.

---

> TL;DR (machine): Medium-risk optional CLI-backed `jikji` retrieval adapter with config, refresh, tests, docs, and real CLI QA; no product implementation starts from this plan alone.

## Scope
### Must have
- Add a new optional Jikji integration package under `src/jikji/`:
  - `src/jikji/types.ts` for config and parsed payload types.
  - `src/jikji/client.ts` for bounded subprocess execution of `jikji`.
  - `src/jikji/path-map.ts` or equivalent helper for safe path containment and AutoRAG source conversion.
  - `src/jikji/index.ts` public exports.
- Add `src/retrieval/methods/jikji.ts` implementing `RetrievalMethod`.
- Extend `AutoRAGAgentOptions` with optional `jikji?: Omit<JikjiMethodOptions, "root" | "searchPaths">`.
- Register `JikjiMethod` only when `options.jikji` is present; Posix remains always registered and MinSync remains independent.
- Add explicit config support for extension refresh only:
  - File: `.autorag/jikji.json`.
  - Schema:
    ```json
    {
      "enabled": true,
      "binaryPath": "jikji",
      "topK": 20,
      "timeoutMs": 10000,
      "maxBufferBytes": 1048576,
      "includeHidden": false,
      "includeSensitive": false,
      "parseTimeout": 5,
      "maxFiles": 0,
      "staleAfterSeconds": 86400,
      "exclude": []
    }
    ```
  - Invalid, missing, disabled, or non-object config means Jikji extension refresh is disabled and must not throw during session startup.
- Add an `autorag-jikji-refresh` extension command that runs `jikji prepare <source> --json` for each configured source root only when `.autorag/jikji.json.enabled === true`.
- Keep extension query-time retrieval explicitly out of scope. The extension currently sets Pi tool surfaces and system prompts; the programmatic retrieval pipeline belongs to `AutoRAGAgent`.
- Map Jikji results into `RetrievalResult`:
  - Prefer `answer_paths[]`, then `paths[]`, then compact `candidates[].p`.
  - Use evidence from matching `evidence_pack[]`, `judge_candidate_slate[]`, or compact `candidates[].ev`.
  - `source` is AutoRAG opaque root-relative source, never an absolute path.
  - `content` is bounded evidence text or the relative path when no evidence exists.
  - `score` is the candidate score when present, otherwise confidence-derived (`high` = `1`, `medium_high` = `0.75`, `medium` = `0.5`, `low` = `0.25`, unknown = `0`).
  - `metadata.method` is exactly `"jikji"` and also includes `confidence`, `handoffAction`, `indexStatus`, `queryType`, `why`, `matchedTerms`, and `nextRead` where available.
- Jikji descriptor:
  - `name: "jikji"`
  - `type: "hybrid"`
  - `status: "active"`
  - capabilities include `local-file-discovery`, `cli-json`, `fielded-search`, `agent-handoff`, `opaque-root-relative-paths`.
- Failure behavior:
  - Missing binary, spawn error, nonzero exit, timeout, cancellation, oversized output, malformed JSON, invalid payload, and all out-of-root paths return `[]`.
  - Do not throw from `retrieve()` for expected Jikji failures.
  - No fake diagnostic `RetrievalResult` for failures; preserve errors only in a client-level result type used by tests.
- Subprocess bounds:
  - Use `spawn`, not `exec`.
  - Default timeout `10000ms`.
  - Default stdout/stderr cap `1 MiB` each.
  - Respect `RetrievalOptions.signal` by terminating the subprocess and returning empty results.
  - Run roots sequentially to avoid surprise CPU/disk fan-out.
  - Use PATH resolution by spawning `"jikji"` directly when `binaryPath` is omitted or equals `"jikji"`; do not require `existsSync("jikji")`.
  - Use a controlled environment that removes any `JIKJI_ENABLE_MEDIA_INDEX` value and never adds `--enable-media-index` unless a future explicit option is added.
- Privacy defaults:
  - Do not pass `--include-hidden` or `--include-sensitive` unless explicit config/options set them true.
  - Do not pass media OCR/ASR flags.
  - Reject absolute paths, `..` traversal, and any resolved Jikji-returned path outside its source root.
- TopK semantics:
  - Per-root Jikji CLI receives `--top-k` equal to `options.topK ?? configuredTopK ?? 20`.
  - AutoRAG's existing `ResultMerger` remains responsible for final global `topK`.
- Tests and docs must prove:
  - Happy path retrieval from fake Jikji JSON.
  - Multi-root mapping and dedupe-safe opaque sources.
  - Missing binary/nonzero/malformed/timeout/outside-root fail closed.
  - Low-confidence and retry/raw-fallback payloads preserve metadata but do not override AutoRAG orchestration.
  - `searchDocuments()` and numbered feedback record `method: "jikji"`.
  - Extension config is explicit, backwards compatible, and does not alter the active Pi tool surface.
  - Real Jikji CLI smoke works against a temp corpus when Python/Jikji are available.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Must not vendor, fork, or rewrite Jikji Python internals.
- Must not add Jikji as a required `package.json` dependency or auto-install it for users.
- Must not replace Posix, Pi built-in `grep/find/read/ls`, parsed mirrors, MinSync, memory, or existing curation.
- Must not expose real paths in structured or curated user-facing responses.
- Must not let Jikji's `tool_call_policy` disable AutoRAG's own search strategy. Store it as metadata only.
- Must not enable hidden/sensitive/media indexing by default.
- Must not add destructive cleanup commands or skill-install commands that mutate user homes or global agent configs.
- Must not extend manifest types unless implementation cannot avoid it; the default plan does not require manifest changes.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD with Vitest for client/method/agent/extension surfaces. Each behavior change gets a failing-first test or focused failing command before production edits.
- Static checks:
  - `npm run typecheck`
  - `npm test`
  - `npm run check`
- Focused tests to add/run:
  - `npx vitest run test/jikji/jikji-client.test.ts`
  - `npx vitest run test/retrieval/jikji.test.ts`
  - `npx vitest run test/integration/jikji-flow.test.ts`
  - `npx vitest run test/extension/extension.test.ts`
  - `npx vitest run test/agent/search-documents.test.ts`
- Real-surface QA:
  - Tool: tmux.
  - Setup invocation:
    ```bash
    rm -rf /tmp/autorag-jikji-qa /tmp/autorag-jikji-venv
    mkdir -p /tmp/autorag-jikji-qa
    git clone --depth 1 https://github.com/NomaDamas/jikji /tmp/autorag-jikji-qa/jikji
    python3 -m venv /tmp/autorag-jikji-venv
    /tmp/autorag-jikji-venv/bin/pip install -e /tmp/autorag-jikji-qa/jikji
    ```
  - Scenario invocation:
    ```bash
    tmux new-session -d -s ulw-qa-jikji 'cd /Users/jeffrey/Projects-dev/AutoRAG-2.0 && node .omo/evidence/integrate-jikji/manual-qa.mjs'
    tmux capture-pane -pt ulw-qa-jikji -S -3000 > .omo/evidence/integrate-jikji/manual-qa-transcript.txt
    tmux kill-session -t ulw-qa-jikji
    ```
  - Binary observable: transcript contains `PASS jikji retrieval source=/docs/q3-report.txt method=jikji`, contains `PASS searchDocuments path hidden`, and does not contain `/tmp/autorag-jikji-qa-corpus` or `/Users/`.
- Evidence paths:
  - `.omo/evidence/integrate-jikji/task-1-red.txt`
  - `.omo/evidence/integrate-jikji/task-1-green.txt`
  - `.omo/evidence/integrate-jikji/task-2-red.txt`
  - `.omo/evidence/integrate-jikji/task-2-green.txt`
  - `.omo/evidence/integrate-jikji/task-3-red.txt`
  - `.omo/evidence/integrate-jikji/task-3-green.txt`
  - `.omo/evidence/integrate-jikji/task-4-red.txt`
  - `.omo/evidence/integrate-jikji/task-4-green.txt`
  - `.omo/evidence/integrate-jikji/task-5-red.txt`
  - `.omo/evidence/integrate-jikji/task-5-green.txt`
  - `.omo/evidence/integrate-jikji/final-typecheck.txt`
  - `.omo/evidence/integrate-jikji/final-test.txt`
  - `.omo/evidence/integrate-jikji/final-biome.txt`
  - `.omo/evidence/integrate-jikji/manual-qa-transcript.txt`

## Execution strategy
### Parallel execution waves
- Wave 1 is sequential enough to establish the client contract and method mapping before dependent agent/extension work.
- Wave 2 can split into agent wiring and extension config/refresh once the method API exists.
- Wave 3 covers docs and real QA harness after behavior is implemented.
- Final verification wave runs only after all todos pass.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| T1 Jikji client contract | none | T2, T4, T5 | none |
| T2 Retrieval method mapping | T1 | T3, T5, T6 | T4 after T1 |
| T3 Agent registration and feedback | T2 | T6, F1-F4 | T4 |
| T4 Extension config and refresh | T1 | T6, F1-F4 | T3 |
| T5 Failure hardening and subprocess bounds | T1, T2 | T6, F1-F4 | T3/T4 after shared API stable |
| T6 Docs and manual QA harness | T3, T4, T5 | F1-F4 | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] T1. Add bounded Jikji CLI client and payload parser
  What to do / Must NOT do: Create `src/jikji/types.ts`, `src/jikji/client.ts`, and `src/jikji/index.ts`. Define `JikjiOptions`, `JikjiClient`, `JikjiFindResult`, typed payload guards, and a subprocess helper that spawns `jikji find <root> <query> --json --top-k <n>` plus configured flags. Use `spawn`, timeout, buffer caps, `AbortSignal`, and controlled env. Do not call Python internals, do not use `exec`, do not auto-install Jikji, do not throw for expected CLI failures.
  Parallelization: Wave 1 | Blocked by: none | Blocks: T2, T4, T5
  References (executor has NO interview context - be exhaustive): `src/minsync/client.ts:1`, `src/minsync/process.ts:1`, `src/minsync/installer.ts:37`, `src/retrieval/types.ts:17`, `/tmp/autorag-jikji-readonly/src/jikji/__main__.py:917`, `/tmp/autorag-jikji-readonly/src/jikji/__main__.py:2050`, `/tmp/autorag-jikji-readonly/pyproject.toml:11`, `.omo/evidence/integrate-jikji/jikji-find-sample.json`
  Acceptance criteria (agent-executable):
  - Failing-first: before implementation, add tests in `test/jikji/jikji-client.test.ts` and run `npx vitest run test/jikji/jikji-client.test.ts > .omo/evidence/integrate-jikji/task-1-red.txt`; it must fail because modules are missing or behavior is unimplemented.
  - Green: `npx vitest run test/jikji/jikji-client.test.ts > .omo/evidence/integrate-jikji/task-1-green.txt` exits 0.
  - Test assertions cover exact args for PATH fallback (`jikji` command), configured `binaryPath`, default no hidden/sensitive/media flags, explicit hidden/sensitive flags, timeout returns `ok:false` or empty client result, malformed JSON returns a typed failure, and `AbortController.abort()` terminates the child.
  QA scenarios (name the exact tool + invocation):
  - Happy: `npx vitest run test/jikji/jikji-client.test.ts -t "runs jikji find with bounded json output"`, evidence `.omo/evidence/integrate-jikji/task-1-green.txt`, PASS if fake binary receives `["find", root, query, "--json", "--top-k", "20"]`.
  - Failure: `npx vitest run test/jikji/jikji-client.test.ts -t "returns failure for malformed json without throwing"`, evidence `.omo/evidence/integrate-jikji/task-1-green.txt`, PASS if result is failure and no exception escapes.
  Commit: Y | `feat(jikji): add bounded CLI client`

- [ ] T2. Add `JikjiMethod` retrieval adapter with safe source mapping
  What to do / Must NOT do: Create `src/retrieval/methods/jikji.ts` and any `src/jikji/path-map.ts` helper needed. Implement `RetrievalMethod` using the client from T1. Iterate configured source roots sequentially, pass per-root topK, reject absolute/traversal/out-of-root paths, convert valid paths to AutoRAG source IDs using `planSourceRoots()` / `sourceIdentifier()`, and produce bounded `RetrievalResult[]`. Descriptor type must be `hybrid`. Do not emit real paths or diagnostic fake hits.
  Parallelization: Wave 1 after T1 | Blocked by: T1 | Blocks: T3, T5, T6
  References (executor has NO interview context - be exhaustive): `src/retrieval/types.ts:1`, `src/retrieval/methods/posix.ts:25`, `src/filesystem/source-paths.ts:8`, `src/filesystem/source-paths.ts:24`, `src/minsync/method.ts:20`, `src/retrieval/merger.ts:8`, `.omo/evidence/integrate-jikji/jikji-find-sample.json`
  Acceptance criteria (agent-executable):
  - Failing-first: add `test/retrieval/jikji.test.ts`, run `npx vitest run test/retrieval/jikji.test.ts > .omo/evidence/integrate-jikji/task-2-red.txt`, and capture failure before production implementation.
  - Green: `npx vitest run test/retrieval/jikji.test.ts > .omo/evidence/integrate-jikji/task-2-green.txt` exits 0.
  - Tests assert descriptor fields exactly, high/direct payload maps to `source: "/docs/q3-report.txt"`, compact `candidates[].p` maps when `answer_paths` is absent, low-confidence payload still maps candidates with `metadata.handoffAction`, duplicate source results are not duplicated within one method result, outside-root absolute and `../` paths are dropped, and JSON serialization of results contains no root absolute path.
  QA scenarios (name the exact tool + invocation):
  - Happy: `npx vitest run test/retrieval/jikji.test.ts -t "maps Jikji answer paths to opaque retrieval results"`, evidence `.omo/evidence/integrate-jikji/task-2-green.txt`, PASS if first result has `metadata.method === "jikji"` and source `/docs/q3-report.txt`.
  - Failure: `npx vitest run test/retrieval/jikji.test.ts -t "drops paths outside the configured source root"`, evidence `.omo/evidence/integrate-jikji/task-2-green.txt`, PASS if results are `[]`.
  Commit: Y | `feat(retrieval): add jikji method`

- [ ] T3. Wire optional Jikji into `AutoRAGAgent`, public exports, curation, memory, and feedback
  What to do / Must NOT do: Extend `AutoRAGAgentOptions` with optional `jikji`, add `private readonly jikjiMethod`, register after Posix and independent of MinSync, update exports in `src/index.ts` and `src/retrieval/index.ts` as appropriate. Add integration tests proving `retrieve()` and `searchDocuments()` include Jikji results without paths and feedback stores `method: "jikji"`. Do not enable Jikji unless `options.jikji` is provided.
  Parallelization: Wave 2 | Blocked by: T2 | Blocks: T6
  References (executor has NO interview context - be exhaustive): `src/agent/agent.ts:34`, `src/agent/agent.ts:58`, `src/agent/agent.ts:64`, `src/agent/agent.ts:211`, `src/agent/agent.ts:231`, `src/agent/search-documents.ts:61`, `src/agent/search-documents.ts:88`, `src/memory/memory.ts:113`, `src/index.ts:1`, `test/integration/minsync-flow.test.ts:65`, `test/agent/search-documents.test.ts:19`
  Acceptance criteria (agent-executable):
  - Failing-first: add `test/integration/jikji-flow.test.ts` and focused `searchDocuments` expectations, run `npx vitest run test/integration/jikji-flow.test.ts test/agent/search-documents.test.ts > .omo/evidence/integrate-jikji/task-3-red.txt`, and capture failure before implementation.
  - Green: `npx vitest run test/integration/jikji-flow.test.ts test/agent/search-documents.test.ts > .omo/evidence/integrate-jikji/task-3-green.txt` exits 0.
  - Tests assert default agent method registry has Posix only; configured agent has Posix + Jikji; MinSync and Jikji can both be configured; `searchDocuments()` answer contains numbered evidence and no absolute path; `recordFeedbackByNumbers()` resolves a Jikji result to useful memory.
  QA scenarios (name the exact tool + invocation):
  - Happy: `npx vitest run test/integration/jikji-flow.test.ts -t "includes Jikji results in retrieve when configured"`, evidence `.omo/evidence/integrate-jikji/task-3-green.txt`, PASS if Jikji result appears with opaque source.
  - Failure: `npx vitest run test/integration/jikji-flow.test.ts -t "does not register Jikji by default"`, evidence `.omo/evidence/integrate-jikji/task-3-green.txt`, PASS if method names equal `["posix"]`.
  Commit: Y | `feat(agent): wire optional jikji retrieval`

- [ ] T4. Add explicit extension config and Jikji refresh/prepare command
  What to do / Must NOT do: Add config loader for `.autorag/jikji.json` in or near `src/extension.ts` without changing `.autorag/sources.json` shape. Register `autorag-jikji-refresh` that, when enabled, runs `jikji prepare <source> --json` sequentially for every configured source root and appends `autorag_jikji_refresh` with per-root summaries. In `session_start`, do not run Jikji automatically unless the plan executor decides startup cost is acceptable and tests prove no regression; default is command-only refresh. Keep `ACTIVE_TOOLS` unchanged. Invalid config disables Jikji refresh and appends or ignores according to existing extension style, but must not throw.
  Parallelization: Wave 2 | Blocked by: T1 | Blocks: T6
  References (executor has NO interview context - be exhaustive): `src/extension.ts:1`, `src/extension.ts:12`, `src/extension.ts:35`, `src/extension.ts:90`, `src/extension.ts:106`, `src/extension.ts:139`, `test/extension/extension.test.ts:64`, `/tmp/autorag-jikji-readonly/src/jikji/__main__.py:1981`, `/tmp/autorag-jikji-readonly/src/jikji/__main__.py:2055`
  Acceptance criteria (agent-executable):
  - Failing-first: extend `test/extension/extension.test.ts`, run `npx vitest run test/extension/extension.test.ts > .omo/evidence/integrate-jikji/task-4-red.txt`, and capture failure.
  - Green: `npx vitest run test/extension/extension.test.ts > .omo/evidence/integrate-jikji/task-4-green.txt` exits 0.
  - Tests assert active tools remain exactly `["bash","check_memory","find","grep","ls","read"]`; no Jikji command runs with missing/disabled/invalid config; enabled config with fake binary calls `prepare <source> --json` for each source; includeHidden/includeSensitive flags are only passed when true; no media flag is passed.
  QA scenarios (name the exact tool + invocation):
  - Happy: `npx vitest run test/extension/extension.test.ts -t "registers autorag-jikji-refresh when explicit config enables Jikji"`, evidence `.omo/evidence/integrate-jikji/task-4-green.txt`, PASS if append entry includes success for the source.
  - Failure: `npx vitest run test/extension/extension.test.ts -t "ignores invalid Jikji config without changing active tools"`, evidence `.omo/evidence/integrate-jikji/task-4-green.txt`, PASS if command is no-op and active tools unchanged.
  Commit: Y | `feat(extension): add explicit jikji refresh config`

- [ ] T5. Harden failure, cancellation, low-confidence, and multi-root behavior
  What to do / Must NOT do: Expand tests and implementation for expected failure modes across client, method, and agent integration. Ensure missing binary, nonzero exit, malformed JSON, timeout, cancellation, oversized stdout/stderr, outside-root paths, empty payloads, low confidence, `jikji_retry`, and `raw_fallback_after_retry` are all deterministic. Jikji handoff/tool policy is metadata only and never blocks AutoRAG/Posix/MinSync methods.
  Parallelization: Wave 2/3 | Blocked by: T1, T2 | Blocks: T6
  References (executor has NO interview context - be exhaustive): `src/retrieval/merger.ts:56`, `src/retrieval/merger.ts:69`, `src/jikji/client.ts` after T1, `src/retrieval/methods/jikji.ts` after T2, `/tmp/autorag-jikji-readonly/src/jikji/answer_pack.py:30`, `/tmp/autorag-jikji-readonly/src/jikji/answer_pack.py:44`, `/tmp/autorag-jikji-readonly/src/jikji/answer_pack.py:106`, `.omo/evidence/integrate-jikji/jikji-find-sample.json`
  Acceptance criteria (agent-executable):
  - Failing-first: add/extend hardening tests, run `npx vitest run test/jikji/jikji-client.test.ts test/retrieval/jikji.test.ts test/integration/jikji-flow.test.ts > .omo/evidence/integrate-jikji/task-5-red.txt`, and capture failure for at least one missing hardening behavior.
  - Green: `npx vitest run test/jikji/jikji-client.test.ts test/retrieval/jikji.test.ts test/integration/jikji-flow.test.ts > .omo/evidence/integrate-jikji/task-5-green.txt` exits 0.
  - Tests assert failure modes return empty results, not thrown errors; low-confidence candidates preserve `metadata.handoffAction`; Posix still contributes results when Jikji fails; global merged topK is honored by existing merger.
  QA scenarios (name the exact tool + invocation):
  - Happy: `npx vitest run test/integration/jikji-flow.test.ts -t "continues merging Posix results when Jikji fails"`, evidence `.omo/evidence/integrate-jikji/task-5-green.txt`, PASS if result method is `posix`.
  - Failure: `npx vitest run test/jikji/jikji-client.test.ts -t "kills a timed out Jikji subprocess"`, evidence `.omo/evidence/integrate-jikji/task-5-green.txt`, PASS if process exits and no hanging child remains.
  Commit: Y | `fix(jikji): fail closed for unsafe CLI results`

- [ ] T6. Update docs, manual QA harness, and final smoke artifacts
  What to do / Must NOT do: Update README or docs to explain optional Jikji setup, `.autorag/jikji.json`, programmatic `AutoRAGAgent({ jikji })`, extension refresh command, guardrails, and that Jikji does not replace Posix/MinSync. Add `.omo/evidence/integrate-jikji/manual-qa.mjs` or equivalent local QA script that creates a temp corpus, uses a real Jikji binary path from `JIKJI_BINARY` or `/tmp/autorag-jikji-venv/bin/jikji`, instantiates AutoRAG with Jikji, calls `retrieve()` and `searchDocuments()`, and prints PASS/FAIL. Add `.omo/evidence/integrate-jikji/plan-compliance-check.mjs`, a small Node script that reads this plan plus expected evidence files after execution and prints `PASS plan compliance` only when every todo has red/green evidence and no Must NOT guardrail is violated in `git diff -- . ':!.omo/**'`. Do not include secrets or absolute user paths in docs examples.
  Parallelization: Wave 3 | Blocked by: T3, T4, T5 | Blocks: F1-F4
  References (executor has NO interview context - be exhaustive): `README.md:58`, `README.md:62`, `README.md:86`, `AGENTS.md:65`, `.omo/evidence/integrate-jikji/jikji-find-sample.receipt.txt`, `/tmp/autorag-jikji-readonly/README.md:44`, `/tmp/autorag-jikji-readonly/docs/agent-installation.md:45`
  Acceptance criteria (agent-executable):
  - Failing-first: before docs/harness implementation, run intended QA script command and capture missing script failure in `.omo/evidence/integrate-jikji/task-6-red.txt`.
  - Green: run the tmux manual QA scenario from Verification strategy and capture `.omo/evidence/integrate-jikji/manual-qa-transcript.txt` with PASS lines.
  - Docs include exact config schema and state that Jikji is optional and CLI-based.
  QA scenarios (name the exact tool + invocation):
  - Happy: `tmux new-session -d -s ulw-qa-jikji 'cd /Users/jeffrey/Projects-dev/AutoRAG-2.0 && JIKJI_BINARY=/tmp/autorag-jikji-venv/bin/jikji node .omo/evidence/integrate-jikji/manual-qa.mjs'`, evidence `.omo/evidence/integrate-jikji/manual-qa-transcript.txt`, PASS if transcript has both PASS observables from Verification strategy.
  - Failure: `node .omo/evidence/integrate-jikji/manual-qa.mjs --missing-binary`, evidence `.omo/evidence/integrate-jikji/task-6-green.txt`, PASS if script prints `PASS missing jikji returns empty` and exits 0.
  Commit: Y | `docs(jikji): document optional retrieval setup`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  - Tool/invocation: `git diff -- . ':!.omo/**' > .omo/evidence/integrate-jikji/final-diff.patch && node .omo/evidence/integrate-jikji/plan-compliance-check.mjs > .omo/evidence/integrate-jikji/final-plan-compliance.txt`
  - PASS observable: `.omo/evidence/integrate-jikji/final-plan-compliance.txt` contains `PASS plan compliance`, confirms every Must Have is covered, every Must NOT Have is not present, and every todo has red/green evidence.
- [ ] F2. Code quality review
  - Tool/invocation: `multi_agent_v1.spawn_agent` with `agent_type: "lazycodex-code-reviewer"`, `fork_context: false`, and message: `TASK: read-only code quality review. DELIVERABLE: APPROVE or findings. SCOPE: review git diff in /Users/jeffrey/Projects-dev/AutoRAG-2.0 against .omo/plans/integrate-jikji.md, plus evidence under .omo/evidence/integrate-jikji. VERIFY: return APPROVE only if implementation satisfies every plan success criterion, tests/evidence are real, no scope creep or path leaks exist, and no Must NOT guardrail is violated. Do not edit files.`
  - Artifact path: copy the reviewer's final message into `.omo/evidence/integrate-jikji/final-code-review.md`.
  - PASS observable: `.omo/evidence/integrate-jikji/final-code-review.md` contains unconditional `APPROVE`; any finding loops back into fixes and full re-verification.
- [ ] F3. Real manual QA
  - Tool/invocation: tmux scenario from Verification strategy.
  - PASS observable: `.omo/evidence/integrate-jikji/manual-qa-transcript.txt` contains `PASS jikji retrieval source=/docs/q3-report.txt method=jikji` and `PASS searchDocuments path hidden`.
- [ ] F4. Scope fidelity
  - Tool/invocation:
    ```bash
    npm run typecheck > .omo/evidence/integrate-jikji/final-typecheck.txt
    npm test > .omo/evidence/integrate-jikji/final-test.txt
    npm run check > .omo/evidence/integrate-jikji/final-biome.txt
    git diff --stat > .omo/evidence/integrate-jikji/final-diffstat.txt
    ```
  - PASS observable: all commands exit 0; diffstat includes expected Jikji, agent, extension, tests, docs only; no vendored Jikji or global installer code.

## Commit strategy
- Do not auto-commit unless the user explicitly requests it.
- If committing is requested, use one logical Conventional Commit per todo:
  - `feat(jikji): add bounded CLI client`
  - `feat(retrieval): add jikji method`
  - `feat(agent): wire optional jikji retrieval`
  - `feat(extension): add explicit jikji refresh config`
  - `fix(jikji): fail closed for unsafe CLI results`
  - `docs(jikji): document optional retrieval setup`
- Each commit must have focused tests, `npm run typecheck`, and any relevant manual QA evidence current before the commit.
- Final commit footer, if a plan footer is required by the execution harness:
  ```text
  Plan: .omo/plans/integrate-jikji.md
  ```

## Success criteria
- Programmatic `AutoRAGAgent` can opt into Jikji and retrieve Jikji-backed results through the existing registry/merger.
- Default AutoRAG behavior is unchanged when Jikji is not configured.
- Jikji result paths are mapped to opaque AutoRAG sources and never leak real roots in `retrieve()` or `searchDocuments()` outputs.
- Numbered feedback for Jikji search results records and resolves `method: "jikji"`.
- Extension mode has explicit, backwards-compatible `.autorag/jikji.json` refresh support and keeps the active Pi tool surface unchanged.
- Jikji CLI failures, malformed payloads, timeouts, cancellation, and unsafe returned paths fail closed without breaking Posix/MinSync retrieval.
- Hidden, sensitive, and media indexing remain disabled by default.
- Documentation explains installation expectations, config, usage, and guardrails.
- Agent-executed QA, typecheck, full tests, Biome, and real Jikji CLI smoke all pass with evidence under `.omo/evidence/integrate-jikji/`.
