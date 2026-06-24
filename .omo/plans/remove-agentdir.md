# Remove Agentdir Integration

## TL;DR
> Summary:      Remove the `@nomadamas/agentdir` dependency, delete the virtual workspace/tool layer, and run AutoRAG directly against real search directories while keeping MinSync semantic retrieval active through parsed mirrors.
> Deliverables:
> - Real-filesystem source catalog shared by POSIX retrieval, parsed mirror sync, and MinSync path mapping.
> - Real-directory `posix` retrieval method registered beside MinSync without `Workspace` or virtual paths.
> - Extension and standalone agent surfaces that rely on Pi real filesystem built-ins plus `check_memory`, with no agentdir tool overrides.
> - Agentdir and virtual organizer source/tests removed; README/package exports/dependencies cleaned.
> - TDD evidence, manual QA evidence, atomic commits, and final push to `main`.
> Effort:       Large
> Risk:         High - this removes a cross-cutting filesystem abstraction used by startup, retrieval, mirror sync, tests, docs, and package metadata.

## Scope
### Must have
- Remove `@nomadamas/agentdir` from `package.json:19-27` and `package-lock.json`.
- Remove all runtime imports of `@nomadamas/agentdir` from `src/extension.ts:4`, `src/agent/agent.ts:6`, `src/mirror/sync.ts:4`, `src/retrieval/methods/posix.ts:1`, `src/agentdir/tools.ts:3`, `src/agentdir/workspace.ts:2`, and `src/agentdir/grep-core.ts:1`.
- Delete the agentdir virtual tree implementation under `src/agentdir/*`; do not leave compatibility wrappers.
- Replace `AgentdirPosixMethod` in `src/retrieval/methods/posix.ts:16-50` with a real-filesystem `PosixRetrievalMethod` that implements `RetrievalMethod` from `src/retrieval/types.ts:24-27`.
- Keep `posix` as the method `name` for memory continuity because `RetrievalMemory.getMethodPriority()` groups by `entry.method` in `src/memory/memory.ts:233-259`.
- Preserve MinSync registration in `src/agent/agent.ts:70-73`, `syncMinSync()` in `src/agent/agent.ts:253-255`, `MinSyncVectorMethod` behavior in `src/minsync/method.ts:20-67`, and MinSync workspace copying/path mapping in `src/minsync/workspace.ts:18-63`.
- Replace parsed mirror sync so it scans real search directories instead of `Workspace.exportMapping()` and `Workspace.readBytes()` in `src/mirror/sync.ts:32-120`.
- Preserve caller-facing path opacity: public `<results>`, `<answer>`, and structured `searchDocuments()` responses must not expose raw paths, while internal mapping/memory may store real file paths.
- Update `src/agent/system-prompt.ts:25-71` so it describes real-directory tools and removes agentdir/virtual-tree fallback wording.
- Update `src/extension.ts:97-180` so the extension no longer registers agentdir overrides, no longer initializes `.autorag/workspace`, and no longer activates agentdir tool names.
- Remove virtual layout organizer behavior from `src/organizer/organize-tool.ts:70-88`, `src/organizer/agents/organizer.md:1-16`, `src/organizer/index.ts:1-4`, and `test/organizer/organizer.test.ts:25-63`. Without agentdir, do not create real-directory mutation tools as a replacement.
- Update README claims at `README.md:58-62` to describe real-directory POSIX retrieval and MinSync preservation.
- Keep unrelated dirty files out of scope: do not edit existing dirty `AGENTS.md`; do not bulk-add unrelated existing `.omo/*` files.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Must not leave `@nomadamas/agentdir`, `src/agentdir`, `.autorag/workspace`, `getWorkspace`, `bootstrapMappings`, `refreshWorkspace`, `createAgentdirTools`, `AGENTDIR_TOOL_NAMES`, or `AgentdirPosixMethod` in runtime code/tests/docs except this plan file and historical `.omo` evidence.
- Must not replace virtual `mv`/`cp`/`mkdir`/`rmdir` with real filesystem mutation tools.
- Must not remove or weaken `RetrievalMethodRegistry`, `ParallelRetriever`, `ResultMerger`, or MinSync.
- Must not expose real paths in user-visible curated answer blocks or `searchDocuments().answer/results`; real paths are allowed only in internal mapping, memory attempts, parsed mirror indexes, and test-only assertions.
- Must not rely on `AGENTS.md` cleanup to make grep checks pass; `AGENTS.md` is dirty and out of scope.
- Must not use destructive git operations (`git reset --hard`, `git checkout --`, broad cleanup of `.omo`) or `git add .`.

### Risk list
- Search identity risk: `ResultMerger` deduplicates by `RetrievalResult.source` at `src/retrieval/merger.ts:39-50`, so changing source IDs can alter ranking/dedup behavior.
- Memory continuity risk: memory priority uses string method names at `src/memory/memory.ts:233-259`; renaming `posix` fragments historical learning.
- Path leakage risk: real paths are now available internally, so public response tests must assert that `searchDocuments()` still hides `/Users/` and fixture paths as in `test/agent/search-documents.test.ts:31-49`.
- MinSync regression risk: MinSync currently maps vector hits back through parsed mirror entries in `src/minsync/method.ts:51-64`; mirror index schema changes must preserve that path map.
- Extension regression risk: Pi supports built-ins and dynamic active tools, but AutoRAG currently overrides built-in names at `src/extension.ts:97-105`; the new extension must rely on Pi built-ins deliberately.
- Test churn risk: `vitest.config.ts:3-6` includes every `test/**/*.test.ts`, so stale agentdir tests will keep failing until deleted or rewritten.
- Dirty worktree risk: `AGENTS.md` is already modified and `.omo/` has unrelated untracked files; commits must stage exact intended paths only.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + Vitest (`package.json:11-14`, `vitest.config.ts:3-6`), followed by `tsc --noEmit` and Biome.
- QA policy: every task has agent-executed scenarios.
- Evidence: `.omo/evidence/task-<N>-<slug>.<ext>`
- Failing-first rule: for each implementation task, write or rewrite the focused test first, run the listed red command, capture red evidence, implement, then run the green command and capture green evidence.
- Full gates: `npm test`, `npm run typecheck`, `npm run check`, targeted no-agentdir grep checks, manual QA scripts, exact-file git staging checks, and `git push origin main`.

## Execution strategy
### Parallel execution waves
> Target 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks to maximize parallelism.

Wave 1 (no dependencies):
- Task 1: Add real filesystem source catalog and parsed mirror real-directory contract.
- Task 2: Replace POSIX retrieval with real-directory search.
- Task 3: Rewrite extension contract tests for Pi built-ins and no agentdir tool overrides.
- Task 4: Remove virtual organizer behavior and tests.
- Task 5: Add no-agentdir dependency/import red checks.

Wave 2 (after Wave 1):
- Task 6: Rewire `AutoRAGAgent` lifecycle to real directories and preserve MinSync.
- Task 7: Update parsed mirror index and MinSync path mapping for real-source IDs.
- Task 8: Rewrite system prompt and high-level agent tests for real-directory guidance.
- Task 9: Delete agentdir source/tests and remove package dependency.
- Task 10: Update README and public exports.

Wave 3 (after Wave 2):
- Task 11: Repair integration tests and structured search path-opacity coverage.
- Task 12: Run manual QA scenarios and capture artifacts.
- Task 13: Commit atomic changes and push to `main`.

Critical path: Task 1 -> Task 6 -> Task 7 -> Task 11 -> Task 12 -> Task 13

### Dependency matrix
| Task | Depends on | Blocks | Can parallelize with |
|------|------------|--------|----------------------|
| 1    | none       | 6, 7, 11 | 2, 3, 4, 5 |
| 2    | none       | 6, 9, 11 | 1, 3, 4, 5 |
| 3    | none       | 8, 9, 11 | 1, 2, 4, 5 |
| 4    | none       | 9, 10 | 1, 2, 3, 5 |
| 5    | none       | 9, 13 | 1, 2, 3, 4 |
| 6    | 1, 2       | 7, 11 | 8, 9, 10 |
| 7    | 1, 6       | 11 | 8, 9, 10 |
| 8    | 3, 6       | 11 | 7, 9, 10 |
| 9    | 2, 3, 4, 5, 6 | 11, 13 | 7, 8, 10 |
| 10   | 4, 8, 9    | 13 | 7, 11 |
| 11   | 6, 7, 8, 9 | 12, 13 | 10 |
| 12   | 11         | 13 | none |
| 13   | 10, 11, 12 | none | none |

## Todos
> Implementation + Test = ONE task. Never separate.
> Every task MUST have: References + Acceptance Criteria + QA Scenarios + Commit.

- [ ] 1. Add real filesystem source catalog and parsed mirror real-directory contract

  What to do: Introduce a shared real-filesystem source catalog, suggested path `src/filesystem/source-catalog.ts`, that accepts `searchPaths: string[]`, resolves real directories/files, recursively lists regular files, skips `.autorag` and `node_modules`, records absolute `sourcePath`, stable `documentPath` or equivalent internal ID, size, mtime, and bytes-reading helpers. Rewrite `syncParsedMirrors` so its public input is real search paths, not `Workspace`; it should call the parser registry with the new document path and source path. Update mirror tests first so they fail against the old `Workspace` signature, then implement. Keep parsed mirror outputs under `.autorag/parsed`.
  Must NOT do: Do not import `@nomadamas/agentdir`; do not recreate `.autorag/workspace`; do not use ad hoc shell commands from library code; do not include generated `.autorag/parsed` files in commits.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [6, 7, 11] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `src/agentdir/workspace.ts:56-94` - current deterministic search path mapping and bootstrap behavior to replace with direct source cataloging.
  - Pattern:  `src/mirror/sync.ts:32-120` - current parsed mirror sync flow that lists workspace mappings, reads bytes, parses, writes atomically, and removes stale outputs.
  - Pattern:  `src/mirror/sync.ts:130-139` - current atomic write and safe remove helpers to keep.
  - API/Type: `src/parser/types.ts:1-16` - parser input currently includes `virtualPath`, `sourcePath`, and `bytes`; update naming/contracts consistently.
  - API/Type: `src/parser/registry.ts:23-25` - parser selection currently keys off virtual path extension; update to key off real/document path extension.
  - API/Type: `src/mirror/index-store.ts:6-19` - parsed mirror index currently stores `virtualPath`, `sourcePath`, `outputPath`, parser and source stat metadata.
  - API/Type: `src/mirror/paths.ts:16-18` - output path hashes the logical path; keep deterministic safe output paths.
  - Test:     `test/mirror/sync.test.ts:53-210` - replace agentdir setup with real source directories and retain stale/poisoned index safety cases.
  - Test:     `test/parser/parser.test.ts:28-66` - parser tests currently pass `virtualPath`; rename or adapt to new parser input.
  - External: `https://nodejs.org/api/fs.html#fspromisesopendirpath-options` - primary Node API reference for directory traversal.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/mirror/sync.test.ts 2>&1 | tee .omo/evidence/task-1-source-catalog-red.txt` fails before implementation because `syncParsedMirrors` no longer accepts an agentdir `Workspace`.
  - [ ] Green evidence exists: `npm test -- test/mirror/sync.test.ts test/parser/parser.test.ts 2>&1 | tee .omo/evidence/task-1-source-catalog-green.txt` exits 0.
  - [ ] Static import check passes: `rg -n "@nomadamas/agentdir|from \"\\.\\./agentdir|from \"../../src/agentdir" src/mirror src/parser test/mirror test/parser` returns no matches.
  - [ ] A focused index assertion proves the mirror index stores real `sourcePath` values and no longer requires an agentdir workspace handle.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Real directory mirror sync
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {syncParsedMirrors, loadMirrorIndex} from './src/mirror/index.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-task1-')); const docs=join(root,'docs'); mkdirSync(docs,{recursive:true}); writeFileSync(join(docs,'note.txt'),'Alpha\\n'); const r=await syncParsedMirrors([docs],{root}); const idx=loadMirrorIndex(root); const entries=Object.values(idx.entries); if(r.scanned!==1||r.written!==1||entries.length!==1) throw new Error(JSON.stringify({r,entries})); if(readFileSync(entries[0].outputPath,'utf8')!=='Alpha\\n') throw new Error('mirror content mismatch'); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-1-source-catalog.txt
    Expected: Command exits 0 and evidence contains no thrown error.
    Evidence: .omo/evidence/task-1-source-catalog.txt

  Scenario: Missing source directory is graceful
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {syncParsedMirrors} from './src/mirror/index.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-task1-missing-')); const r=await syncParsedMirrors([join(root,'missing')],{root}); if(r.scanned!==0||r.written!==0) throw new Error(JSON.stringify(r)); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-1-source-catalog-error.txt
    Expected: Command exits 0 with scanned 0 and written 0.
    Evidence: .omo/evidence/task-1-source-catalog-error.txt
  ```

  Commit: YES | Message: `feat(filesystem): add real source catalog for mirrors` | Files: [`src/filesystem/source-catalog.ts`, `src/mirror/sync.ts`, `src/parser/types.ts`, `src/parser/registry.ts`, `src/mirror/index-store.ts`, `src/mirror/paths.ts`, `test/mirror/sync.test.ts`, `test/parser/parser.test.ts`]

- [ ] 2. Replace POSIX retrieval with real-directory search

  What to do: Rewrite `src/retrieval/methods/posix.ts` to export a real-filesystem `PosixRetrievalMethod` using the source catalog from Task 1 or an equivalent direct real directory provider. It must scan actual files under `searchPaths`, support regex and invalid-regex literal fallback like `agentdirGrep`, support `RetrievalOptions.scope`, `topK`, and `signal`, rank by match count plus shallow-path tie-break, and return `RetrievalResult` objects with `metadata.method === "posix"` and line metadata. Keep method `name: "posix"` and `type: "posix"`, but remove `virtual-paths` from capabilities.
  Must NOT do: Do not import `Workspace`, `agentdirGrep`, or any `src/agentdir` module; do not change `ResultMerger` or `RetrievalMethod` contracts.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [6, 9, 11] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `src/agentdir/grep-core.ts:31-59` - invalid regex literal fallback and match counting semantics to preserve.
  - Pattern:  `src/agentdir/grep-core.ts:70-96` - first matching line, match count, scoring, sorting, and `maxResults` behavior to preserve over real files.
  - Pattern:  `src/retrieval/methods/posix.ts:27-50` - current retrieval descriptor and result mapping to replace.
  - API/Type: `src/retrieval/types.ts:1-27` - result, descriptor, options, and method interface contract.
  - API/Type: `src/retrieval/merger.ts:39-50` - dedup uses `source`, so real source identity must be stable and meaningful.
  - Test:     `test/retrieval/posix.test.ts:27-86` - rewrite from `AgentdirPosixMethod`/virtual path assertions to real path/source ID assertions.
  - Test:     `test/retrieval/merger.test.ts:16-89` - keep existing merger/parallel behavior green.
  - External: `https://nodejs.org/api/fs.html#fspromisesreadfilepath-options` - primary Node API reference for reading real files.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/retrieval/posix.test.ts 2>&1 | tee .omo/evidence/task-2-posix-red.txt` fails after tests are rewritten to require real-directory behavior.
  - [ ] Green evidence exists: `npm test -- test/retrieval/posix.test.ts test/retrieval/merger.test.ts 2>&1 | tee .omo/evidence/task-2-posix-green.txt` exits 0.
  - [ ] `rg -n "AgentdirPosixMethod|agentdirGrep|WorkspaceProvider|virtual-paths|@nomadamas/agentdir" src/retrieval test/retrieval` returns no matches.
  - [ ] `test/retrieval/posix.test.ts` asserts that a file with more matches ranks above one with fewer matches and that invalid regex input is treated as a literal query.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Real POSIX retrieval ranks by match count
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, mkdirSync, writeFileSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {PosixRetrievalMethod} from './src/retrieval/methods/posix.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-task2-')); const docs=join(root,'docs'); mkdirSync(docs,{recursive:true}); writeFileSync(join(docs,'many.txt'),'alpha alpha\\nalpha\\n'); writeFileSync(join(docs,'few.md'),'alpha\\n'); const method=new PosixRetrievalMethod([docs]); const results=await method.retrieve('alpha',{topK:2}); if(results.length!==2) throw new Error('expected 2 results'); if(!String(results[0].source).endsWith('many.txt')) throw new Error(JSON.stringify(results)); if(results[0].metadata.method!=='posix') throw new Error('missing method metadata'); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-2-posix.txt
    Expected: Command exits 0 and top result is `many.txt`.
    Evidence: .omo/evidence/task-2-posix.txt

  Scenario: Invalid regex falls back to literal
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, mkdirSync, writeFileSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {PosixRetrievalMethod} from './src/retrieval/methods/posix.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-task2-invalid-')); const docs=join(root,'docs'); mkdirSync(docs,{recursive:true}); writeFileSync(join(docs,'literal.txt'),'has ( parenthesis\\n'); const method=new PosixRetrievalMethod([docs]); const results=await method.retrieve('(',{}); if(results.length!==1) throw new Error(JSON.stringify(results)); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-2-posix-error.txt
    Expected: Command exits 0 and one literal match is returned.
    Evidence: .omo/evidence/task-2-posix-error.txt
  ```

  Commit: YES | Message: `feat(retrieval): search real directories for posix results` | Files: [`src/retrieval/methods/posix.ts`, `test/retrieval/posix.test.ts`, `test/retrieval/merger.test.ts`]

- [ ] 3. Rewrite extension tests for Pi built-ins and no agentdir tool overrides

  What to do: Rewrite extension tests first to require that AutoRAG registers only AutoRAG-specific tools/commands, preserves Pi built-in real filesystem tools (`grep`, `find`, `read`, `ls`, `bash`) through `setActiveTools`, and does not register agentdir overrides or virtual mutation tools. Update `src/extension.ts` to stop importing `Workspace`, `ACTIVE_TOOLS`, `AGENTDIR_TOOL_NAMES`, `createAgentdirToolDefinitions`, `bootstrapMappings`, `getWorkspace`, and `refreshWorkspace`. Keep `check_memory`, memory event handling, manifest prompt injection, `autorag-parse` if it can call real-directory `syncParsedMirrors`, and remove or rename `autorag-refresh` if it only existed for agentdir hash refresh.
  Must NOT do: Do not register tools named `mv`, `cp`, `mkdir`, `rmdir`, or `stat` as AutoRAG virtual tools; do not disable built-in `grep/find/read/ls/bash`; do not require `.autorag/sources.json` to create a virtual workspace.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [8, 9, 11] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `src/extension.ts:58-95` - current `check_memory` registration to keep.
  - Pattern:  `src/extension.ts:97-105` - agentdir tool override loop to remove.
  - Pattern:  `src/extension.ts:110-124` - commands currently tied to agentdir refresh/parse; adapt parse to real dirs and remove hash-refresh claims.
  - Pattern:  `src/extension.ts:127-140` - session start currently opens workspace, bootstraps mappings, refreshes, and syncs mirrors; replace with memory init plus optional real parsed mirror sync using configured sources.
  - Pattern:  `src/extension.ts:165-180` - active tool surface and system prompt injection to rewrite around Pi built-ins.
  - API/Type: `src/extension.ts:16-35` - helper functions for tool result memory recording to preserve.
  - Test:     `test/extension/extension.test.ts:68-130` - rewrite agentdir-specific registration/active-tool/refresh tests.
  - External: `https://github.com/earendil-works/pi/blob/6338661485a0c71d430e391c2d92833212bbfb85/packages/coding-agent/src/core/extensions/types.ts#L1125-L1263` - Pi `ExtensionAPI` exposes `registerTool`, `getActiveTools`, `getAllTools`, and `setActiveTools`.
  - External: `https://github.com/earendil-works/pi/blob/6338661485a0c71d430e391c2d92833212bbfb85/packages/coding-agent/docs/extensions.md#L1554-L1578` - Pi active tools can include built-ins and dynamically registered tools.
  - External: `https://github.com/earendil-works/pi/blob/6338661485a0c71d430e391c2d92833212bbfb85/packages/coding-agent/README.md#L573-L578` - Pi built-ins include `read`, `bash`, `edit`, `write`, `grep`, `find`, and `ls`.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/extension/extension.test.ts 2>&1 | tee .omo/evidence/task-3-extension-red.txt` fails after tests assert no agentdir registration.
  - [ ] Green evidence exists: `npm test -- test/extension/extension.test.ts 2>&1 | tee .omo/evidence/task-3-extension-green.txt` exits 0.
  - [ ] `rg -n "agentdir|Agentdir|AGENTDIR|Workspace|refreshWithHashVerification|createAgentdir|bootstrapMappings|getWorkspace|refreshWorkspace" src/extension.ts test/extension/extension.test.ts` returns no matches.
  - [ ] Extension test asserts `setActiveTools` is called with real built-in names plus `check_memory` and excludes `edit`, `write`, `mv`, `cp`, `mkdir`, `rmdir`.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Extension activates real filesystem tool surface
    Tool:     bash
    Steps:    npm test -- test/extension/extension.test.ts --testNamePattern "active tool surface" 2>&1 | tee .omo/evidence/task-3-extension.txt
    Expected: Command exits 0 and the focused test asserts `grep`, `find`, `read`, `ls`, `bash`, and `check_memory` are active with no virtual mutation tools.
    Evidence: .omo/evidence/task-3-extension.txt

  Scenario: Extension has no agentdir import
    Tool:     bash
    Steps:    if rg -n "agentdir|@nomadamas|Workspace" src/extension.ts test/extension/extension.test.ts; then exit 1; fi 2>&1 | tee .omo/evidence/task-3-extension-error.txt
    Expected: Command exits 0 with no matches.
    Evidence: .omo/evidence/task-3-extension-error.txt
  ```

  Commit: YES | Message: `refactor(extension): use pi real filesystem tools` | Files: [`src/extension.ts`, `test/extension/extension.test.ts`]

- [ ] 4. Remove virtual organizer behavior and tests

  What to do: Remove the organizer sub-agent/tool surface because it is explicitly defined around agentdir virtual operations and would otherwise imply real filesystem mutation. Delete or de-export `createOrganizeTool`, `createOrganizeToolDefinition`, `runOrganizer`, `discoverAgents`, `findAgent`, and bundled `organizer.md` unless another non-agentdir caller still uses them. Rewrite tests to assert the organizer is gone from the public API and active tool surface, or delete `test/organizer/organizer.test.ts` if no organizer module remains.
  Must NOT do: Do not replace virtual reorganization with real `mv`/`cp` operations; do not keep a dormant tool named `organize` that mentions virtual layout.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [9, 10] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `src/organizer/organize-tool.ts:70-88` - extension tool definition currently says organizer uses agentdir virtual ops.
  - Pattern:  `src/organizer/organize-tool.ts:90-100` - standalone `AgentTool` wrapper to remove or de-export.
  - Pattern:  `src/organizer/agents/organizer.md:1-16` - bundled agent prompt exclusively references agentdir virtual namespace tools.
  - Pattern:  `src/organizer/agents.ts:49-64` - bundled/project organizer discovery used only by organizer delegation.
  - API/Type: `src/organizer/index.ts:1-4` - public exports that must be removed or adjusted.
  - Test:     `test/organizer/organizer.test.ts:25-63` - current expectations assert bundled virtual organizer and spawn-tolerant tool.
  - External: `https://github.com/earendil-works/pi/blob/6338661485a0c71d430e391c2d92833212bbfb85/packages/coding-agent/README.md#L573-L578` - Pi built-ins include real mutation tools; AutoRAG must not enable mutation tools as a replacement.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/organizer/organizer.test.ts test/extension/extension.test.ts 2>&1 | tee .omo/evidence/task-4-organizer-red.txt` fails after tests assert no organizer surface.
  - [ ] Green evidence exists: `npm test -- test/organizer/organizer.test.ts test/extension/extension.test.ts 2>&1 | tee .omo/evidence/task-4-organizer-green.txt` exits 0, or `test/organizer/organizer.test.ts` is deleted and `npm test -- test/extension/extension.test.ts` exits 0.
  - [ ] `rg -n "organize|organizer|virtual layout|virtual namespace|agentdir virtual" src test README.md package.json` returns no matches except unrelated historical `.omo` files excluded from the command.
  - [ ] Public exports in `src/index.ts` and `src/organizer/index.ts` do not expose organizer APIs if the module is removed.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: No organize tool in extension surface
    Tool:     bash
    Steps:    npm test -- test/extension/extension.test.ts --testNamePattern "registers" 2>&1 | tee .omo/evidence/task-4-organizer.txt
    Expected: Command exits 0 and the registered tool names do not include `organize`.
    Evidence: .omo/evidence/task-4-organizer.txt

  Scenario: Organizer virtual text removed
    Tool:     bash
    Steps:    if rg -n "agentdir virtual|virtual namespace|virtual layout|reorganize the virtual" src test README.md package.json; then exit 1; fi 2>&1 | tee .omo/evidence/task-4-organizer-error.txt
    Expected: Command exits 0 with no matches.
    Evidence: .omo/evidence/task-4-organizer-error.txt
  ```

  Commit: YES | Message: `refactor(organizer): remove virtual layout delegation` | Files: [`src/organizer/organize-tool.ts`, `src/organizer/agents.ts`, `src/organizer/agents/organizer.md`, `src/organizer/index.ts`, `test/organizer/organizer.test.ts`, `src/index.ts`, `src/extension.ts`, `src/agent/agent.ts`]

- [ ] 5. Add no-agentdir dependency and import checks

  What to do: Add focused tests or scripts that fail while any runtime source/test/doc/package reference still depends on agentdir. Prefer a Vitest test such as `test/no-agentdir.test.ts` that scans `package.json`, `package-lock.json`, `src`, `test`, and `README.md` while explicitly excluding `.omo` and `AGENTS.md`. This task creates the red bar that Task 9 must turn green.
  Must NOT do: Do not edit `AGENTS.md`; do not scan `.omo` historical evidence; do not make the check pass by allowlisting runtime agentdir references.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [9, 13] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `package.json:19-27` - dependency block currently includes `@nomadamas/agentdir`.
  - Pattern:  `vitest.config.ts:3-6` - all `test/**/*.test.ts` files are included by default.
  - Pattern:  `src/index.ts:1-11` - public exports currently include `./agentdir/workspace.ts` at line 4.
  - Pattern:  `README.md:58-62` - docs currently advertise agentdir-backed POSIX and virtual layouts.
  - API/Type: `src/retrieval/types.ts:9-15` - descriptor capabilities can be checked to prevent `virtual-paths` capability from lingering.
  - Test:     `test/smoke.test.ts:5-10` - existing public API smoke test pattern.
  - External: `https://nodejs.org/api/fs.html#fsreadfilesyncpath-options` - Node API for reading source/package files in a static test.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/no-agentdir.test.ts 2>&1 | tee .omo/evidence/task-5-no-agentdir-red.txt` fails before cleanup.
  - [ ] Test asserts no matches for `@nomadamas/agentdir`, `from "../agentdir`, `from "./agentdir`, `src/agentdir`, `agentdir`, `virtual tree`, `virtual-paths`, `AGENTDIR_TOOL_NAMES`, and `AgentdirPosixMethod` in scanned runtime scopes.
  - [ ] The test explicitly excludes `.omo/**` and `AGENTS.md`.
  - [ ] The test is committed while red is captured, and Task 9 is responsible for making it green.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Static no-agentdir test is active
    Tool:     bash
    Steps:    npm test -- test/no-agentdir.test.ts 2>&1 | tee .omo/evidence/task-5-no-agentdir.txt
    Expected: Before cleanup this exits nonzero and reports at least one current agentdir reference.
    Evidence: .omo/evidence/task-5-no-agentdir.txt

  Scenario: Scanner excludes unrelated dirty guidance/evidence
    Tool:     bash
    Steps:    node --input-type=module -e "import {readFileSync} from 'node:fs'; const text=readFileSync('test/no-agentdir.test.ts','utf8'); if(!text.includes('AGENTS.md')||!text.includes('.omo')) throw new Error('missing explicit exclusions');" 2>&1 | tee .omo/evidence/task-5-no-agentdir-error.txt
    Expected: Command exits 0 and the test contains explicit exclusions.
    Evidence: .omo/evidence/task-5-no-agentdir-error.txt
  ```

  Commit: YES | Message: `test(agentdir): add removal guard` | Files: [`test/no-agentdir.test.ts`]

- [ ] 6. Rewire `AutoRAGAgent` lifecycle to real directories and preserve MinSync

  What to do: Update `src/agent/agent.ts` to remove `Workspace` state, `ensureWorkspace()`, `bootstrapMappings()`, `getWorkspace()`, `refreshWorkspace()`, and `createAgentdirTools()`. Register the new `PosixRetrievalMethod` with `this.searchPaths`; keep `MinSyncVectorMethod` registration untouched. Make `refresh(verifyHashes = false)` rescan parsed mirrors from real `searchPaths`, then call `syncMinSync()`. Decide the return type explicitly: either introduce a small `AutoRAGRefreshSummary` type for real scans or return the parsed mirror summary; update tests and docs accordingly. Standalone tools should include caller-provided tools plus `check_memory`; do not include agentdir `grep/find/read/ls` wrappers.
  Must NOT do: Do not remove `searchDocuments()`, feedback APIs, memory transform, MinSync registration, or retrieval merger pipeline.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [7, 11] | Blocked by: [1, 2]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `src/agent/agent.ts:63-89` - constructor currently registers agentdir POSIX, MinSync, tools, and prompt.
  - Pattern:  `src/agent/agent.ts:110-135` - memory afterToolCall contract; preserve for any caller-provided search tools with `details`.
  - Pattern:  `src/agent/agent.ts:218-251` - workspace lifecycle, refresh, and parsed mirror sync to replace.
  - Pattern:  `src/agent/agent.ts:253-267` - MinSync sync and retrieval merger flow to preserve.
  - API/Type: `src/minsync/method.ts:20-67` - MinSync semantic method remains registered and active only when configured.
  - API/Type: `src/retrieval/registry.ts:3-24` - method registration uniqueness and type grouping.
  - API/Type: `src/retrieval/merger.ts:56-76` - parallel retrieval failure isolation.
  - Test:     `test/agent/agent.test.ts:43-228` - update creation, prompt, feedback, and public API tests.
  - Test:     `test/agent/parser-mirror.test.ts:24-92` - refresh-to-mirror integration must stay green without agentdir cache.
  - Test:     `test/integration/minsync-flow.test.ts:68-94` - `AutoRAGAgent.refresh(true)` plus MinSync retrieval must still work.
  - External: `https://github.com/earendil-works/pi/blob/6338661485a0c71d430e391c2d92833212bbfb85/packages/coding-agent/docs/extensions.md#L1290-L1298` - Pi dynamic tool registration does not require AutoRAG to override built-ins.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/agent/agent.test.ts test/agent/parser-mirror.test.ts test/integration/minsync-flow.test.ts 2>&1 | tee .omo/evidence/task-6-agent-red.txt` fails before agent rewiring.
  - [ ] Green evidence exists: `npm test -- test/agent/agent.test.ts test/agent/parser-mirror.test.ts test/integration/minsync-flow.test.ts 2>&1 | tee .omo/evidence/task-6-agent-green.txt` exits 0.
  - [ ] `rg -n "Workspace|ensureWorkspace|workspaceHandle|workspaceReady|createAgentdirTools|bootstrapMappings|getWorkspace|refreshWorkspace|AgentdirPosixMethod" src/agent/agent.ts test/agent test/integration/minsync-flow.test.ts` returns no matches.
  - [ ] `agent.getMethodRegistry().getByType("posix").length` remains 1 and `agent.getMethodRegistry().getByType("vector").length` remains 1 when MinSync is configured.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Agent retrieves from real directory
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, mkdirSync, writeFileSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {AutoRAGAgent} from './src/agent/agent.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-task6-')); const docs=join(root,'docs'); mkdirSync(docs,{recursive:true}); writeFileSync(join(docs,'a.txt'),'alpha alpha\\n'); const agent=new AutoRAGAgent({searchPaths:[docs],memoryPath:join(root,'memory.json'),workspacePath:root}); const results=await agent.retrieve('alpha',{topK:1}); if(results.length!==1||results[0].metadata.method!=='posix') throw new Error(JSON.stringify(results)); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-6-agent.txt
    Expected: Command exits 0 and one POSIX result is returned from the real directory.
    Evidence: .omo/evidence/task-6-agent.txt

  Scenario: Agent missing source does not block MinSync-disabled retrieval
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {AutoRAGAgent} from './src/agent/agent.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-task6-missing-')); const agent=new AutoRAGAgent({searchPaths:[join(root,'missing')],memoryPath:join(root,'memory.json'),workspacePath:root}); const results=await agent.retrieve('alpha',{topK:1}); if(!Array.isArray(results)||results.length!==0) throw new Error(JSON.stringify(results)); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-6-agent-error.txt
    Expected: Command exits 0 and returns an empty array.
    Evidence: .omo/evidence/task-6-agent-error.txt
  ```

  Commit: YES | Message: `refactor(agent): remove virtual workspace lifecycle` | Files: [`src/agent/agent.ts`, `test/agent/agent.test.ts`, `test/agent/parser-mirror.test.ts`, `test/integration/minsync-flow.test.ts`]

- [ ] 7. Update parsed mirror index and MinSync path mapping for real-source IDs

  What to do: Update mirror index types and MinSync workspace mapping so MinSync keeps semantic retrieval over parsed mirrors after source IDs move from agentdir virtual paths to real-directory document IDs/source paths. `syncMinSyncWorkspace()` should still copy parsed mirror files under `.autorag/minsync/files`, and `buildMinSyncPathMap()` should resolve MinSync absolute and relative hit paths back to the internal real `sourcePath` used by retrieval/memory. Keep backward compatibility only as a read-time migration if needed; do not keep agentdir in the new write path.
  Must NOT do: Do not change MinSync installer behavior, binary invocation, or semantic result content parsing.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [11] | Blocked by: [1, 6]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `src/minsync/workspace.ts:18-43` - current mirror copy behavior to keep.
  - Pattern:  `src/minsync/workspace.ts:46-63` - current hit path remapping to update for real-source IDs.
  - Pattern:  `src/minsync/method.ts:49-67` - MinSync query results are mapped through `buildMinSyncPathMap()` into `RetrievalResult`.
  - API/Type: `src/minsync/types.ts` - MinSync CLI result types; do not alter unless tests require it.
  - API/Type: `src/mirror/index-store.ts:6-19` - mirror entry schema to migrate from `virtualPath` to new logical/real source fields.
  - Test:     `test/minsync/minsync.test.ts:95-212` - sync, query remap, relative path remap, missing binary, malformed JSON must stay green.
  - Test:     `test/integration/minsync-flow.test.ts:68-94` - end-to-end AutoRAGAgent plus fake MinSync must stay green.
  - External: `https://nodejs.org/api/path.html#pathrelativefrom-to` - Node path API for relative MinSync workspace paths.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/minsync/minsync.test.ts test/integration/minsync-flow.test.ts 2>&1 | tee .omo/evidence/task-7-minsync-red.txt` fails after tests require real-source IDs.
  - [ ] Green evidence exists: `npm test -- test/minsync/minsync.test.ts test/integration/minsync-flow.test.ts 2>&1 | tee .omo/evidence/task-7-minsync-green.txt` exits 0.
  - [ ] `MinSyncVectorMethod.describe()` still returns `name: "minsync"`, `type: "vector"`, `status: "active"` as in `src/minsync/method.ts:33-40`, but capabilities no longer include `virtual-paths`.
  - [ ] MinSync result `metadata.method` remains `"minsync"` and result content comes from fake MinSync query text.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: MinSync unit remapping remains green
    Tool:     bash
    Steps:    npm test -- test/minsync/minsync.test.ts --testNamePattern "returns vector results" 2>&1 | tee .omo/evidence/task-7-minsync.txt
    Expected: Command exits 0 and vector results map back to the real source identity without source path leakage in public assertions.
    Evidence: .omo/evidence/task-7-minsync.txt

  Scenario: Missing MinSync binary remains graceful
    Tool:     bash
    Steps:    npm test -- test/minsync/minsync.test.ts --testNamePattern "missing" 2>&1 | tee .omo/evidence/task-7-minsync-error.txt
    Expected: Command exits 0 and missing binary returns empty vector results or missing-binary sync status per existing tests.
    Evidence: .omo/evidence/task-7-minsync-error.txt
  ```

  Commit: YES | Message: `refactor(minsync): map vectors to real sources` | Files: [`src/minsync/workspace.ts`, `src/minsync/method.ts`, `src/mirror/index-store.ts`, `test/minsync/minsync.test.ts`, `test/integration/minsync-flow.test.ts`]

- [ ] 8. Rewrite system prompt and high-level agent tests for real-directory guidance

  What to do: Update `buildSystemPrompt()` to remove agentdir-first and virtual tree language. It should describe `grep`, `find`, `read`, `ls`, and `bash` as real filesystem tools when available, keep `check_memory` strategy guidance, keep read-only and no-public-path rules, and allow internal mapping to store real source paths. Update standalone and extension tests to assert no virtual-tree text and to preserve output contract.
  Must NOT do: Do not weaken the curated output path-opacity rule; do not tell the agent to prefer MinSync exclusively over POSIX or built-ins.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [11] | Blocked by: [3, 6]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `src/agent/system-prompt.ts:25-45` - current search tool guidance with virtual tree and agentdir fallback wording.
  - Pattern:  `src/agent/system-prompt.ts:67-71` - methods section currently says agentdir virtual tools are primary.
  - Pattern:  `src/agent/system-prompt.ts:130-158` - curated output and internal mapping contract to preserve, adjusted for real internal paths.
  - Pattern:  `src/agent/system-prompt.ts:160-168` - read-only and no raw path constraints to preserve.
  - Test:     `test/agent/agent.test.ts:82-164` - prompt tests for built-in tools, search strategy, output format, constraints, and tool quick reference.
  - Test:     `test/integration/full-flow.test.ts:31-73` - full-flow prompt/memory checks.
  - External: `https://github.com/earendil-works/pi/blob/6338661485a0c71d430e391c2d92833212bbfb85/packages/coding-agent/README.md#L573-L578` - Pi built-in real filesystem tool names.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/agent/agent.test.ts test/integration/full-flow.test.ts 2>&1 | tee .omo/evidence/task-8-prompt-red.txt` fails after tests assert real-directory prompt wording.
  - [ ] Green evidence exists: `npm test -- test/agent/agent.test.ts test/integration/full-flow.test.ts 2>&1 | tee .omo/evidence/task-8-prompt-green.txt` exits 0.
  - [ ] `rg -n "agentdir|virtual tree|virtual path|virtual-path|virtual document|virtual layout" src/agent/system-prompt.ts test/agent test/integration/full-flow.test.ts` returns no matches.
  - [ ] Prompt tests still assert `No raw paths`, `<internal_mapping>`, `READ-ONLY`, `check_memory`, regex guidance, glob guidance, and fallback chain.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Prompt advertises real filesystem tools
    Tool:     bash
    Steps:    node --input-type=module -e "import {buildSystemPrompt} from './src/agent/system-prompt.ts'; const p=buildSystemPrompt({mode:'extension',toolNames:['grep','find','read','ls','bash','check_memory'],memoryEntries:[],manifests:[]}); for (const word of ['real','grep','find','read','check_memory','No raw paths']) if(!p.includes(word)) throw new Error('missing '+word); if(/agentdir|virtual tree|virtual path/i.test(p)) throw new Error('stale virtual guidance');" 2>&1 | tee .omo/evidence/task-8-prompt.txt
    Expected: Command exits 0 and prompt has real-directory guidance with no agentdir/virtual wording.
    Evidence: .omo/evidence/task-8-prompt.txt

  Scenario: Prompt fallback when no search tools
    Tool:     bash
    Steps:    node --input-type=module -e "import {buildSystemPrompt} from './src/agent/system-prompt.ts'; const p=buildSystemPrompt({mode:'standalone',toolNames:['check_memory'],memoryEntries:[],manifests:[]}); if(!p.includes('No search tools')) throw new Error('missing no-search-tools fallback'); if(/agentdir|virtual/i.test(p)) throw new Error('stale wording');" 2>&1 | tee .omo/evidence/task-8-prompt-error.txt
    Expected: Command exits 0 and no-tool prompt fallback has no virtual wording.
    Evidence: .omo/evidence/task-8-prompt-error.txt
  ```

  Commit: YES | Message: `docs(prompt): describe real directory retrieval` | Files: [`src/agent/system-prompt.ts`, `test/agent/agent.test.ts`, `test/integration/full-flow.test.ts`]

- [ ] 9. Delete agentdir source/tests and remove package dependency

  What to do: Delete `src/agentdir/*`, delete or rewrite agentdir-only tests, remove `@nomadamas/agentdir` from `package.json` and `package-lock.json`, and update imports. The exact deletion set should include `src/agentdir/tools.ts`, `src/agentdir/workspace.ts`, `src/agentdir/grep-core.ts`, `src/agentdir/assert-no-source-path.ts`, `test/agentdir/tools.test.ts`, `test/agentdir/workspace.test.ts`, `test/agentdir/change-tracking.test.ts`, and `test/integration/agentdir-reachability.test.ts`. Run `npm uninstall @nomadamas/agentdir` or an equivalent package-manager operation that updates both package files.
  Must NOT do: Do not edit `AGENTS.md`; do not remove MinSync tests; do not remove source catalog or real POSIX tests; do not leave empty `src/agentdir` directories.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [11, 13] | Blocked by: [2, 3, 4, 5, 6]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `src/agentdir/tools.ts:7-276` - virtual tool surface to delete.
  - Pattern:  `src/agentdir/workspace.ts:1-109` - workspace lifecycle to delete.
  - Pattern:  `src/agentdir/grep-core.ts:1-97` - virtual grep core to delete after real POSIX replacement.
  - Pattern:  `src/agentdir/assert-no-source-path.ts:1-16` - agentdir-specific path guard to remove; replace public path-opacity assertions in high-level tests instead.
  - Pattern:  `package.json:19-27` - dependency block to update.
  - Pattern:  `package-lock.json:2973-3067` - current agentdir package entries to remove.
  - Test:     `test/agentdir/tools.test.ts:48-171` - virtual tools/ops tests to delete.
  - Test:     `test/agentdir/workspace.test.ts:38-124` - workspace lifecycle/mapping/refresh tests to delete or replace by Task 1/6 tests.
  - Test:     `test/agentdir/change-tracking.test.ts:38-89` - hash refresh tests tied to agentdir to delete; real filesystem scan no longer has agentdir issue #2.
  - Test:     `test/integration/agentdir-reachability.test.ts:37-56` - virtual grep-read reachability test to delete.
  - Test:     `test/no-agentdir.test.ts` - static guard from Task 5 must pass after this task.
  - External: `https://docs.npmjs.com/cli/v10/commands/npm-uninstall` - package manager command that removes package and updates lockfile.

  Acceptance criteria (agent-executable only):
  - [ ] Green evidence exists: `npm test -- test/no-agentdir.test.ts 2>&1 | tee .omo/evidence/task-9-no-agentdir-green.txt` exits 0.
  - [ ] `test ! -d src/agentdir` exits 0.
  - [ ] `test ! -d test/agentdir` exits 0.
  - [ ] `node -e "const p=require('./package.json'); if (p.dependencies && p.dependencies['@nomadamas/agentdir']) process.exit(1)"` exits 0.
  - [ ] `if rg -n "@nomadamas/agentdir|src/agentdir|from .*agentdir|agentdir|Agentdir|AGENTDIR|virtual tree|virtual-paths" package.json package-lock.json src test README.md; then exit 1; fi` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Agentdir package removed
    Tool:     bash
    Steps:    node -e "const fs=require('node:fs'); const pkg=require('./package.json'); if(pkg.dependencies?.['@nomadamas/agentdir']) throw new Error('package dependency still present'); const lock=fs.readFileSync('package-lock.json','utf8'); if(lock.includes('@nomadamas/agentdir')) throw new Error('lockfile dependency still present');" 2>&1 | tee .omo/evidence/task-9-dependency.txt
    Expected: Command exits 0 with no package or lockfile dependency.
    Evidence: .omo/evidence/task-9-dependency.txt

  Scenario: Agentdir runtime tree removed
    Tool:     bash
    Steps:    if test -d src/agentdir || test -d test/agentdir || rg -n "@nomadamas/agentdir|src/agentdir|agentdir" package.json package-lock.json src test README.md; then exit 1; fi 2>&1 | tee .omo/evidence/task-9-dependency-error.txt
    Expected: Command exits 0 with no runtime/test/docs agentdir references.
    Evidence: .omo/evidence/task-9-dependency-error.txt
  ```

  Commit: YES | Message: `refactor(agentdir): remove virtual workspace dependency` | Files: [`package.json`, `package-lock.json`, `src/agentdir/tools.ts`, `src/agentdir/workspace.ts`, `src/agentdir/grep-core.ts`, `src/agentdir/assert-no-source-path.ts`, `test/agentdir/tools.test.ts`, `test/agentdir/workspace.test.ts`, `test/agentdir/change-tracking.test.ts`, `test/integration/agentdir-reachability.test.ts`, `test/no-agentdir.test.ts`]

- [ ] 10. Update README and public exports

  What to do: Update `README.md` so it no longer advertises an agentdir-backed virtual layout and instead explains direct real-directory search plus MinSync semantic retrieval. Update `src/index.ts` and any sub-index files to remove agentdir/organizer exports and export the new real filesystem/source catalog API only if it is intended as public. Keep public `AutoRAGAgent`, `autoragExtension`, `buildSystemPrompt`, manifest/memory/minsync/mirror/parser/retrieval exports green.
  Must NOT do: Do not edit `AGENTS.md`; do not expose private filesystem helpers unless the executor intentionally treats them as package API.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [13] | Blocked by: [4, 8, 9]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `README.md:47-63` - current retrieval and virtual layout sections to rewrite.
  - Pattern:  `README.md:70-99` - quick start examples to keep, verifying `searchPaths` still mean real directories.
  - Pattern:  `README.md:101-133` - high-level flow to keep with direct search wording.
  - API/Type: `src/index.ts:1-11` - public exports currently include agentdir workspace at line 4.
  - API/Type: `src/retrieval/index.ts:1-11` - retrieval exports to keep.
  - API/Type: `src/minsync/index.ts:1-18` - MinSync exports to keep.
  - Test:     `test/smoke.test.ts:5-10` - public API smoke test must stay green.
  - Test:     `test/no-agentdir.test.ts` - README and exports are scanned by the no-agentdir guard.
  - External: `https://github.com/earendil-works/pi/blob/6338661485a0c71d430e391c2d92833212bbfb85/packages/coding-agent/README.md#L573-L578` - Pi built-in tool names for docs accuracy.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/smoke.test.ts test/no-agentdir.test.ts 2>&1 | tee .omo/evidence/task-10-docs-red.txt` fails before docs/export cleanup.
  - [ ] Green evidence exists: `npm test -- test/smoke.test.ts test/no-agentdir.test.ts 2>&1 | tee .omo/evidence/task-10-docs-green.txt` exits 0.
  - [ ] `rg -n "agentdir|virtual layout|virtual tree|virtual-path|src/agentdir" README.md src/index.ts src/*/index.ts` returns no matches.
  - [ ] `node -e "import('./src/index.ts').then(m=>{for(const k of ['AutoRAGAgent','autoragExtension','buildSystemPrompt']) if(!(k in m)) throw new Error(k)})"` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Public exports remain available
    Tool:     bash
    Steps:    node --input-type=module -e "const m=await import('./src/index.ts'); for (const k of ['AutoRAGAgent','autoragExtension','buildSystemPrompt']) if(!(k in m)) throw new Error('missing '+k); if('getWorkspace' in m || 'bootstrapMappings' in m) throw new Error('agentdir workspace export leaked');" 2>&1 | tee .omo/evidence/task-10-docs.txt
    Expected: Command exits 0 and public core APIs remain while agentdir workspace exports are absent.
    Evidence: .omo/evidence/task-10-docs.txt

  Scenario: README has no stale virtual layout claims
    Tool:     bash
    Steps:    if rg -n "agentdir|virtual layout|virtual tree|virtual-path" README.md; then exit 1; fi 2>&1 | tee .omo/evidence/task-10-docs-error.txt
    Expected: Command exits 0 with no stale README terms.
    Evidence: .omo/evidence/task-10-docs-error.txt
  ```

  Commit: YES | Message: `docs(readme): document real directory retrieval` | Files: [`README.md`, `src/index.ts`, `src/retrieval/index.ts`, `src/minsync/index.ts`, `test/smoke.test.ts`]

- [ ] 11. Repair integration tests and structured search path-opacity coverage

  What to do: Update high-level tests to prove real POSIX retrieval, MinSync retrieval, feedback, memory, parsed mirrors, and structured search all work together without agentdir. Keep public response path-opacity assertions strong. Update `test/agent/search-documents.test.ts` so retrieval source can be a real path internally while JSON response still omits `/Users/` and fixture paths. Update `test/integration/full-flow.test.ts` and `test/integration/minsync-flow.test.ts` for real-directory behavior. Remove stale `clearWorkspaceCache()` calls from tests.
  Must NOT do: Do not assert virtual paths; do not remove path-opacity assertions; do not skip MinSync integration.

  Parallelization: Can parallel: YES | Wave 3 | Blocks: [12, 13] | Blocked by: [1, 2, 6, 7, 8, 9]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `test/agent/search-documents.test.ts:20-49` - structured search happy path and no-source-path assertions.
  - Pattern:  `test/agent/search-documents.test.ts:70-148` - feedback resolution by numbered search results.
  - Pattern:  `src/agent/search-documents.ts:61-112` - structured response generation and memory registration.
  - Pattern:  `src/agent/search-documents.ts:114-135` - numbered feedback resolution by source.
  - Pattern:  `test/integration/full-flow.test.ts:31-149` - manifest, memory, feedback, session registry integration.
  - Pattern:  `test/integration/minsync-flow.test.ts:68-94` - MinSync end-to-end retrieval plus path-opacity assertion.
  - API/Type: `src/memory/memory.ts:176-209` - source-to-attempt mapping for feedback.
  - API/Type: `src/retrieval/merger.ts:8-53` - merge/dedup/topK semantics.
  - Test:     `test/minsync/minsync.test.ts:95-212` - lower-level MinSync coverage should remain green.
  - External: `https://nodejs.org/api/path.html#pathbasenamepath-suffix` - Node path API useful for public titles/summaries without leaking directories.

  Acceptance criteria (agent-executable only):
  - [ ] Red evidence exists: `npm test -- test/agent/search-documents.test.ts test/integration/full-flow.test.ts test/integration/minsync-flow.test.ts 2>&1 | tee .omo/evidence/task-11-integration-red.txt` fails before integration repair.
  - [ ] Green evidence exists: `npm test -- test/agent/search-documents.test.ts test/integration/full-flow.test.ts test/integration/minsync-flow.test.ts 2>&1 | tee .omo/evidence/task-11-integration-green.txt` exits 0.
  - [ ] `rg -n "clearWorkspaceCache|getWorkspace|bootstrapMappings|/docs/|virtual|agentdir" test/agent test/integration test/retrieval test/mirror` returns no matches except test fixture text intentionally unrelated to agentdir.
  - [ ] `npm test -- test/minsync/minsync.test.ts test/integration/minsync-flow.test.ts` exits 0 after the integration updates.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Structured search hides real source paths
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, mkdirSync, writeFileSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {AutoRAGAgent} from './src/agent/agent.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-task11-')); const docs=join(root,'docs'); mkdirSync(docs,{recursive:true}); writeFileSync(join(docs,'meeting.txt'),'Meeting notes from real directory\\n'); const agent=new AutoRAGAgent({searchPaths:[docs],memoryPath:join(root,'memory.json'),workspacePath:root}); const response=await agent.searchDocuments('Meeting',{topK:1}); const json=JSON.stringify(response); if(!json.includes('Meeting notes')) throw new Error(json); if(json.includes(root)||json.includes(docs)||json.includes('/Users/')) throw new Error('public response leaked path: '+json); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-11-integration.txt
    Expected: Command exits 0; public response contains content but no real paths.
    Evidence: .omo/evidence/task-11-integration.txt

  Scenario: Blank query still skips filesystem access
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {AutoRAGAgent} from './src/agent/agent.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-task11-blank-')); const agent=new AutoRAGAgent({searchPaths:[join(root,'missing')],memoryPath:join(root,'memory.json'),workspacePath:root}); const response=await agent.searchDocuments('   ',{topK:1}); if(response.searched!==0||response.warnings[0]!=='empty-query') throw new Error(JSON.stringify(response)); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-11-integration-error.txt
    Expected: Command exits 0 and returns `empty-query` without touching missing source.
    Evidence: .omo/evidence/task-11-integration-error.txt
  ```

  Commit: YES | Message: `test(integration): verify real directory flows` | Files: [`test/agent/search-documents.test.ts`, `test/integration/full-flow.test.ts`, `test/integration/minsync-flow.test.ts`, `src/agent/search-documents.ts`, `src/memory/memory.ts`]

- [ ] 12. Run manual QA scenarios and capture artifacts

  What to do: Execute end-to-end manual QA from the command line using real temporary directories. Cover real POSIX search, MinSync semantic retrieval with fake binary, extension tool-surface activation, no-agentdir static scan, and package-level full checks. Store every transcript under `.omo/evidence/task-12-*`. This task is agent-executed manual QA, not user testing.
  Must NOT do: Do not require a human to inspect UI; do not use live external MinSync downloads; do not write outside temp directories except `.omo/evidence`.

  Parallelization: Can parallel: NO | Wave 3 | Blocks: [13] | Blocked by: [11]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `package.json:11-14` - full verification commands.
  - Pattern:  `src/agent/agent.ts:203-267` - structured search and retrieval entrypoints to exercise.
  - Pattern:  `src/extension.ts:53-180` - extension registration and prompt injection surface to exercise through tests.
  - Pattern:  `src/minsync/client.ts:19-43` - MinSync fake binary should return JSON for init/sync/query.
  - Test:     `test/extension/extension.test.ts` - extension unit tests serve as manual QA harness for tool surface.
  - Test:     `test/integration/minsync-flow.test.ts:68-94` - fake MinSync end-to-end scenario.
  - Test:     `test/no-agentdir.test.ts` - static no-agentdir scan.
  - External: `https://github.com/earendil-works/pi/blob/6338661485a0c71d430e391c2d92833212bbfb85/packages/coding-agent/README.md#L573-L578` - built-in tool assumptions used by extension QA.

  Acceptance criteria (agent-executable only):
  - [ ] `npm test 2>&1 | tee .omo/evidence/task-12-full-test.txt` exits 0.
  - [ ] `npm run typecheck 2>&1 | tee .omo/evidence/task-12-typecheck.txt` exits 0.
  - [ ] `npm run check 2>&1 | tee .omo/evidence/task-12-biome.txt` exits 0.
  - [ ] `npm test -- test/no-agentdir.test.ts 2>&1 | tee .omo/evidence/task-12-no-agentdir.txt` exits 0.
  - [ ] Manual QA transcripts exist for real POSIX, MinSync fake binary, extension surface, and error/empty query cases.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Real POSIX plus structured search manual QA
    Tool:     bash
    Steps:    node --input-type=module -e "import {mkdtempSync, mkdirSync, writeFileSync, rmSync} from 'node:fs'; import {tmpdir} from 'node:os'; import {join} from 'node:path'; import {AutoRAGAgent} from './src/agent/agent.ts'; const root=mkdtempSync(join(tmpdir(),'autorag-manual-posix-')); const docs=join(root,'docs'); mkdirSync(docs,{recursive:true}); writeFileSync(join(docs,'policy.txt'),'Refund policy requires manager approval\\nRefund windows are 30 days\\n'); const agent=new AutoRAGAgent({searchPaths:[docs],memoryPath:join(root,'memory.json'),workspacePath:root}); const raw=await agent.retrieve('Refund',{topK:2}); const structured=await agent.searchDocuments('Refund',{topK:1}); if(raw.length<1||structured.results.length!==1) throw new Error(JSON.stringify({raw,structured})); const publicJson=JSON.stringify(structured); if(publicJson.includes(root)||publicJson.includes(docs)) throw new Error('public path leak'); console.log(JSON.stringify({rawCount:raw.length,structuredCount:structured.results.length,answer:structured.answer},null,2)); rmSync(root,{recursive:true,force:true});" 2>&1 | tee .omo/evidence/task-12-posix-manual.txt
    Expected: Command exits 0 and prints counts plus answer without real paths.
    Evidence: .omo/evidence/task-12-posix-manual.txt

  Scenario: MinSync fake binary end-to-end manual QA
    Tool:     bash
    Steps:    npm test -- test/integration/minsync-flow.test.ts 2>&1 | tee .omo/evidence/task-12-minsync-manual.txt
    Expected: Command exits 0 and fake MinSync result appears through `AutoRAGAgent.retrieve()`.
    Evidence: .omo/evidence/task-12-minsync-manual.txt
  ```

  Commit: YES | Message: `test(qa): record remove-agentdir manual evidence` | Files: [`.omo/evidence/task-12-full-test.txt`, `.omo/evidence/task-12-typecheck.txt`, `.omo/evidence/task-12-biome.txt`, `.omo/evidence/task-12-no-agentdir.txt`, `.omo/evidence/task-12-posix-manual.txt`, `.omo/evidence/task-12-minsync-manual.txt`]

- [ ] 13. Commit atomic changes and push to `main`

  What to do: Review the accumulated diff, ensure all intended commits are atomic and buildable, re-run the final verification wave, then push the branch to `origin/main`. Use exact staging paths for each commit; never stage dirty `AGENTS.md` or unrelated `.omo` files. Include `Plan: .omo/plans/remove-agentdir.md` in the final commit footer. If already on `main`, push `main`; if not, switch only with a non-destructive command after verifying no unstaged intended changes would be overwritten.
  Must NOT do: Do not squash away red/green evidence commits if the project wants atomic history; do not force push; do not use `git add .`; do not reset unrelated user changes.

  Parallelization: Can parallel: NO | Wave 3 | Blocks: [] | Blocked by: [10, 11, 12]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `package.json:11-14` - final test/typecheck/check scripts.
  - Pattern:  `AGENTS.md` - dirty guidance file is out of scope and must not be staged.
  - Pattern:  `.omo/plans/remove-agentdir.md` - plan file to reference in final commit footer.
  - API/Type: `src/index.ts:1-11` - public API must still load before push.
  - Test:     `test/no-agentdir.test.ts` - final no-agentdir guard.
  - External: `https://git-scm.com/docs/git-status` - status command for exact-file staging verification.
  - External: `https://git-scm.com/docs/git-push` - non-force push command reference.

  Acceptance criteria (agent-executable only):
  - [ ] `git diff --check 2>&1 | tee .omo/evidence/task-13-diff-check.txt` exits 0.
  - [ ] `npm test 2>&1 | tee .omo/evidence/task-13-full-test.txt` exits 0.
  - [ ] `npm run typecheck 2>&1 | tee .omo/evidence/task-13-typecheck.txt` exits 0.
  - [ ] `npm run check 2>&1 | tee .omo/evidence/task-13-biome.txt` exits 0.
  - [ ] `npm test -- test/no-agentdir.test.ts 2>&1 | tee .omo/evidence/task-13-no-agentdir.txt` exits 0.
  - [ ] `git status --short` shows no unstaged or staged intended code/package/test/doc changes; unrelated pre-existing `AGENTS.md` and unrelated `.omo/*` may remain unstaged.
  - [ ] `git push origin main 2>&1 | tee .omo/evidence/task-13-push.txt` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Final intended-file status is clean before push
    Tool:     bash
    Steps:    git status --short -- package.json package-lock.json README.md src test .omo/plans/remove-agentdir.md 2>&1 | tee .omo/evidence/task-13-status.txt
    Expected: Output contains no unstaged intended changes after commits; unrelated `AGENTS.md` is not part of this scoped status command.
    Evidence: .omo/evidence/task-13-status.txt

  Scenario: Push main succeeds
    Tool:     bash
    Steps:    git branch --show-current && git push origin main 2>&1 | tee .omo/evidence/task-13-push.txt
    Expected: Current branch is `main`; push exits 0 without force.
    Evidence: .omo/evidence/task-13-push.txt
  ```

  Commit: YES | Message: `chore(release): push real directory autorag refactor` | Files: [`package.json`, `package-lock.json`, `README.md`, `src/**`, `test/**`, `.omo/plans/remove-agentdir.md`, `.omo/evidence/task-12-*.txt`, `.omo/evidence/task-13-*.txt`]

## Final verification wave (MANDATORY - after all implementation tasks)
> Runs in PARALLEL. ALL must APPROVE. Surface results to the caller and wait for an explicit "okay" before declaring complete.
- [ ] F1. Plan compliance audit - verify every task checkbox is complete, every acceptance criterion has evidence, no Must-NOT-Have was introduced, and `.omo/plans/remove-agentdir.md` is referenced in the final commit footer.
- [ ] F2. Code quality review - run `npm run typecheck`, `npm run check`, inspect `git diff --check`, verify idioms match existing TypeScript style, and confirm no dead `src/agentdir`/organizer imports remain.
- [ ] F3. Real manual QA - rerun Task 12 real POSIX and MinSync scenarios, inspect evidence files, and verify public structured responses still hide real paths.
- [ ] F4. Scope fidelity - run `npm test -- test/no-agentdir.test.ts`, `rg` no-agentdir scans excluding `.omo`/`AGENTS.md`, confirm MinSync tests still pass, and confirm no real filesystem mutation tools replaced virtual operations.

## Commit strategy
- One logical change per commit. Conventional Commits (`<type>(<scope>): <subject>` body + footer).
- Atomic: every commit builds and passes its focused tests on its own.
- Use exact-file staging only, for example `git add src/retrieval/methods/posix.ts test/retrieval/posix.test.ts`; never use `git add .`.
- Leave unrelated dirty `AGENTS.md` and unrelated pre-existing `.omo/*` untouched unless explicitly listed in a task.
- No "WIP" / "fix typo squash later" commits on the final branch - clean up before merge.
- Reference the plan file path in the final commit footer: `Plan: .omo/plans/remove-agentdir.md`.
- After F1-F4 approve and the caller gives the explicit okay required by the final verification wave, push with `git push origin main` and capture `.omo/evidence/task-13-push.txt`.

## Success criteria
- All Must-Have items shipped and all Must-NOT-Have guardrails respected.
- `@nomadamas/agentdir`, `src/agentdir`, agentdir tests, virtual workspace commands, and virtual organizer behavior are gone from runtime source/tests/docs/package files.
- Real-directory POSIX retrieval and structured `searchDocuments()` work against actual directories.
- MinSync unit and integration tests pass and semantic vector retrieval remains active.
- Public curated outputs hide raw real paths while internal memory/mapping may store real source paths.
- All QA scenarios pass with captured evidence; F1-F4 approve; atomic commits are pushed to `origin/main`.
