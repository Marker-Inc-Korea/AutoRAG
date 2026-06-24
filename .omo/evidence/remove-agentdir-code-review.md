# AutoRAG Agentdir Removal Re-Review

Goal: re-review the current diff after the external-path blocker fix. Scope is the non-`.omo` diff in `/Users/jeffrey/Projects-dev/AutoRAG-2.0`; no production fixes were made by this review.

codeQualityStatus: CLEAR
recommendation: APPROVE
blockers: none

## Skill Perspective Check

Ran before judging test relevance and maintainability.

- `omo:remove-ai-slops`: consulted the full skill. The current diff does not leave blocker-level slop. Deletion of agentdir/organizer tests is scoped to the requested integration removal. One low-severity tautological test remains noted below, but it does not block the agentdir removal because the actual direct-directory behavior, external IDs, duplicate basenames, and MinSync mapping were independently verified.
- `omo:programming`: consulted the full skill plus TypeScript README, TypeScript type/data/error references, and code-smells reference. No blocker-level TypeScript maintainability issue remains. The new path-mapping seam is typed, shared by posix and mirror sync, and covered by targeted tests.

## Evidence Inspected

- Diff: `git diff -- . ':(exclude).omo'`
- Changed files: `git diff --name-status -- . ':(exclude).omo'`
- Residual removal scan: `rg -n "agentdir|organizer|organize|AGENTDIR|@nomadamas" . --glob '!.omo/**' --glob '!node_modules/**'` returned exit 1 with no matches.
- Submitted RED evidence: `.omo/ulw-loop/evidence/remove-agentdir-external-path-red.txt` showed external paths producing `/sub/...` IDs and failing focused tests.
- Submitted GREEN evidence:
  - `.omo/ulw-loop/evidence/remove-agentdir-external-path-green.txt`
  - `.omo/ulw-loop/evidence/remove-agentdir-minsync-reviewfix.txt`
  - `.omo/ulw-loop/evidence/remove-agentdir-full-test-reviewfix.txt`
  - `.omo/ulw-loop/evidence/remove-agentdir-typecheck-reviewfix-final.txt`
  - `.omo/ulw-loop/evidence/remove-agentdir-biome-reviewfix.txt`
  - `.omo/ulw-loop/evidence/remove-agentdir-manual-qa-reviewfix.txt`
  - `.omo/ulw-loop/evidence/remove-agentdir-rg-reviewfix.txt`

## Independent Verification

- `npm test -- --run test/retrieval/posix.test.ts test/mirror/sync.test.ts test/integration/minsync-flow.test.ts test/minsync/minsync.test.ts`: pass, 22 tests.
- `npm test`: pass, 18 files / 128 tests.
- `npm run typecheck`: pass.
- `npx biome check .`: pass, no fixes applied.
- `git diff --check -- . ':(exclude).omo'`: pass.
- Manual QA rerun: `node --experimental-strip-types .omo/ulw-loop/evidence/remove-agentdir-manual-qa.mjs`: pass, `leakedRoot=false`.
- Additional duplicate-basename check: two external `docs` roots produced stable `/docs/...` and `/docs-2/...` IDs regardless of input order; parsed mirror keys matched posix IDs and no real roots leaked.
- Additional MinSync mapping check: `buildMinSyncPathMap()` mapped parsed output paths, absolute MinSync workspace paths, and relative `files/...` paths back to `/docs/...` and `/docs-2/...`.
- LSP diagnostics: attempted for `src` and `test`, but the LSP daemon timed out. This is not treated as a blocker because `tsc --noEmit` and Biome passed.

## CRITICAL

None.

## HIGH

None.

## MEDIUM

None.

## LOW

1. `test/retrieval/posix.test.ts:91` contains a weak "memory recording gate" test that asserts a local lambda (`name === "grep" || name === "find"`) instead of exercising production hook behavior.

   This is a remove-ai-slops/programming-perspective test-quality issue because the assertion is tautological. It is not a blocker for this re-review because the tested concern is not the fixed external-path blocker, and production removal/mapping behavior is covered by stronger tests plus independent checks.

## Review Notes

- External `searchPaths` now use deterministic source-root prefixes in the shared `src/filesystem/source-paths.ts` helper, and both `src/retrieval/methods/posix.ts` and `src/mirror/sync.ts` use that helper.
- Duplicate basenames are suffix-disambiguated (`/docs`, `/docs-2`) over sorted absolute roots, so results are stable across caller input order.
- MinSync preservation is intact: parsed mirror IDs remain the source of MinSync path-map resolution, and the current tests plus additional mapping check resolve vector paths back to opaque IDs.
- Agentdir and organizer integration are removed from source, tests, package manifests, README/AGENTS, and exports. The only remaining occurrences are in `.omo` evidence, which is excluded from the requested diff/scope.
- The dependency on `@nomadamas/agentdir` is removed from `package.json` and `package-lock.json`.

Final status: APPROVE.
