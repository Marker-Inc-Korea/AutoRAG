# Indexing and retrieval benchmarks

Two kinds of file live here.

**`*.test.ts` are correctness gates.** They run in the normal suite (`vitest run`) and assert the
properties that make the optimisations legitimate: identical results, correct invalidation, no stale
answers. They are not timing tests and do not assert wall-clock numbers, so they are stable in CI.

**`measure-*.ts` are measurement scripts.** They are not tests and are not collected by vitest
(`vitest.config.ts` includes only `test/**/*.test.ts`). They print a table and exit. Run them by
hand when a performance claim needs re-checking; they build their own throwaway corpora under the
system temp directory and clean up after themselves.

## Correctness gates

| File | What it pins |
|---|---|
| `fingerprint-skip.test.ts` | An unchanged rebuild opens no mirror document and does not rewrite the artifact, while the first build provably needs those same documents. Also covers digest backfill for indexes written before `contentSha256` existed, and rebuild-on-change including the size-and-mtime-restored case. |
| `query-cache-equivalence.test.ts` | Query-time reuse is invisible: repeated queries, a warm instance versus a cold one, and every scope return byte-identical `[id, source, score]`. Mutation cases prove the reuse never serves stale results, including with the fingerprint sidecar missing. Runs against both engines. |
| `query-cache-bounds.test.ts` | Dropping the cache never changes an answer, the cache is retained below the byte budget (and dropped when the retained-byte estimate exceeds it even far below the old token bound), a missing fingerprint sidecar never pins a stale cache entry, and a rebuild performed by a separate operating system process is observed by a warm reader. |
| `refresh-lock.test.ts` | `refresh()` is one transaction: concurrent same-parameter calls run the pipeline once and share a result, different-parameter calls are refused without running any downstream stage, artifact and fingerprint stay consistent under twenty rounds of interleaving, and a thrown refresh releases the lock. |
| `refresh-lock-cross-process.test.ts` | The lock works between operating system processes, which is the collision that actually happens: a held lock refuses a second process, an owner that no longer exists is reclaimed, a live owner is honoured, racing processes leave a consistent artifact, and deterministic controls prove the refusal is the lock's doing (a child is refused while the parent holds the lock and completes after release) and that two concurrent completions are reachable. The racing pair overlaps by construction rather than scheduling luck: both children race on the lock primitive itself, then the winner holds the lock across a file-signal handshake until the loser has attempted its refresh, so the one-completion outcome holds however CI schedules them. |
| `artifact-integrity.test.ts` | Damaged on-disk state rebuilds instead of skipping: fingerprint deleted, artifact deleted, artifact truncated, fingerprint corrupt, or a different engine requested. |
| `mirror-sync-equivalence.test.ts` | The serial `syncParsedMirrors()` loop, pinned as an observable contract before it is parallelized: repeat-run determinism, per-corpus-shape counts and diagnostics, virtualPath `localeCompare` ordering independent of creation order, checkpoint-resume without reparsing, and the invariants parallelism must not break (no cross-file contamination, `contentSha256` matches disk bytes, path-opaque diagnostics). |

## Measurement scripts

| File | Command | What it measures |
|---|---|---|
| `measure-noop-refresh.ts` | `bun run test/bench/measure-noop-refresh.ts` | Cost of an unchanged BM25 sync across corpus sizes, with the fingerprint skip enabled and disabled. `force: true` reproduces the pre-change path. |
| `measure-stage-share.ts` | `bun run test/bench/measure-stage-share.ts` | Splits an unchanged refresh into its mirror and BM25 stages, so the end-to-end effect is reported instead of the BM25 stage alone. |
| `measure-query-cost.ts` | `bun run test/bench/measure-query-cost.ts [tantivy\|typescript-fallback]` | Per-query retrieval cost against corpus size **and query selectivity**. Selectivity is a required axis: a query matching every document exercises a different cost than one matching a handful, and reporting only one of them would misstate any change that narrows the candidate set. |
| `measure-break-even.ts` | `bun run test/bench/measure-break-even.ts` | How many queries a rebuild must be followed by before query-time reuse pays for itself, reported for a broad and a selective query. Reuse moves work onto the first query after a rebuild, so the honest question is whether the total is lower, not whether later queries are faster. |
| `measure-decode.ts` | `bun run test/bench/measure-decode.ts` | Per-document `decodeText` cost before and after the pure-ASCII fast path (ASCII / Korean UTF-8 / CP949), the isolated purity-scan cost, and full mirror refresh deltas at two corpus sizes per document shape. The Korean rows are expected to stay at 1x: chardet still runs there, and the scan is the only added cost. |

### Reading the numbers

These run on synthetic corpora, so treat the **shape** of each curve as the result and the absolute
milliseconds as environment-specific. A cost that grows with corpus size for a query that never
changes is the signal worth acting on; a flat curve for a selective query is what an index is
supposed to produce.

`measure-query-cost.ts` defaults to the `tantivy` engine because that is the default when the native
binding loads. The `typescript-fallback` figures describe the degraded path only.

### `reset-lock.test.ts` (5건)

`index reset`이 refresh lock에 참여함을 고정한다. lock이 잡힌 동안 reset은 센티널 파일을 하나도 지우지 못하고 종료 코드 1로 거부하며, 해제 후에는 정상 삭제된다. 네 번째 케이스는 lock 파일이 reset이 지우는 어떤 디렉터리 안에도 있지 않음을 단언한다 — 배치 회귀 전용이다.

변이 검증: `runIndex`의 lock 획득을 무력화하면 2건 실패, lock 경로를 `parsed/` 안으로 되돌려도 2건 실패.

### `mirror-sync-equivalence.test.ts` (13건)

`syncParsedMirrors()`의 **직렬 루프를 병렬화하기 전에 관찰 가능한 계약으로 고정**하는 안전망이다. 병렬화 이후 이 파일이 그대로 통과하는 것이 완료 조건이다. 고정 대상: (a) 같은 코퍼스를 새 워크스페이스에서 `force: true`로 두 번 돌렸을 때 카운트·진단 전체(순서 포함)·인덱스 필드·미러 바이트의 완전한 결정성, (b) 코퍼스 형태별 덤(신규 전량·일부 변경·일부 삭제·파싱 실패 섞임·크기 상한 초과·빈 코퍼스·중첩 디렉토리), (c) 생성 순서와 무관한 `virtualPath` `localeCompare` 정렬(병렬화 후 가장 깨지기 쉬운 계약), (d) 25개마다 찍히는 체크포인트가 진행 중에 디스크에 남고, 그 후 재동기화가 파서 호출을 한 번도 하지 않음, (e) 병렬화가 깨뜨리면 안 되는 불변식(미러-소스 교차 오염 없음, `contentSha256`과 디스크 바이트 일치, 진단 `source`의 path-opaque). 실행마다 달라지는 필드(`sourceMtimeNs`, `updatedAt`, 절대 경로)는 비교에서 제외하거나 정규화하며, 이유는 테스트 주석에 적혀 있다.

변이 검증: `listCurrentFiles`의 `entries.sort` 제거 → 5건 실패(결정적 순서 포함), `ParseError` catch의 `throw` 전환 → 4건 실패(실패 격리 포함), `deleted-mirror` 진단 push 제거 → 1건 실패, 진단 배열 reverse → 3건 실패(진단 순서 포함).
