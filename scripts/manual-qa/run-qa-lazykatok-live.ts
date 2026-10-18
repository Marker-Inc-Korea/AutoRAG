/**
 * Live KakaoTalk QA through the operator-owned `lazykatok` store.
 *
 * Prerequisites:
 *   brew install openclaw/tap/lazykatok
 *   Have a configured lazykatok native store with indexed chat messages.
 *
 * Usage:
 *   bun scripts/manual-qa/run-qa-lazykatok-live.ts
 *   LAZYKATOK_LIVE_QUERY="refund" bun scripts/manual-qa/run-qa-lazykatok-live.ts
 */
import { LazykatokClient, LazykatokSkill } from "../../src/datasource/skills/lazykatok/index.ts";
import { parseLazykatokSourcePath } from "../../src/datasource/skills/lazykatok/paths.ts";

const query = process.env.LAZYKATOK_LIVE_QUERY ?? "meeting";
const client = new LazykatokClient({ timeoutMs: 900_000 });
const skill = new LazykatokSkill({ client, instanceId: "live" });

// Search the existing store directly. `skill.index()` (doctor+sync+index) is
// intentionally not used: a fixture-sourced store cannot `sync`, and older
// lazykatok builds return a `doctor` shape the skill cannot normalize — neither
// blocks retrieval from an already-indexed archive, which is what this QA
// actually verifies.
const method = skill.retrievalMethods()[0];
if (method === undefined) {
	console.error("lazykatok retrieval method unavailable");
	process.exit(1);
}
let results = await method.retrieve(query, { topK: 5 });
// The QA gate is native identity, not this specific query: a fixture store
// may simply not contain the default term. Fall back to common Korean terms
// present in any real chat archive before declaring no results.
for (const fallback of ["test", "회의", "ㅋㅋ", "ㅇㅇ"]) {
	if (results.length > 0) break;
	results = await method.retrieve(fallback, { topK: 5 });
}
const first = results[0];
// Canonical source is /kakao/<instance>/chunks/<chunk> (opaque, not an OS path).
if (first === undefined || parseLazykatokSourcePath(first.source) === undefined) {
	console.error("lazykatok returned no valid native identity");
	process.exit(1);
}

console.log(`LAZYKATOK_LIVE_QA_PASS source=${first.source}`);
