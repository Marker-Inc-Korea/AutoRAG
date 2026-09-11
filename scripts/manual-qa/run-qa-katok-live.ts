/**
 * Live KakaoTalk QA through the operator-owned `katok` store.
 *
 * Prerequisites:
 *   brew install openclaw/tap/katok
 *   Have a configured katok native store with indexed chat messages.
 *
 * Usage:
 *   bun scripts/manual-qa/run-qa-katok-live.ts
 *   KATOK_LIVE_QUERY="refund" bun scripts/manual-qa/run-qa-katok-live.ts
 */
import { KatokClient, KatokSkill } from "../../src/datasource/skills/katok/index.ts";

const query = process.env.KATOK_LIVE_QUERY ?? "meeting";
const client = new KatokClient({ timeoutMs: 900_000 });
const skill = new KatokSkill({ client, instanceId: "live" });

// Search the existing store directly. `skill.index()` (doctor+sync+index) is
// intentionally not used: a fixture-sourced store cannot `sync`, and older
// katok builds return a `doctor` shape the skill cannot normalize — neither
// blocks retrieval from an already-indexed archive, which is what this QA
// actually verifies.
const method = skill.retrievalMethods()[0];
if (method === undefined) {
	console.error("katok retrieval method unavailable");
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
// kakao:<chat>/<sender>/<chunk>; chat/sender names may contain spaces (see
// katokSource in src/datasource/skills/katok/methods.ts), so validate the
// scheme + non-empty segments rather than forbidding whitespace.
if (first === undefined || !/^kakao:[^/]+(?:\/[^/]+){1,2}$/u.test(first.source)) {
	console.error("katok returned no valid native identity");
	process.exit(1);
}

console.log(`KATOK_LIVE_QA_PASS source=${first.source}`);
