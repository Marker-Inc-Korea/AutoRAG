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

const indexed = await skill.index();
if (!indexed.ok) {
	console.error(`katok index failed: ${indexed.code}`);
	process.exit(1);
}

const method = skill.retrievalMethods()[0];
if (method === undefined) {
	console.error("katok retrieval method unavailable");
	process.exit(1);
}
const results = await method.retrieve(query, { topK: 5 });
const first = results[0];
if (first === undefined || !/^kakao:[^\s/]+(?:\/[^\s/]+){1,2}$/u.test(first.source)) {
	console.error("katok returned no valid native identity");
	process.exit(1);
}

console.log(`KATOK_LIVE_QA_PASS source=${first.source}`);
