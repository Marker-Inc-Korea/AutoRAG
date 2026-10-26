// Live calibration of the Jev citation relevance check. Every case is one
// answer that cites one chunk; `keep` chunks are on topic (fully or only in
// part) and must survive, `drop` chunks carry no information on the cited
// sentence or on what the user asked (table of contents, intro, navigation,
// footer, chatter, another topic) and must be removed. A case lists every
// message the user typed in the conversation; the multi-turn cases are the ones
// where the last message alone ("and the rent?") cannot tell a related chunk
// from an unrelated one. The check under test is the agent's own
// `citedChunks` + `judgeCitationRelevance`, so the state and questions are
// exactly what a search sends.
//
//   OPENROUTER_API_KEY=... bun scripts/manual-qa/run-qa-citation-relevance-live.ts [--runs N] [--verbose]
import { citedChunks } from "../../src/agent/answer-citations.ts";
import { CITATION_RELEVANCE_FLOOR, judgeCitationRelevance } from "../../src/agent/citation-relevance.ts";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { createJevJudge } from "../../src/agent/jev-extension.ts";

interface LabeledCase {
	readonly group: string;
	readonly label: "keep" | "drop";
	readonly name: string;
	/** Everything the user typed, oldest first. */
	readonly userMessages: readonly string[];
	readonly answer: (id: string) => string;
	readonly chunk: string;
}

const LEASE_Q = "서울 사무실 임대차 계약은 언제 갱신되고, 월세는 얼마야?";
const leaseRenew = (id: string): string => `- 서울 사무실 임대차 계약은 **2027-03-31**에 갱신됩니다 [${id}].`;
const leaseRent = (id: string): string => `- 월세는 **4,800,000원**입니다 [${id}].`;
const API_Q = "What is the API rate limit?";
const apiLimit = (id: string): string => `- The API allows **100 requests per minute** per key [${id}].`;
const COURT_Q = "How did the court divide the matrimonial assets in XRM v XRN?";
const courtSplit = (id: string): string =>
	`- The court applied the structured approach and ordered a **60/40** division [${id}].`;
const STANDUP_Q = "회사 단톡 스탠드업 몇 시야?";
const standup = (id: string): string => `- 스탠드업은 **오전 10시**입니다 [${id}].`;
const REFUND_Q = "Who must approve refund exceptions?";
const refund = (id: string): string => `- Refund exceptions require **director approval** before payout [${id}].`;

const CASES: readonly LabeledCase[] = [
	// On topic: must stay.
	{ group: "on-topic", label: "keep", name: "lease renewal date", userMessages: [LEASE_Q], answer: leaseRenew, chunk: "The Seoul office lease renews on 2027-03-31. The landlord must be notified of non-renewal by 2026-12-31." },
	{ group: "on-topic", label: "keep", name: "lease rent", userMessages: [LEASE_Q], answer: leaseRent, chunk: "Monthly rent is 4,800,000 KRW. The security deposit is 96,000,000 KRW." },
	{ group: "on-topic", label: "keep", name: "lease rent (Korean)", userMessages: [LEASE_Q], answer: leaseRent, chunk: "제3조(차임) 월 차임은 금 4,800,000원으로 하며 매월 말일까지 지급한다. 제4조(보증금) 보증금은 96,000,000원으로 한다." },
	{ group: "on-topic", label: "keep", name: "api limit", userMessages: [API_Q], answer: apiLimit, chunk: "Rate limit: 100 requests per minute per API key. Exceeding it returns HTTP 429." },
	{ group: "on-topic", label: "keep", name: "court split", userMessages: [COURT_Q], answer: courtSplit, chunk: "[45] Applying the structured approach, I assess the wife's indirect contributions at 40% and the husband's direct contributions at 60%. The matrimonial assets are accordingly divided 60/40." },
	{ group: "on-topic", label: "keep", name: "standup chat", userMessages: [STANDUP_Q], answer: standup, chunk: "[09:12] 지영: 스탠드업 내일부터 10시로 바꿀게요 / [09:13] 민수: 네 10시 확인했습니다" },
	{ group: "on-topic", label: "keep", name: "refund policy", userMessages: [REFUND_Q], answer: refund, chunk: "Refund exceptions now require director approval before payout. Customer support should cite the July handbook addendum." },
	// Related but only partial or adjacent: must stay (the memory gate rejects these; this gate must not).
	{ group: "partial", label: "keep", name: "renewal sentence, notice-deadline chunk", userMessages: [LEASE_Q], answer: leaseRenew, chunk: "The landlord must be notified of non-renewal by 2026-12-31." },
	{ group: "partial", label: "keep", name: "rent sentence, deposit chunk", userMessages: [LEASE_Q], answer: leaseRent, chunk: "The security deposit is 96,000,000 KRW, payable on signing." },
	{ group: "partial", label: "keep", name: "api limit, burst chunk", userMessages: [API_Q], answer: apiLimit, chunk: "Requests are throttled per key. Short bursts above the limit are tolerated for up to 5 seconds before 429 responses begin." },
	{ group: "partial", label: "keep", name: "refund, one-clause chunk", userMessages: [REFUND_Q], answer: refund, chunk: "Exceptions to the standard refund window must be signed off by a director." },
	// Junk: must go.
	{ group: "toc", label: "drop", name: "Korean table of contents", userMessages: [LEASE_Q], answer: leaseRenew, chunk: "목차\n1. 개요 ........ 1\n2. 임대 조건 ........ 3\n3. 계약 해지 ........ 7\n4. 부록 ........ 12" },
	{ group: "toc", label: "drop", name: "English table of contents", userMessages: [COURT_Q], answer: courtSplit, chunk: "Table of Contents\nIntroduction ........ [1]\nFacts ........ [5]\nIssues ........ [12]\nAnalysis ........ [20]\nConclusion ........ [60]" },
	{ group: "toc", label: "drop", name: "manual index", userMessages: [API_Q], answer: apiLimit, chunk: "Contents: Getting started 3 | Authentication 9 | Endpoints 14 | Errors 31 | Changelog 44 | Glossary 52" },
	{ group: "intro", label: "drop", name: "handbook intro", userMessages: [LEASE_Q], answer: leaseRenew, chunk: "Introduction\nThis handbook introduces the facilities team. Chapter 1 describes onboarding, Chapter 2 covers badges and parking, and Chapter 3 lists contacts." },
	{ group: "intro", label: "drop", name: "developer guide preface", userMessages: [API_Q], answer: apiLimit, chunk: "Preface\nWelcome to the Acme developer guide. This guide is organised into five parts. Part I introduces concepts; Part II covers authentication; Part III covers endpoints." },
	{ group: "intro", label: "drop", name: "Korean 머리말", userMessages: [REFUND_Q], answer: refund, chunk: "머리말\n본 지침서는 고객지원팀 신규 입사자를 위해 작성되었습니다. 제1장은 조직 소개, 제2장은 근무 규칙, 제3장은 업무 도구를 다룹니다." },
	{ group: "boilerplate", label: "drop", name: "Korean copyright", userMessages: [LEASE_Q], answer: leaseRent, chunk: "본 문서는 주식회사 ABC의 내부 자료이며 무단 복제 및 배포를 금합니다. Copyright 2024 ABC Corp. All rights reserved." },
	{ group: "boilerplate", label: "drop", name: "docs navigation bar", userMessages: [API_Q], answer: apiLimit, chunk: "Home > Docs > API Reference > Getting started | Previous | Next | Search docs | Edit this page on GitHub" },
	{ group: "boilerplate", label: "drop", name: "court header block", userMessages: [COURT_Q], answer: courtSplit, chunk: "XRM v XRN [2025] SGHCF 55\nHigh Court (Family Division) - Divorce (Transfer) No 1234 of 2024\nJudgment reserved. 12 March 2025." },
	{ group: "boilerplate", label: "drop", name: "email signature", userMessages: [REFUND_Q], answer: refund, chunk: "Best regards,\nMina Park | Customer Support Lead\nAcme Corp, 12 Teheran-ro, Seoul | mina@acme.example | Please consider the environment before printing this email." },
	{ group: "off-topic", label: "drop", name: "lunch menu", userMessages: [LEASE_Q], answer: leaseRent, chunk: "오늘의 점심 메뉴: 비빔밥 9,000원, 김치찌개 8,000원, 제육볶음 9,500원" },
	{ group: "off-topic", label: "drop", name: "scheduling chatter", userMessages: [LEASE_Q], answer: leaseRenew, chunk: "[10:32] 민수: 내일 회의 3시로 옮길게요 / [10:33] 지영: 넵 확인했습니다" },
	{ group: "off-topic", label: "drop", name: "auth section for rate limit", userMessages: [API_Q], answer: apiLimit, chunk: "Authentication: send your key in the X-API-Key header. Keys can be rotated from the dashboard." },
	{ group: "off-topic", label: "drop", name: "party background for division", userMessages: [COURT_Q], answer: courtSplit, chunk: "[5] The parties married in 2008 and have two children. The husband is a 45-year-old software engineer and the wife is a 43-year-old teacher." },
	{ group: "off-topic", label: "drop", name: "GitHub bot notification", userMessages: [STANDUP_Q], answer: standup, chunk: "dependabot[bot] opened a pull request: Bump vitest from 5.0.3 to 5.0.4 in /app. 3 checks passed." },
	// Multi-turn: the last message is elliptical, so what makes a chunk relevant is in the earlier turns.
	{ group: "multi-turn", label: "keep", name: "'and the rent?' after lease renewal", userMessages: [LEASE_Q.replace(", 월세는 얼마야?", "?"), "그럼 월세는?"], answer: leaseRent, chunk: "Monthly rent is 4,800,000 KRW. The security deposit is 96,000,000 KRW." },
	{ group: "multi-turn", label: "keep", name: "'what about burst?' after rate limit", userMessages: [API_Q, "what about bursts?"], answer: apiLimit, chunk: "Requests are throttled per key. Short bursts above the limit are tolerated for up to 5 seconds before 429 responses begin." },
	{ group: "multi-turn", label: "keep", name: "steer 'only the Seoul office' keeps Seoul chunk", userMessages: ["What is the monthly rent?", "Only the Seoul office, not Busan."], answer: leaseRent, chunk: "Seoul office: monthly rent is 4,800,000 KRW." },
	{ group: "multi-turn", label: "drop", name: "'and the rent?' still drops a TOC", userMessages: [LEASE_Q.replace(", 월세는 얼마야?", "?"), "그럼 월세는?"], answer: leaseRent, chunk: "목차\n1. 개요 ........ 1\n2. 임대 조건 ........ 3\n3. 계약 해지 ........ 7\n4. 부록 ........ 12" },
	{ group: "multi-turn", label: "drop", name: "'what about bursts?' still drops a nav bar", userMessages: [API_Q, "what about bursts?"], answer: apiLimit, chunk: "Home > Docs > API Reference > Getting started | Previous | Next | Search docs | Edit this page on GitHub" },
	{ group: "multi-turn", label: "drop", name: "topic change: earlier lease turns do not rescue lunch menu", userMessages: [LEASE_Q, "점심 메뉴 뭐였지?"], answer: (id: string) => `- 오늘 점심은 **비빔밥**입니다 [${id}].`, chunk: "아파트 관리비 납부 안내: 매월 25일까지 입금 바랍니다." },
];

const runsFlag = process.argv.indexOf("--runs");
const runs = runsFlag >= 0 ? Number(process.argv[runsFlag + 1] ?? "1") : 1;
const verbose = process.argv.includes("--verbose");

const judge = createJevJudge({ backend: "openrouter" });

async function rejectedByJev(testCase: LabeledCase): Promise<boolean> {
	const ledger = new EvidenceLedger();
	const id = ledger.register({ method: "semantic_search_local_docs", source: "/qa/doc.md", content: testCase.chunk });
	const chunks = citedChunks(testCase.answer(id), ledger, { allowLocalFiles: false });
	const result = await judgeCitationRelevance(judge, { userMessages: testCase.userMessages, chunks });
	if (result.fallbackReason !== undefined) throw new Error(`${testCase.name}: ${result.fallbackReason}`);
	return result.rejected.size > 0;
}

const wrong = new Map<string, number>();
let keptOk = 0;
let keptTotal = 0;
let droppedOk = 0;
let droppedTotal = 0;
const byGroup = new Map<string, { ok: number; total: number }>();
for (let run = 1; run <= runs; run++) {
	const verdicts = await Promise.all(CASES.map(rejectedByJev));
	for (const [index, testCase] of CASES.entries()) {
		const rejected = verdicts[index] === true;
		const correct = testCase.label === "drop" ? rejected : !rejected;
		if (testCase.label === "keep") {
			keptTotal += 1;
			if (correct) keptOk += 1;
		} else {
			droppedTotal += 1;
			if (correct) droppedOk += 1;
		}
		const group = byGroup.get(testCase.group) ?? { ok: 0, total: 0 };
		group.total += 1;
		if (correct) group.ok += 1;
		byGroup.set(testCase.group, group);
		if (!correct) wrong.set(`${testCase.label}: ${testCase.name}`, (wrong.get(`${testCase.label}: ${testCase.name}`) ?? 0) + 1);
		if (verbose) console.log(`run ${run} ${correct ? "ok   " : "WRONG"} ${testCase.label} ${testCase.name}`);
	}
}

console.log(`\nfloor P(relevant) < ${CITATION_RELEVANCE_FLOOR} rejects; ${CASES.length} cases x ${runs} run(s)`);
for (const [group, { ok, total }] of byGroup) console.log(`  ${group.padEnd(12)} ${ok}/${total}`);
console.log(`junk removed      ${droppedOk}/${droppedTotal}`);
console.log(`relevant retained ${keptOk}/${keptTotal}`);
if (wrong.size > 0) {
	console.log("misjudged:");
	for (const [name, count] of wrong) console.log(`  ${count}/${runs}  ${name}`);
}
process.exit(wrong.size === 0 ? 0 : 1);
