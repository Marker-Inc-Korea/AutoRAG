// Live calibration of the Jev datasource check against the datasources and
// retrieval memory of the real AutoRAG config. It drives the agent's own
// check (registered datasources + descriptions + similar past searches from
// memory), so the state Jev sees is exactly what a search sends.
//
//   OPENROUTER_API_KEY=... bun scripts/manual-qa/run-qa-jev-datasource-selection-live.ts [--runs N] [--state]
//
// Each case names the datasources that must be searched (>= 0.5) and the ones
// that must not (< 0.5); every other datasource is reported but not scored.
// `--state` prints the state of the first run of every case.
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { JevJudge } from "../../src/agent/jev-extension.ts";
import { buildAgentOptions, resolveConfigReadOnly } from "../../src/cli/config.ts";

interface LabeledCase {
	readonly query: string;
	readonly searched: readonly string[];
	readonly skipped: readonly string[];
}

const CASES: readonly LabeledCase[] = [
	{
		query: "What did we discuss with Team Attention about AutoRAG Inc. in Slack?",
		searched: ["slack"],
		skipped: ["mailcrawl", "whatsapp"],
	},
	{
		query: "금천 아트빌 디스코드에서 공지된 관리비 얘기 뭐였지?",
		searched: ["discord"],
		skipped: ["slack", "mailcrawl"],
	},
	{
		query: "Find the invoice email from AWS last month",
		searched: ["mailcrawl"],
		skipped: ["discord", "nomadamas"],
	},
	{
		query: "카톡에서 엄마가 보낸 주소 찾아줘",
		searched: ["kakao"],
		skipped: ["slack", "discord", "nomadamas"],
	},
	{
		query: "NomaDamas 디스코드 서버에서 회의 일정 공지 찾아줘",
		searched: ["nomadamas"],
		skipped: ["mailcrawl", "kakao"],
	},
	{
		query: "Where is my 2025 tax return PDF?",
		searched: [],
		skipped: ["slack", "discord", "kakao", "telegram", "whatsapp", "nomadamas"],
	},
	// Topics no description mentions: the medium named in the question must
	// still be searched, because descriptions are non-exhaustive summaries.
	{
		query: "카톡에서 친구가 추천해준 제주도 맛집 이름 뭐였지?",
		searched: ["kakao"],
		skipped: ["slack", "nomadamas", "telegram"],
	},
	{
		query: "NomaDamas 디코에서 누가 생일이라고 했었지?",
		searched: ["nomadamas"],
		skipped: ["mailcrawl", "whatsapp"],
	},
	{
		query: "메일로 온 항공권 예약 확인서 찾아줘",
		searched: ["mailcrawl"],
		skipped: ["discord", "nomadamas", "slack", "telegram"],
	},
	// Descriptions also disambiguate two connections of the same medium.
	{
		query: "디스코드에서 공금 장부 잔액 얼마라고 했어?",
		searched: ["nomadamas"],
		skipped: ["mailcrawl", "whatsapp"],
	},
];

const runsFlag = process.argv.indexOf("--runs");
const runs = runsFlag >= 0 ? Number(process.argv[runsFlag + 1] ?? "1") : 1;
const printState = process.argv.includes("--state");

const config = resolveConfigReadOnly({ flags: {} });
const agent = new AutoRAGAgent({ ...buildAgentOptions(config), minSync: false, jikji: false });
// The judge is a private field; wrapping it records the exact state per call.
const internals = agent as unknown as {
	jevJudge: JevJudge | undefined;
	routingDiagnostics: { code: string; message: string }[];
	selectSearchDatasources(query: string): Promise<readonly string[]>;
};
const judge = internals.jevJudge;
if (judge === undefined) throw new Error("jev is disabled in the config");
let lastState = "";
internals.jevJudge = async (state, questions, options) => {
	lastState = String(state);
	return judge(state, questions, options);
};

let passed = 0;
let scored = 0;
for (let run = 1; run <= runs; run++) {
	for (const labeled of CASES) {
		internals.routingDiagnostics.length = 0;
		const started = Date.now();
		const selected = await internals.selectSearchDatasources(labeled.query);
		const elapsed = Date.now() - started;
		const diagnostic = internals.routingDiagnostics.find(
			(entry) => entry.code === "datasources-selected" || entry.code === "datasource-selection-fallback",
		);
		scored += 1;
		if (diagnostic?.code !== "datasources-selected") {
			console.log(`\n[run ${run}] FALLBACK ${labeled.query}: ${diagnostic?.message ?? "no diagnostic"}`);
			continue;
		}
		const misses = [
			...labeled.searched.filter((id) => !selected.includes(id)).map((id) => `missed ${id}`),
			...labeled.skipped.filter((id) => selected.includes(id)).map((id) => `wrongly searched ${id}`),
		];
		if (misses.length === 0) passed += 1;
		const memoryStart = lastState.indexOf("Similar past questions");
		const memoryBlock =
			memoryStart < 0
				? []
				: lastState
						.slice(memoryStart, lastState.indexOf("\n\nUser question:"))
						.split("\n")
						.slice(1);
		console.log(
			`\n[run ${run}] ${misses.length === 0 ? "PASS" : "FAIL"} (${elapsed}ms) ${labeled.query}\n  ${diagnostic.message}` +
				(memoryBlock.length > 0 ? `\n  memory:\n    ${memoryBlock.join("\n    ")}` : "") +
				(misses.length > 0 ? `\n  ${misses.join("; ")}` : ""),
		);
		if (printState && run === 1) console.log(`  --- state ---\n${lastState}\n  -------------`);
	}
}
console.log(`\n${passed}/${scored} cases passed`);
process.exit(0);
