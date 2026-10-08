// Live calibration of the Jev datasource check (selectDatasources) against the
// datasources registered in the real AutoRAG config.
//
//   OPENROUTER_API_KEY=... bun scripts/manual-qa/run-qa-jev-datasource-selection-live.ts [--runs N]
//
// Each case names the datasources that must be searched (>= 0.5) and the ones
// that must not (< 0.5); every other datasource is reported but not scored.
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { type DatasourceCandidate, selectDatasources } from "../../src/agent/query-routing.ts";
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
];

const runsFlag = process.argv.indexOf("--runs");
const runs = runsFlag >= 0 ? Number(process.argv[runsFlag + 1] ?? "1") : 1;

const config = resolveConfigReadOnly({ flags: {} });
const agent = new AutoRAGAgent({ ...buildAgentOptions(config), jev: false, minSync: false, jikji: false });
const searchable = new Set(
	agent
		.getMethodRegistry()
		.list()
		.map((method) => method.describe().datasourceId)
		.filter((datasourceId) => datasourceId !== undefined),
);
const candidates: DatasourceCandidate[] = agent
	.listDatasources()
	.filter((entry) => searchable.has(entry.datasourceId))
	.map(({ datasourceId, type, description }) => ({ datasourceId, type, description }));
console.log("Registered searchable datasources:");
for (const candidate of candidates) console.log(`  ${candidate.datasourceId} (${candidate.type}): ${candidate.description}`);

const judge = createJevJudge({ backend: "openrouter" });
let passed = 0;
let scored = 0;
for (let run = 1; run <= runs; run++) {
	for (const labeled of CASES) {
		const started = Date.now();
		const selection = await selectDatasources(judge, labeled.query, candidates);
		const elapsed = Date.now() - started;
		if (selection.fallbackReason !== undefined) {
			console.log(`\n[run ${run}] FALLBACK ${labeled.query}: ${selection.fallbackReason}`);
			scored += 1;
			continue;
		}
		const misses = [
			...labeled.searched.filter((id) => !selection.selected.includes(id)).map((id) => `missed ${id}`),
			...labeled.skipped.filter((id) => selection.selected.includes(id)).map((id) => `wrongly searched ${id}`),
		].filter((miss) => candidates.some((candidate) => miss.endsWith(` ${candidate.datasourceId}`)));
		scored += 1;
		if (misses.length === 0) passed += 1;
		const probabilities = Object.entries(selection.probabilities)
			.map(([id, probability]) => `${id}=${probability.toFixed(2)}`)
			.join(" ");
		console.log(
			`\n[run ${run}] ${misses.length === 0 ? "PASS" : "FAIL"} (${elapsed}ms) ${labeled.query}\n  selected: ${selection.selected.join(", ") || "none"}\n  p: ${probabilities}${misses.length > 0 ? `\n  ${misses.join("; ")}` : ""}`,
		);
	}
}
console.log(`\n${passed}/${scored} cases passed`);
