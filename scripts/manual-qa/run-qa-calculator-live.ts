/**
 * Live manual QA for the `calculator` agent tool.
 *
 * Drives a real AutoRAGAgent (same option path as the CLI, via buildAgentOptions) with a real model and an
 * isolated corpus, asks arithmetic, date, time, and time-zone questions, and records for every question which
 * calculator calls the model made, what the tool printed, and the final answer. Each case lists substrings the
 * answer must carry (from an independent reference) and the operations the model is expected to have used.
 *
 * Prerequisites: a resolvable model (OPENROUTER_API_KEY), and an isolated AUTORAG_HOME/AUTORAG_CONFIG that
 * already has MinSync in `<workspace>/.autorag/bin` (run `autorag refresh --method minsync` there first).
 *
 * Usage:
 *   AUTORAG_CALC_QA_ROOT=<isolated root> AUTORAG_CALC_QA_MODEL=anthropic/claude-haiku-5.5 \
 *     bun scripts/manual-qa/run-qa-calculator-live.ts
 *   AUTORAG_CALC_QA_EVIDENCE=.omo/evidence/qa-calculator-live.json  (default)
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { CALCULATOR_TOOL_NAME } from "../../src/agent/calculator-tool.ts";
import { buildAgentOptions, resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";

const root = process.env.AUTORAG_CALC_QA_ROOT;
if (root === undefined || process.env.AUTORAG_CONFIG === undefined || !process.env.AUTORAG_CONFIG.startsWith(root)) {
	throw new Error("Set AUTORAG_CALC_QA_ROOT and point AUTORAG_CONFIG/AUTORAG_HOME inside it (see AGENTS.md manual-QA rules).");
}
const modelId = process.env.AUTORAG_CALC_QA_MODEL ?? "anthropic/claude-haiku-5.5";
const evidencePath = process.env.AUTORAG_CALC_QA_EVIDENCE ?? join(".omo", "evidence", "qa-calculator-live.json");
const only = process.env.AUTORAG_CALC_QA_ONLY?.split(",").map(Number);

interface Case {
	readonly name: string;
	readonly query: string;
	/** Operations the model should have called. */
	readonly operations: readonly string[];
	/** Every one must appear in the answer. */
	readonly answerHas: readonly RegExp[];
	/** Must appear in the calculator output, proving the figure came from the tool. */
	readonly toolHas: readonly string[];
	/** Must appear in the first line of the answer. */
	readonly leadHas?: readonly RegExp[];
	/** Must not appear in the first line of the answer. */
	readonly leadLacks?: readonly RegExp[];
	/** No calculator call may carry a `timeZone` matching this. */
	readonly forbidZoneArg?: RegExp;
}


function leadOf(answer: string): string {
	return answer.split("\n").find((line) => line.trim() !== "") ?? "";
}
/** Expected values that depend on the clock, computed independently of the calculator with plain Date arithmetic. */
const TODAY = (() => {
	const parts = new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Seoul", year: "numeric", month: "2-digit", day: "2-digit" })
		.format(new Date())
		.split("-")
		.map(Number) as [number, number, number];
	const today = Date.UTC(parts[0], parts[1] - 1, parts[2]);
	const weekday = new Date(today).getUTCDay();
	return {
		weekdayEn: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][weekday] as string,
		weekdayKo: `${"일월화수목금토"[weekday]}요일`,
		daysToChristmas: Math.round((Date.UTC(2026, 11, 25) - today) / 86_400_000),
	};
})();

const CASES: readonly Case[] = [
	{
		name: "today's weekday (needs the clock)",
		query: "오늘 무슨 요일이야?",
		operations: ["now", "date_info"],
		answerHas: [new RegExp(`${TODAY.weekdayKo}|${TODAY.weekdayEn}`, "i")],
		toolHas: [`weekday: ${TODAY.weekdayEn}`],
	},
	{
		name: "days until a date from today",
		query: "오늘부터 2026년 12월 25일까지 며칠 남았어?",
		operations: ["date_diff"],
		answerHas: [new RegExp(`(?<!\\d)${TODAY.daysToChristmas}(?!\\d)`)],
		toolHas: [`totalDays: ${TODAY.daysToChristmas}`],
	},
	{
		name: "date plus N days with weekday",
		query: "2026-03-15에서 100일 뒤는 며칠이고 무슨 요일이야?",
		operations: ["date_add"],
		answerHas: [/2026-06-23|6월 23일|June 23/i, /화요일|Tuesday/i],
		toolHas: ["result: 2026-06-23", "weekday: Tuesday"],
	},
	{
		name: "leap-day plus a year (clamp)",
		query: "2024-02-29의 정확히 1년 뒤 날짜는?",
		operations: ["date_add"],
		answerHas: [/2025-02-28|2월 28일|February 28/i],
		toolHas: ["result: 2025-02-28"],
	},
	{
		name: "time zone conversion with the zone given",
		query: "서울(Asia/Seoul) 2026-11-08 오전 9시는 뉴욕에서 몇 시야? 날짜도 알려줘.",
		operations: ["convert_timezone"],
		answerHas: [/11-07|11월 7일|November 7|Nov 7/i, /19:00|7:00\s*PM|오후 7시/i],
		toolHas: ["2026-11-07T19:00:00-05:00"],
	},
	{
		name: "time zone conversion in the other direction across a date line",
		query: "LA(America/Los_Angeles) 2026-11-09 오전 9시를 한국 시간으로 바꿔줘.",
		operations: ["convert_timezone"],
		answerHas: [/11-10|11월 10일|November 10|Nov 10/i, /02:00|2:00|오전 2시|새벽 2시/i],
		toolHas: ["2026-11-10T02:00:00+09:00"],
	},
	{
		name: "business days with holidays and an inclusive end",
		query: "2026년 10월 1일부터 10월 31일까지(양끝 포함) 영업일은 며칠이야? 10월 5일과 10월 9일은 휴일이야.",
		operations: ["business_days_between"],
		answerHas: [/20/],
		toolHas: ["businessDays: 20"],
	},
	{
		name: "elapsed hours and minutes across midnight",
		query: "2026-11-08 22:10(Asia/Seoul)부터 2026-11-10 07:45(Asia/Seoul)까지 총 몇 분 지났어?",
		operations: ["date_diff"],
		answerHas: [/2,?015/],
		toolHas: ["elapsedSeconds: 120900"],
	},
	{
		name: "exact money arithmetic (VAT)",
		query: "공급가액 1,234,567원에 부가세 10%를 더한 합계는 정확히 얼마야?",
		operations: ["evaluate"],
		answerHas: [/1,?358,?023\.7|1,?358,?023(?!\d)/],
		toolHas: ["1358023.7"],
	},
	{
		name: "compound interest to the cent",
		query: "1000달러를 연 7% 복리로 10년 굴리면 센트 단위까지 정확히 얼마야?",
		operations: ["evaluate"],
		answerHas: [/1,?967\.15/],
		toolHas: ["1967.15"],
	},
	{
		name: "statistics (mean and sample stdev)",
		query: "점수가 82, 91, 77, 68, 95, 88일 때 평균과 표본표준편차(소수 6자리)를 알려줘.",
		operations: ["evaluate"],
		answerHas: [/83\.5/, /9\.9347/],
		toolHas: ["83.5", "9.9347"],
	},
	{
		name: "unit conversion with a constant",
		query: "마라톤 42.195km는 마일로 몇 마일이야? 소수 넷째 자리까지. (1마일 = 1.609344km)",
		operations: ["evaluate"],
		answerHas: [/26\.2188/],
		toolHas: ["26.218"],
	},
	{
		name: "ambiguous time-zone abbreviation must not be guessed",
		query: "PST 2026-11-09 오후 3시를 한국 시간으로 바꿔줘.",
		operations: ["convert_timezone"],
		answerHas: [/11-10|11월 10일|November 10|Nov 10/i, /08:00|8:00|오전 8시|아침 8시/i],
		toolHas: ["2026-11-10T08:00:00+09:00"],
	},
	{
		name: "daylight-saving gap: two elapsed hours after 01:30",
		query: "America/New_York 2026-03-08 01:30에서 정확히 2시간(경과 시간) 뒤는 몇 시야?",
		operations: ["date_add"],
		answerHas: [/04:30|4:30|오전 4시 30분/i],
		toolHas: ["2026-03-08T04:30:00-04:00"],
	},
	{
		name: "no tool needed: plain knowledge question",
		query: "대한민국의 수도는 어디야?",
		operations: [],
		answerHas: [/서울/],
		toolHas: [],
	},
	{
		// In July the Pacific region is on PDT (UTC-7), so "PST" (UTC-8, all year) and America/Los_Angeles differ by an hour.
		// The model must pass the offset PST stands for, never swap in a region that follows daylight saving.
		name: "summer PST: an abbreviation is a fixed offset, never swapped for a daylight-saving region",
		query: "PST 2026-07-01 오후 3시를 한국 시간으로 바꿔줘.",
		operations: ["convert_timezone"],
		answerHas: [/07-02|7월 2일|July 2|Jul 2/i, /08:00|8:00|오전 8시|아침 8시/i],
		leadHas: [/08:00|8:00|오전 8시|아침 8시/i],
		leadLacks: [/07:00|(?<!\d)7:00|오전 7시|아침 7시/i],
		toolHas: ["2026-07-02T08:00:00+09:00"],
		forbidZoneArg: /\//,
	},
];

interface Recorded {
	readonly args: Record<string, unknown>;
	readonly output: string;
	readonly isError: boolean;
}

function textOf(content: unknown): string {
	if (!Array.isArray(content)) return "";
	return content
		.map((part) => (typeof part === "object" && part !== null && "text" in part && typeof part.text === "string" ? part.text : ""))
		.join("");
}

interface AgentInternals {
	readonly tools: { name: string; execute: (...args: never[]) => Promise<{ content: unknown; isError?: boolean }> }[];
}

const docs = join(root, "docs");
const config = resolveConfig({ flags: { "model-provider": "openrouter", "model-id": modelId }, cwd: root });
const isolated = { ...config, searchPaths: [docs], workspacePath: root, memoryPath: join(root, "memory.json") };
const resolved = await resolveAgentModel(isolated);

const results: unknown[] = [];
let failed = 0;
for (const [index, testCase] of CASES.entries()) {
	if (only !== undefined && !only.includes(index)) continue;
	const agent = new AutoRAGAgent({
		...buildAgentOptions(isolated),
		model: resolved.model,
		...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
		...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
	});
	const calls: Recorded[] = [];
	const tool = (agent as unknown as AgentInternals).tools.find((candidate) => candidate.name === CALCULATOR_TOOL_NAME);
	if (tool === undefined) throw new Error("the agent registered no calculator tool");
	const real = tool.execute as (id: string, params: Record<string, unknown>, ...rest: never[]) => Promise<{ content: unknown; isError?: boolean }>;
	tool.execute = (async (id: string, params: Record<string, unknown>, ...rest: never[]) => {
		const result = await real(id, params, ...rest);
		calls.push({ args: params, output: textOf(result.content), isError: result.isError === true });
		return result;
	}) as never;

	const started = Date.now();
	let answer = "";
	let error: string | undefined;
	try {
		for await (const event of agent.searchDocumentsStream(testCase.query, {})) {
			if (event.type === "complete") answer = event.response.answer;
		}
	} catch (caught) {
		error = caught instanceof Error ? caught.message : String(caught);
	}

	const used = [...new Set(calls.map((call) => String(call.args.operation)))];
	const toolText = calls.map((call) => call.output).join("\n");
	const checks = {
		// A case that lists no operations is a control: the model must answer without the calculator.
		calledCalculator: testCase.operations.length === 0 ? calls.length === 0 : calls.length > 0,
		usedExpectedOperation: testCase.operations.length === 0 || testCase.operations.some((operation) => used.includes(operation)),
		toolHasFigure: testCase.toolHas.every((needle) => toolText.includes(needle)),
		answerHasFigure: testCase.answerHas.every((pattern) => pattern.test(answer)),
		// The first line is what the reader takes away; an alternative mentioned further down must not rescue a wrong headline.
		leadHasFigure: (testCase.leadHas ?? []).every((pattern) => pattern.test(leadOf(answer))),
		leadLacksWrongFigure: (testCase.leadLacks ?? []).every((pattern) => !pattern.test(leadOf(answer))),
		noForbiddenZoneArg: testCase.forbidZoneArg === undefined || calls.every((call) => !testCase.forbidZoneArg?.test(String(call.args.timeZone ?? ""))),
		noToolErrors: calls.every((call) => !call.isError),
	};
	const pass = error === undefined && Object.values(checks).every(Boolean);
	if (!pass) failed++;
	results.push({ index, name: testCase.name, query: testCase.query, pass, checks, used, calls, answer, error, seconds: (Date.now() - started) / 1000 });
	console.log(`${pass ? "PASS" : "FAIL"} [${index}] ${testCase.name}  (${used.join(", ") || "no calculator call"}, ${calls.length} call(s))`);
	if (!pass) console.log(`     checks: ${JSON.stringify(checks)}${error === undefined ? "" : `  error: ${error}`}`);
}

mkdirSync(dirname(evidencePath), { recursive: true });
writeFileSync(evidencePath, JSON.stringify({ model: modelId, results }, null, 2));
console.log(`\n${results.length - failed}/${results.length} passed; evidence: ${evidencePath}`);
process.exit(failed === 0 ? 0 : 1);
