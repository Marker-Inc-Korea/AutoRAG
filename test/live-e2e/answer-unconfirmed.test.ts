/**
 * Live E2E: when the corpus does not answer the question, the answer stays short and never pads itself with
 * unrelated or merely adjacent findings.
 *
 * Gated on AUTORAG_UNCONFIRMED_LIVE=1 plus an OpenRouter credential; the offline suite never calls a model.
 * The librarian, retrieval, the real MinSync index and the model are real. The corpus is Korean office
 * documents that sit next to every unanswerable question (same company, same kind of document, same
 * attribute for a different subject), so a model that pads has plenty of tempting material.
 *
 *   AUTORAG_UNCONFIRMED_LIVE=1 AUTORAG_UNCONFIRMED_LIVE_MODEL=qwen/qwen3.7-flash \
 *   AUTORAG_UNCONFIRMED_LIVE_REPS=3 AUTORAG_UNCONFIRMED_LIVE_OUT=/tmp/unconfirmed.json \
 *     bun run vitest run test/live-e2e/answer-unconfirmed.test.ts
 *
 * The model must be a 2026+ release (see the global model-usage rule). Every fast answer and final answer is
 * written to AUTORAG_UNCONFIRMED_LIVE_OUT (when set) so two builds can be compared run for run.
 */
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, delimiter, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";

const LIVE = process.env.AUTORAG_UNCONFIRMED_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_UNCONFIRMED_LIVE_MODEL ?? "qwen/qwen3.7-flash";
const REPS = Number(process.env.AUTORAG_UNCONFIRMED_LIVE_REPS ?? "1");
const OUT_PATH = process.env.AUTORAG_UNCONFIRMED_LIVE_OUT;

/** The developer's own `minsync`, skipping the inert stand-in the test setup prepends to PATH. */
function realMinSync(): string {
	const name = process.platform === "win32" ? "minsync.exe" : "minsync";
	for (const directory of (process.env.PATH ?? "").split(delimiter)) {
		if (directory === "" || basename(directory).startsWith("autorag-fake-minsync-path-")) continue;
		const candidate = join(directory, name);
		if (existsSync(candidate)) return candidate;
	}
	throw new Error("a real `minsync` on PATH is required for the live unconfirmed-answer test");
}

const CORPUS: Record<string, string> = {
	"maintenance-contract.md": [
		"# A사 소프트웨어 유지보수 계약",
		"",
		"계약기간은 2026-01-01부터 2026-12-31까지이다.",
		"월 유지보수료는 3,200,000원이다.",
		"A사 담당자는 김민수 과장이다.",
		"",
	].join("\n"),
	"office-lease.md": [
		"# 서울 본사 사무실 임대차",
		"",
		"서울 본사 사무실 임대차 계약은 2027-03-31에 갱신된다.",
		"월세는 4,800,000원이다.",
		"보증금은 96,000,000원이다.",
		"",
	].join("\n"),
	"travel-expense-policy.md": [
		"# 국내 출장비 정산 규정",
		"",
		"국내 출장 숙박비 상한은 1박당 150,000원이다.",
		"국내 출장 식비 상한은 하루 30,000원이다.",
		"출장비는 귀임 후 7일 이내에 정산한다.",
		"",
	].join("\n"),
	"security-audit-2026.md": [
		"# 2026 정보보안 점검 결과",
		"",
		"2026년 점검에서 취약점 7건이 발견되었다.",
		"그중 5건은 조치가 완료되었고 2건은 조치 중이다.",
		"",
	].join("\n"),
	"leave-policy.md": [
		"# 연차 휴가 규정",
		"",
		"입사 1년 미만 직원은 매월 개근 시 1일의 연차가 발생한다.",
		"입사 1년 이상 직원의 연차는 15일이다.",
		"",
	].join("\n"),
	"ops-meeting-2026-09.md": [
		"# 2026년 9월 운영회의록",
		"",
		"신규 개발자 2명 채용 안건을 승인했다.",
		"서버 증설은 다음 분기로 연기했다.",
		"",
	].join("\n"),
};

/** Distinctive values from the corpus. None of them answers an unanswerable question below. */
const DISTRACTOR_TOKENS = [
	"3,200,000",
	"김민수",
	"4,800,000",
	"96,000,000",
	"2027-03-31",
	"150,000",
	"30,000",
	"취약점 7건",
	"7건",
	"15일",
	"개발자 2명",
	"서버 증설",
	"A사",
] as const;

interface Case {
	readonly id: string;
	readonly query: string;
	/** The corpus holds no answer: the right answer is one short "not confirmed" line. */
	readonly answerable: boolean;
	/** Facts a correct answer must contain (answerable cases only). */
	readonly gold?: readonly RegExp[];
}

const CASES: readonly Case[] = [
	{ id: "other-vendor-penalty", query: "B사와의 유지보수 계약을 해지하면 위약금이 얼마야?", answerable: false },
	{ id: "other-site-rent", query: "부산 지사 사무실 월세가 얼마야?", answerable: false },
	{ id: "overseas-per-diem", query: "해외 출장 일비 상한은 얼마야?", answerable: false },
	{ id: "other-year-audit", query: "2025년 보안점검에서 발견된 취약점은 몇 건이야?", answerable: false },
	{ id: "unrelated-person", query: "대표이사 김영수의 작년 성과급은 얼마였어?", answerable: false },
	{
		id: "control-lease",
		query: "서울 본사 사무실 임대차 계약은 언제 갱신되고 월세는 얼마야?",
		answerable: true,
		gold: [/2027/u, /4,?800,?000/u],
	},
	{ id: "control-leave", query: "입사 1년 이상 직원의 연차는 며칠이야?", answerable: true, gold: [/15/u] },
];

let root: string;
let savedEnv: Record<string, string | undefined>;
const RESTORED_VARIABLES = ["PI_CODING_AGENT_DIR", "KIWI_LIBRARY_PATH", "KIWI_MODEL_PATH"] as const;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "arag-unconfirmed-live-"));
	savedEnv = Object.fromEntries(RESTORED_VARIABLES.map((name) => [name, process.env[name]]));
	delete process.env.KIWI_LIBRARY_PATH;
	delete process.env.KIWI_MODEL_PATH;
});

afterEach(() => {
	for (const name of RESTORED_VARIABLES) {
		const value = savedEnv[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
	rmSync(root, { recursive: true, force: true });
});

function leaked(answer: string): string[] {
	return DISTRACTOR_TOKENS.filter((token) => answer.includes(token));
}

function contentLines(answer: string): string[] {
	return answer.split("\n").filter((line) => line.trim() !== "");
}

interface Record_ {
	caseId: string;
	rep: number;
	model: string;
	fast: string | undefined;
	final: string;
	finalLeaks: string[];
	fastLeaks: string[];
	finalLines: number;
}

describe.skipIf(!LIVE)("unconfirmed answers, live model", () => {
	it("answers an unanswerable question in one short line and never pads with adjacent findings", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		for (const [name, text] of Object.entries(CORPUS)) writeFileSync(join(docs, name), text);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		process.env.PI_CODING_AGENT_DIR = agentDir;
		writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [docs] }));
		const config = resolveConfig({
			flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
			env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json") },
			cwd: root,
			readOnly: true,
		});
		const resolved = await resolveAgentModel(config);
		const minsyncBinary = realMinSync();
		const newAgent = (tag: string): AutoRAGAgent =>
			new AutoRAGAgent({
				model: resolved.model,
				...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
				...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
				searchPaths: [docs],
				workspacePath: root,
				// A fresh memory and session per question: each one is the first of its conversation.
				memoryPath: join(root, `memory-${tag}.json`),
				minSync: { autoInstall: false, binaryPath: minsyncBinary },
				jikji: false,
				everything: false,
				fsearch: false,
				webSearch: false,
				piAgentDir: agentDir,
				piSessionDir: join(root, `sessions-${tag}`),
			});

		const indexer = newAgent("index");
		const refreshed = await indexer.refresh(true, { methods: ["parsed", "minsync"] });
		expect(refreshed.minsync?.ok, JSON.stringify(refreshed.minsync)).toBe(true);

		const records: Record_[] = [];
		for (let rep = 0; rep < REPS; rep += 1) {
			for (const testCase of CASES) {
				const agent = newAgent(`${testCase.id}-${rep}`);
				let fast: string | undefined;
				let final = "";
				for await (const event of agent.searchDocumentsStream(testCase.query)) {
					if (event.type === "preliminary") fast = event.response.answer ?? "";
					if (event.type === "complete") final = event.response.answer ?? "";
				}
				const record: Record_ = {
					caseId: testCase.id,
					rep,
					model: MODEL_ID,
					fast,
					final,
					finalLeaks: testCase.answerable ? [] : leaked(final),
					fastLeaks: testCase.answerable || fast === undefined ? [] : leaked(fast),
					finalLines: contentLines(final).length,
				};
				records.push(record);
				console.log(
					`--- ${testCase.id} rep ${rep} (${MODEL_ID}) ---\nFAST: ${fast ?? "(none)"}\nFINAL: ${final}\nleaks fast=${JSON.stringify(record.fastLeaks)} final=${JSON.stringify(record.finalLeaks)} lines=${record.finalLines}\n--- end ---`,
				);
			}
		}
		if (OUT_PATH !== undefined) writeFileSync(OUT_PATH, JSON.stringify(records, null, 2));

		for (const record of records) {
			const testCase = CASES.find((candidate) => candidate.id === record.caseId) as Case;
			if (testCase.answerable) {
				// The rule must not over-suppress: a question the corpus answers still gets its facts.
				// The caller reads the fast answer and the verification delta together; the delta only confirms it.
				const shown = `${record.fast ?? ""}\n${record.final}`;
				for (const fact of testCase.gold ?? []) expect(shown, `${record.caseId} gold ${fact}`).toMatch(fact);
				continue;
			}
			expect(record.finalLeaks, `${record.caseId} final answer padded with adjacent findings`).toEqual([]);
			expect(record.fastLeaks, `${record.caseId} fast answer padded with adjacent findings`).toEqual([]);
			expect(record.finalLines, `${record.caseId} final answer is one short line`).toBeLessThanOrEqual(2);
			expect(record.final.trim(), `${record.caseId} final answer is not empty`).not.toBe("");
		}
	}, 1_800_000);
});
