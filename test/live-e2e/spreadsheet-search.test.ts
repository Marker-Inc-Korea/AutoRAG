/**
 * Live E2E: spreadsheets through the real CLI, the parse worker, MinSync and its local embedder.
 *
 * Gated on AUTORAG_SPREADSHEET_LIVE=1 so the offline suite never needs MinSync or a model. Run
 * explicitly with `bun` and `minsync` on PATH (a running `minsync serve` daemon or the in-process
 * addon provides Qwen3 embeddings):
 *
 *   AUTORAG_SPREADSHEET_LIVE=1 bun run vitest run test/live-e2e/spreadsheet-search.test.ts
 *
 * With OPENROUTER_API_KEY also set, the agent search case asks the issue #1684 question through a
 * real model (AUTORAG_SPREADSHEET_LIVE_MODEL, default anthropic/claude-haiku-5.5) and expects the
 * cached formula value 742,000 in the answer.
 *
 * Every command runs with AUTORAG_HOME and AUTORAG_CONFIG inside a temp root, so the real
 * ~/.autorag is never read or written.
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
	createBudgetWorkbook,
	createDetailWorkbook,
	createFormWorkbook,
	createStackedWorkbook,
	createWideWorkbook,
	writeWorkbook,
} from "../fixtures/spreadsheets.ts";

const LIVE = process.env.AUTORAG_SPREADSHEET_LIVE === "1";
const MODEL_ID = process.env.AUTORAG_SPREADSHEET_LIVE_MODEL ?? "anthropic/claude-haiku-5.5";
const CLI = fileURLToPath(new URL("../../src/cli/index.ts", import.meta.url));

interface RetrievedResult {
	readonly content: string;
	readonly source: string;
}

describe.skipIf(!LIVE)("live: spreadsheets are searchable with their labels", () => {
	let root: string;
	let env: NodeJS.ProcessEnv;

	function cli(args: readonly string[], timeoutMs = 600_000): string {
		const result = spawnSync("bun", ["run", CLI, ...args], { cwd: root, env, encoding: "utf8", timeout: timeoutMs });
		if (result.status !== 0) throw new Error(`autorag ${args.join(" ")} exited ${result.status}: ${result.stderr}`);
		return result.stdout;
	}

	function retrieve(query: string): RetrievedResult[] {
		const output = cli(["lite", "retrieve", query, "--json", "--top-k", "5"]);
		const parsed = JSON.parse(output) as { readonly results?: readonly RetrievedResult[] };
		if (parsed.results === undefined || parsed.results.length === 0)
			throw new Error(`no results for ${query}: ${output}`);
		return [...parsed.results];
	}

	beforeAll(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-spreadsheet-live-"));
		const home = join(root, ".autorag-home");
		// The suite setup puts an inert `minsync` first on PATH and points Kiwi at empty stand-in files; this
		// live run needs the real MinSync and its real Korean tokenizer.
		const { KIWI_LIBRARY_PATH: _library, KIWI_MODEL_PATH: _model, ...inherited } = process.env;
		const path = (process.env.PATH ?? "")
			.split(delimiter)
			.filter((entry) => !entry.includes("autorag-fake-minsync-path-"))
			.join(delimiter);
		env = { ...inherited, PATH: path, AUTORAG_HOME: home, AUTORAG_CONFIG: join(home, "config.json") };
		mkdirSync(join(root, "docs"), { recursive: true });
		mkdirSync(join(root, "ws"), { recursive: true });
		mkdirSync(home, { recursive: true });
		writeFileSync(join(root, "docs", "연구비 예산.xlsx"), writeWorkbook(createBudgetWorkbook()));
		writeFileSync(join(root, "docs", "세부예산.xlsx"), writeWorkbook(createDetailWorkbook()));
		writeFileSync(join(root, "docs", "신청서.xls"), writeWorkbook(createFormWorkbook(), "xls"));
		writeFileSync(join(root, "docs", "출장.ods"), writeWorkbook(createStackedWorkbook(), "ods"));
		writeFileSync(join(root, "docs", "세부사업별.xlsb"), writeWorkbook(createWideWorkbook(), "xlsb"));
		const modelArgs = process.env.OPENROUTER_API_KEY
			? ["--model-provider", "openrouter", "--model-id", MODEL_ID]
			: [];
		cli(["init", "--search-paths", join(root, "docs"), "--workspace", join(root, "ws"), ...modelArgs, "--force"]);
		const refresh = JSON.parse(cli(["lite", "refresh", "--method", "parsed,minsync", "--json"])) as {
			readonly ok: boolean;
			readonly counts?: { readonly written: number; readonly skipped: number };
		};
		expect(refresh.ok, JSON.stringify(refresh)).toBe(true);
		expect(refresh.counts).toMatchObject({ written: 5, skipped: 0 });
	}, 900_000);

	afterAll(() => rmSync(root, { recursive: true, force: true }));

	it.each([
		[
			"인건비 편성액은 얼마인가?",
			"연구비 예산.xlsx",
			"비목: 인건비 | 산출근거: 책임 1명 × 12월 / 참여 2명 × 12월 | 편성액: 742,000",
		],
		[
			"외부전문가활용비 편성액",
			"세부예산.xlsx",
			"비목: 외부전문가활용비 | 세목: 자문료 | 산출근거: 자문 3회 × 105,000 | 편성액: 315000",
		],
		[
			"해외출장 일본 체재비",
			"출장.ods",
			"[출장!A9:D9 · 해외출장] 일자: 2026-05-20 | 국가: 일본 | 항공료: 450000 | 체재비: 630000",
		],
		["연구책임자는 누구인가", "신청서.xls", "주관기관 | 마커 | 연구책임자 | 홍길동"],
		["인건비 세부사업237 집행액", "세부사업별.xlsb", "세부사업237 집행액: 4321987"],
		["직원30 항목18", "연구비 예산.xlsx", "| 항목18: 987654 |"],
	])("ranks the row that answers %j first, with its labels, from %s", (query, file, expected) => {
		const [top] = retrieve(query);

		expect(top?.content).toContain(expected);
		expect(top?.source.endsWith(file)).toBe(true);
		expect(isAbsolute(top?.source ?? "")).toBe(true);
		expect(existsSync(top?.source ?? "")).toBe(true);
	});

	it("keeps the row key on every piece of a row split for being too wide", () => {
		const hit = retrieve("인건비 세부사업237 집행액")[0];
		const line = hit?.content.split("\n").find((candidate) => candidate.includes("세부사업237 집행액: 4321987"));

		expect(line).toMatch(/^\[세부사업별![A-Z]+2:[A-Z]+2\] 구분: 인건비 \| /);
	});

	it.skipIf(!process.env.OPENROUTER_API_KEY)(
		"answers the issue #1684 question with the cached formula value through a live model",
		() => {
			const output = cli(["search", "--json", "인건비 편성액은 얼마인가?"]);
			// Progress events are one-line JSON; the final response is one pretty-printed JSON document.
			const final = output.slice(output.lastIndexOf("\n{\n") + 1);
			const response = JSON.parse(final) as {
				readonly answer: string;
				readonly results: readonly { source: string }[];
			};

			expect(response.answer).toMatch(/742,000/);
			expect(response.results.some((result) => result.source.endsWith("연구비 예산.xlsx"))).toBe(true);
		},
		600_000,
	);
});
