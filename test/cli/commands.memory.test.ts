import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runMemory } from "../../src/cli/commands/memory.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";
import { RankingLogStore, RankingRunRecorder, rankingLogPath } from "../../src/memory/ranking-log.ts";
import { evidenceChunkKey } from "../../src/retrieval/evidence-id.ts";

let root: string;
let memoryPath: string;
let previousHome: string | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-memory-"));
	previousHome = process.env.HOME;
	process.env.HOME = join(root, "home");
	memoryPath = join(root, "memory.json");
});

afterEach(() => {
	if (previousHome === undefined) delete process.env.HOME;
	else process.env.HOME = previousHome;
	rmSync(root, { recursive: true, force: true });
});

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: ["inspect"],
		flags: {},
		json: true,
		debug: false,
		cwd: root,
		stdout: () => {},
		stderr: () => {},
		...overrides,
	};
}

function writeConfig(): void {
	const config = {
		searchPaths: ["docs"],
		workspacePath: root,
		memoryPath,
	};
	const configDir = join(process.env.HOME as string, ".autorag");
	mkdirSync(configDir, { recursive: true });
	writeFileSync(join(configDir, "config.json"), `${JSON.stringify(config, null, 2)}\n`);
}

const SESSION_ID = "sess-001";
const QUERY = "how does bm25 rank documents";
const SOURCE = "docs/ranking.md";
const EXCERPT = "BM25 is a probabilistic ranking function.";

function seedMemory(): void {
	const mem = new RetrievalMemory({ storagePath: memoryPath });
	mem.load();
	mem.recordCuratedResultsSession({
		sessionId: SESSION_ID,
		query: QUERY,
		results: [
			{
				number: 1,
				title: "BM25 ranking primer",
				summary: "BM25 scores by IDF-weighted term frequency and document length.",
				content: EXCERPT,
				method: "bm25",
				source: SOURCE,
				evidenceRefs: [normalizeSessionEvidenceRef({ method: "bm25", source: SOURCE, excerpt: EXCERPT })],
			},
		],
	});
	mem.save();
}

describe("runMemory inspect (cli)", () => {
	beforeEach(() => {
		// config + memory live under the same temp root.
		writeConfig();
		seedMemory();
	});

	it("emits a JSON envelope with counts, judged evidence, and insights", async () => {
		const stdout: string[] = [];
		const code = await runMemory(makeCtx({ stdout: (line) => stdout.push(line) }));

		expect(code).toBe(0);
		expect(stdout).toHaveLength(1);

		const schema = JSON.parse(stdout[0]);
		expect(schema.version).toBe(5);
		expect(Array.isArray(schema.curatedResults)).toBe(true);
		expect(Array.isArray(schema.evidenceChunks)).toBe(true);
		expect(Array.isArray(schema.judgedEvidence)).toBe(true);
		expect(Array.isArray(schema.insights)).toBe(true);
		expect(Array.isArray(schema.pendingInsightEntries)).toBe(true);
		expect(Array.isArray(schema.warnings)).toBe(true);
		expect(schema.curatedResults.length).toBeGreaterThanOrEqual(1);

		// The curated result is present.
		const curated = schema.curatedResults.find(
			(entry: { sessionId?: string; number?: number }) => entry.sessionId === SESSION_ID && entry.number === 1,
		);
		expect(curated).toBeDefined();
		expect(curated.query).toBe(QUERY);
	});

	it("does not leak the memory storage path", async () => {
		const stdout: string[] = [];
		await runMemory(makeCtx({ stdout: (line) => stdout.push(line) }));

		const blob = stdout[0];
		expect(blob).not.toContain(memoryPath);
		expect(blob).not.toContain(root);
	});

	it("returns 2 with a usage error when the subcommand is missing or unknown", async () => {
		const stderr: string[] = [];
		const codeMissing = await runMemory(makeCtx({ positionals: [], stderr: (line) => stderr.push(line) }));
		expect(codeMissing).toBe(2);
		expect(stderr).toHaveLength(1);

		stderr.length = 0;
		const codeUnknown = await runMemory(makeCtx({ positionals: ["bogus"], stderr: (line) => stderr.push(line) }));
		expect(codeUnknown).toBe(2);
		expect(stderr).toHaveLength(1);
	});

	it("emits a non-json human rendering when --json is unset", async () => {
		const stdout: string[] = [];
		const code = await runMemory(makeCtx({ json: false, stdout: (line) => stdout.push(line) }));
		expect(code).toBe(0);
		expect(stdout).toHaveLength(1);
		// Human mode is not a JSON object literal.
		expect(stdout[0].trimStart()[0]).not.toBe("{");
	});
});

describe("runMemory export-ltr (cli)", () => {
	function seedRankingLog(count: number): void {
		const store = new RankingLogStore(rankingLogPath(memoryPath));
		for (let index = 0; index < count; index++) {
			const run = new RankingRunRecorder({
				runId: `run-${index}`,
				conversationId: "c",
				question: "who approved the budget?",
				startedAt: 100 + index,
			});
			const cited = {
				id: "a",
				source: "/docs/budget.md",
				content: "Mina approved the budget.",
				score: 1,
				metadata: {},
			};
			const other = { id: "b", source: "/docs/other.md", content: "Lunch menu.", score: 0.5, metadata: {} };
			run.retrieved({ surface: "minsync", query: "who approved the budget?", results: [cited, other] });
			store.append(
				run.finish({ finalCitationKeys: [evidenceChunkKey(cited.source, cited.content)], finishedAt: 200 }),
			);
		}
		store.close();
	}

	beforeEach(() => {
		writeConfig();
	});

	it("writes LightGBM files for the logged runs into --output and reports the split", async () => {
		seedRankingLog(5);
		const output = join(root, "ltr");
		const stdout: string[] = [];
		const code = await runMemory(
			makeCtx({ positionals: ["export-ltr"], flags: { output }, stdout: (line) => stdout.push(line) }),
		);

		expect(code).toBe(0);
		const summary = JSON.parse(stdout[0] as string);
		expect(summary).toMatchObject({ runs: 5, train: { groups: 4, rows: 8 }, valid: { groups: 1, rows: 2 } });
		expect(readFileSync(join(output, "train.tsv.query"), "utf-8")).toBe("2\n2\n2\n2\n");
		expect(readFileSync(join(output, "valid.tsv.query"), "utf-8")).toBe("2\n");
	});

	it("exports an empty, valid manifest when nothing was logged yet, without creating the log", async () => {
		const output = join(root, "ltr");
		const stdout: string[] = [];
		const code = await runMemory(
			makeCtx({ positionals: ["export-ltr"], flags: { output }, stdout: (line) => stdout.push(line) }),
		);

		expect(code).toBe(0);
		expect(JSON.parse(stdout[0] as string)).toMatchObject({ runs: 0, train: { groups: 0 } });
		expect(existsSync(rankingLogPath(memoryPath))).toBe(false);
	});

	it("requires --output and rejects a validation fraction outside [0, 1)", async () => {
		const stderr: string[] = [];
		expect(await runMemory(makeCtx({ positionals: ["export-ltr"], stderr: (line) => stderr.push(line) }))).toBe(2);
		expect(
			await runMemory(
				makeCtx({
					positionals: ["export-ltr"],
					flags: { output: join(root, "x"), "valid-fraction": "1.5" },
					stderr: (line) => stderr.push(line),
				}),
			),
		).toBe(2);
		expect(stderr.join("\n")).toMatch(/--output/u);
		expect(stderr.join("\n")).toMatch(/valid-fraction/u);
	});
});
