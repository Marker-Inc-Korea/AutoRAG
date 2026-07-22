import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runMemory } from "../../src/cli/commands/memory.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";

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
	// Record explicit numbered feedback so a feedback signal materializes against the curated result.
	const applied = mem.recordNumberedFeedback({
		sessionId: SESSION_ID,
		query: "",
		feedback: [{ number: 1, useful: true }],
	});
	expect(applied).toBe(true);
	mem.save();
}

describe("runMemory inspect (cli)", () => {
	beforeEach(() => {
		// config + memory live under the same temp root.
		writeConfig();
		seedMemory();
	});

	it("emits a JSON envelope with counts and signal defaults", async () => {
		const stdout: string[] = [];
		const code = await runMemory(makeCtx({ stdout: (line) => stdout.push(line) }));

		expect(code).toBe(0);
		expect(stdout).toHaveLength(1);

		const schema = JSON.parse(stdout[0]);
		expect(schema.version).toBe(4);
		expect(Array.isArray(schema.curatedResults)).toBe(true);
		expect(Array.isArray(schema.feedbackSignals)).toBe(true);
		expect(Array.isArray(schema.insights)).toBe(true);
		expect(schema.curatedResults.length).toBeGreaterThanOrEqual(1);
		expect(schema.feedbackSignals.length).toBeGreaterThanOrEqual(1);

		expect(schema.signalDefaults).toBeDefined();
		expect(typeof schema.signalDefaults.explicitWeight).toBe("number");
		expect(typeof schema.signalDefaults.followupWeight).toBe("number");
		expect(typeof schema.signalDefaults.retryWeight).toBe("number");
		expect(typeof schema.signalDefaults.implicitCap).toBe("number");

		// The curated result and the explicit useful signal are present.
		const curated = schema.curatedResults.find(
			(entry: { sessionId?: string; number?: number }) => entry.sessionId === SESSION_ID && entry.number === 1,
		);
		expect(curated).toBeDefined();
		expect(curated.query).toBe(QUERY);
		const usefulSignal = schema.feedbackSignals.find(
			(signal: { sentiment?: string; source?: string }) =>
				signal.sentiment === "useful" && signal.source === "explicit",
		);
		expect(usefulSignal).toBeDefined();
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
