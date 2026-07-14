import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runFeedback } from "../../src/cli/commands/feedback.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";

let tmpDir: string;
let previousHome: string | undefined;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-cli-feedback-"));
	previousHome = process.env.HOME;
	process.env.HOME = join(tmpDir, "home");
});

afterEach(() => {
	if (previousHome === undefined) delete process.env.HOME;
	else process.env.HOME = previousHome;
	rmSync(tmpDir, { recursive: true, force: true });
});

interface Captured {
	readonly stdout: string[];
	readonly stderr: string[];
	readonly ctx: CommandContext;
}

function makeCtx(opts: {
	positionals?: string[];
	flags?: Record<string, string | boolean | undefined>;
	json?: boolean;
	cwd?: string;
}): Captured {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals: opts.positionals ?? [],
		flags: opts.flags ?? {},
		json: opts.json ?? false,
		debug: false,
		cwd: opts.cwd ?? tmpDir,
		stdout: (line: string) => {
			stdout.push(line);
		},
		stderr: (line: string) => {
			stderr.push(line);
		},
	};
	return { stdout, stderr, ctx };
}

// resolveConfig defaults memoryPath to ~/.autorag/memory.json; seed there.
function defaultMemoryPath(): string {
	return join(process.env.HOME as string, ".autorag", "memory.json");
}

function seedMemory(memoryPath: string, sessionId: string, query: string): void {
	mkdirSync(dirname(memoryPath), { recursive: true });
	const mem = new RetrievalMemory({ storagePath: memoryPath });
	mem.load();
	mem.recordCuratedResultsSession({
		sessionId,
		query,
		results: [
			{
				number: 1,
				title: "First result",
				summary: "Useful summary",
				content: "content one",
				method: "grep",
				source: "src/a.ts",
				evidenceRefs: [normalizeSessionEvidenceRef({ method: "grep", source: "src/a.ts", content: "content one" })],
			},
			{
				number: 2,
				title: "Second result",
				summary: "Not useful summary",
				content: "content two",
				method: "grep",
				source: "src/b.ts",
				evidenceRefs: [normalizeSessionEvidenceRef({ method: "grep", source: "src/b.ts", content: "content two" })],
			},
		],
	});
	mem.save();
}

function loadFresh(memoryPath: string): RetrievalMemory {
	const mem = new RetrievalMemory({ storagePath: memoryPath });
	mem.load();
	return mem;
}

describe("runFeedback", () => {
	it("records useful and not-useful numbered feedback against a seeded session", async () => {
		const sessionId = "session-seeded";
		const memoryPath = defaultMemoryPath();
		seedMemory(memoryPath, sessionId, "q");

		const before = loadFresh(memoryPath).getSignalCount();
		const { ctx, stdout } = makeCtx({
			positionals: [sessionId],
			flags: { useful: "1", "not-useful": "2" },
		});

		const code = await runFeedback(ctx);
		expect(code).toBe(0);
		expect(stdout.length).toBe(1);

		const reloaded = loadFresh(memoryPath);
		const signals = reloaded.getSchema().feedbackSignals.filter((s) => s.source === "explicit");
		expect(reloaded.getSignalCount()).toBeGreaterThan(before);

		const usefulSignal = signals.find((s) => s.sentiment === "useful");
		expect(usefulSignal).toBeDefined();
		expect(usefulSignal?.eventId).toBe(`${sessionId}:1:useful`);
		expect(usefulSignal?.target).toEqual({ type: "curated_result", resultId: `${sessionId}:1` });

		const notUsefulSignal = signals.find((s) => s.sentiment === "not_useful");
		expect(notUsefulSignal).toBeDefined();
		expect(notUsefulSignal?.eventId).toBe(`${sessionId}:2:not_useful`);
		expect(notUsefulSignal?.target).toEqual({ type: "curated_result", resultId: `${sessionId}:2` });
	});

	it("returns exit 2 and writes no new signals for an unknown session", async () => {
		const sessionId = "session-seeded";
		const memoryPath = defaultMemoryPath();
		seedMemory(memoryPath, sessionId, "q");

		const before = loadFresh(memoryPath).getSignalCount();
		const { ctx, stdout } = makeCtx({
			positionals: ["nonexistent-session"],
			flags: { useful: "1" },
			json: true,
		});

		const code = await runFeedback(ctx);
		expect(code).toBe(2);

		const reloaded = loadFresh(memoryPath);
		expect(reloaded.getSignalCount()).toBe(before);
		expect(stdout.length).toBe(1);
		const parsed = JSON.parse(stdout[0]);
		expect(parsed.applied).toBe(false);
		expect(parsed.sessionId).toBe("nonexistent-session");
	});

	it("returns exit 2 with a usage error when no session id is given", async () => {
		const { ctx, stderr } = makeCtx({ flags: { useful: "1" } });
		const code = await runFeedback(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("Usage");
	});

	it("returns exit 2 when neither --useful nor --not-useful is provided", async () => {
		const { ctx, stderr } = makeCtx({ positionals: ["session-x"] });
		const code = await runFeedback(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("Usage");
	});

	it("emits a json envelope when --json is set", async () => {
		const sessionId = "session-json";
		const memoryPath = defaultMemoryPath();
		seedMemory(memoryPath, sessionId, "q");

		const { ctx, stdout } = makeCtx({
			positionals: [sessionId],
			flags: { useful: "1" },
			json: true,
		});

		const code = await runFeedback(ctx);
		expect(code).toBe(0);
		const parsed = JSON.parse(stdout[0]);
		expect(parsed.ok).toBe(true);
		expect(parsed.applied).toBe(true);
		expect(parsed.sessionId).toBe(sessionId);
	});
});
