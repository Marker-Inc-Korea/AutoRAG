import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";
import type { CuratedResult } from "../../src/retrieval/types.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-feedback-numbers-test-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

interface AgentInternals {
	readonly memory: RetrievalMemory;
	readonly sessions: Map<string, { query: string; registry: Map<number, CuratedResult> }>;
}

function internals(agent: AutoRAGAgent): AgentInternals {
	return agent as unknown as AgentInternals;
}
function seedCuratedResult(memory: RetrievalMemory, sessionId: string, source: string): void {
	memory.recordCuratedResultsSession({
		sessionId,
		query: "q",
		results: [
			{
				number: 1,
				title: "Result",
				summary: "Summary",
				content: "content",
				method: "grep",
				source,
				evidenceRefs: [normalizeSessionEvidenceRef({ method: "grep", source, content: "content" })],
			},
		],
	});
}

describe("AutoRAGAgent numbered feedback", () => {
	it("exposes recordFeedbackByNumbers as a public method", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});

		expect(typeof agent.recordFeedbackByNumbers).toBe("function");
	});

	it("resolves useful entries by number with session", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		const sid = "test-session-1";
		const reg = new Map<number, CuratedResult>();
		reg.set(1, { index: 1, source: "src/a.ts", content: "", method: "grep" });
		internals(agent).sessions.set(sid, { query: "q", registry: reg });
		seedCuratedResult(internals(agent).memory, sid, "src/a.ts");

		agent.recordFeedbackByNumbers(sid, [1]);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getMethodHints("q").find((hint) => hint.method === "grep")?.score).toBeGreaterThan(0);
	});

	it("resolves not-useful entries by number with session", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		const sid = "test-session-2";
		const reg = new Map<number, CuratedResult>();
		reg.set(1, { index: 1, source: "src/b.ts", content: "", method: "grep" });
		internals(agent).sessions.set(sid, { query: "q", registry: reg });
		seedCuratedResult(internals(agent).memory, sid, "src/b.ts");

		agent.recordFeedbackByNumbers(sid, [], [1]);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getMethodHints("q").find((hint) => hint.method === "grep")?.score).toBeLessThan(0);
	});

	it("ignores unknown session without error", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});

		expect(() => agent.recordFeedbackByNumbers("nonexistent", [99, 100])).not.toThrow();
	});
});
