import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";
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
		const entry = internals(agent).memory.append({ query: "q", method: "grep", outcome: "pending" });
		internals(agent).memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "grep",
			sources: ["src/a.ts"],
			timestamp: Date.now(),
		});

		agent.recordFeedbackByNumbers(sid, [1]);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("useful");
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
		const entry = internals(agent).memory.append({ query: "q", method: "grep", outcome: "pending" });
		internals(agent).memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "grep",
			sources: ["src/b.ts"],
			timestamp: Date.now(),
		});

		agent.recordFeedbackByNumbers(sid, [], [1]);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("not_useful");
	});

	it("ignores unknown session without error", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});

		expect(() => agent.recordFeedbackByNumbers("nonexistent", [99, 100])).not.toThrow();
	});
});
