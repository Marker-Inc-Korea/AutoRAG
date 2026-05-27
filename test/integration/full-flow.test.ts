import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";
import type { CuratedResult } from "../../src/retrieval/types.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

interface AgentInternals {
	lastQuery: string | undefined;
	memory: RetrievalMemory;
	sessions: Map<string, { query: string; registry: Map<number, CuratedResult> }>;
}

function internals(agent: AutoRAGAgent): AgentInternals {
	return agent as unknown as AgentInternals;
}

beforeEach(() => {
	tmpDir = join(tmpdir(), `autorag-integration-${Date.now()}`);
	mkdirSync(tmpDir, { recursive: true });
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("Full flow integration", () => {
	it("agent creates with manifests and includes them in system prompt", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			manifestDir: "test/fixtures/manifests",
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("codebase-vectors");
		expect(prompt).toContain("documentation-index");
	});

	it("feedback → memory → priority flow works end-to-end", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});

		internals(agent).lastQuery = "search typescript code";
		internals(agent).memory.append({ query: "search typescript code", method: "posix", outcome: "pending" });
		internals(agent).memory.append({ query: "search typescript code", method: "posix", outcome: "pending" });
		internals(agent).memory.append({ query: "search typescript code", method: "posix", outcome: "pending" });
		agent.submitFeedback(undefined, true);

		expect(existsSync(memPath)).toBe(true);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		const priority = memory.getMethodPriority("search typescript code");
		expect(priority.length).toBeGreaterThan(0);
		expect(priority[0].method).toBe("posix");
		expect(priority[0].score).toBe(1.0);
	});

	it("agent runs without built-in retrieval methods", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(agent.getSystemPrompt()).toContain("check_memory");
		expect(agent.getSystemPrompt()).not.toContain("search_posix");
	});

	it("submitFeedback resolves pending entries across all methods", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		internals(agent).lastQuery = "cold start query";
		internals(agent).memory.append({ query: "cold start query", method: "posix", outcome: "pending" });
		internals(agent).memory.append({ query: "cold start query", method: "vector", outcome: "pending" });
		agent.submitFeedback(undefined, false);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		const entries = memory.getEntries();
		expect(entries.length).toBe(2);
		expect(entries.every((e) => e.outcome === "not_useful")).toBe(true);
	});

	it("agent system prompt includes check_memory in tools", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("check_memory");
		expect(prompt).toContain("Memory & Strategy");
	});

	it("session-based feedback flow: feedback by number with session ID → memory update", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});

		const sid = "test-session-flow";
		const reg = new Map();
		reg.set(1, { index: 1, source: "src/utils.ts", content: "", method: "posix" });
		reg.set(2, { index: 2, source: "src/main.ts", content: "", method: "posix" });
		internals(agent).sessions.set(sid, { query: "find helper function", registry: reg });

		internals(agent).lastQuery = "find helper function";
		const e1 = internals(agent).memory.append({
			query: "find helper function",
			method: "posix",
			outcome: "pending",
		});
		internals(agent).memory.registerAttempt({
			id: e1.id,
			query: "find helper function",
			method: "posix",
			sources: ["src/utils.ts"],
			timestamp: Date.now(),
		});
		const e2 = internals(agent).memory.append({
			query: "find helper function",
			method: "posix",
			outcome: "pending",
		});
		internals(agent).memory.registerAttempt({
			id: e2.id,
			query: "find helper function",
			method: "posix",
			sources: ["src/main.ts"],
			timestamp: Date.now(),
		});

		agent.recordFeedbackByNumbers(sid, [1], [2]);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		const entries = memory.getEntries();
		expect(entries.find((e) => e.id === e1.id)?.outcome).toBe("useful");
		expect(entries.find((e) => e.id === e2.id)?.outcome).toBe("not_useful");
	});
});
