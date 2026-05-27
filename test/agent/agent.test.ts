import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";
import type { CuratedResult } from "../../src/retrieval/types.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = join(tmpdir(), `autorag-agent-test-${Date.now()}`);
	mkdirSync(tmpDir, { recursive: true });
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function makeTool(name: string): AgentTool {
	return {
		name,
		label: name,
		description: `${name} tool`,
		parameters: Type.Object({ query: Type.String() }),
		async execute() {
			return { content: [{ type: "text", text: "ok" }], details: { resultCount: 1, method: name, sources: [] } };
		},
	};
}

interface AgentInternals {
	lastQuery: string | undefined;
	memory: RetrievalMemory;
	sessions: Map<string, { query: string; registry: Map<number, CuratedResult> }>;
}

function internals(agent: AutoRAGAgent): AgentInternals {
	return agent as unknown as AgentInternals;
}

describe("AutoRAGAgent", () => {
	it("creates with default config", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(agent).toBeDefined();
	});

	it("defaults to check_memory for library mode", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("check_memory");
		expect(prompt).not.toContain("search_posix");
	});

	it("includes caller-provided standalone tools in system prompt", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool("search_custom")],
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("search_custom");
	});

	it("includes manifest descriptions in system prompt when manifestDir provided", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			manifestDir: "test/fixtures/manifests",
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("codebase-vectors");
	});

	it("extension system prompt references Pi built-in tools", () => {
		const prompt = buildSystemPrompt({
			mode: "extension",
			toolNames: ["grep", "find", "read", "ls", "check_memory"],
			memoryEntries: [],
			manifests: [],
		});
		expect(prompt).toContain("grep");
		expect(prompt).toContain("find");
		expect(prompt).toContain("read");
		expect(prompt).not.toContain("read_file");
	});

	it("submitFeedback resolves pending entries and saves to disk", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		internals(agent).lastQuery = "find typescript files";
		internals(agent).memory.append({ query: "find typescript files", method: "grep", outcome: "pending" });
		agent.submitFeedback(undefined, true);
		expect(existsSync(memPath)).toBe(true);
		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getEntries()[0].outcome).toBe("useful");
	});

	it("subscribe returns an unsubscribe function", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const unsubscribe = agent.subscribe(() => {});
		expect(typeof unsubscribe).toBe("function");
		expect(() => unsubscribe()).not.toThrow();
	});

	it("system prompt includes search strategy guidance", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("Search Strategy");
		expect(prompt).toContain("glob");
		expect(prompt).toContain("regex");
		expect(prompt).toContain("Fallback Chain");
	});

	it("system prompt includes curated output format with internal_mapping", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("<results>");
		expect(prompt).toContain("<answer>");
		expect(prompt).toContain("<internal_mapping>");
		expect(prompt).toContain("[1]");
		expect(prompt).toContain("curate");
	});

	it("system prompt includes behavioral constraints", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("READ-ONLY");
		expect(prompt).toContain("No raw paths");
		expect(prompt).toContain("internal_mapping");
	});

	it("system prompt tool reference includes check_memory", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		const toolRefSection = prompt.split("Tool Quick Reference")[1];
		expect(toolRefSection).toContain("check_memory");
	});

	it("submitFeedback resolves all pending entries for the query", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		internals(agent).lastQuery = "test query";
		internals(agent).memory.append({ query: "test query", method: "grep", outcome: "pending" });
		internals(agent).memory.append({ query: "test query", method: "find", outcome: "pending" });
		agent.submitFeedback(undefined, true);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		const entries = memory.getEntries();
		expect(entries.every((e) => e.outcome === "useful")).toBe(true);
	});

	it("submitFeedback does nothing when no lastQuery", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		agent.submitFeedback(undefined, true);
		expect(existsSync(memPath)).toBe(false);
	});

	it("recordResultFeedback() is a public method", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(typeof agent.recordResultFeedback).toBe("function");
	});

	it("recordResultFeedback() resolves pending entries by source", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		const entry = internals(agent).memory.append({ query: "q", method: "grep", outcome: "pending" });
		internals(agent).memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "grep",
			sources: ["src/a.ts"],
			timestamp: Date.now(),
		});
		agent.recordResultFeedback([{ source: "src/a.ts", useful: true }]);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("useful");
	});

	it("getResultRegistry returns empty map initially", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(agent.getResultRegistry().size).toBe(0);
	});

	it("recordFeedbackByNumbers is a public method", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(typeof agent.recordFeedbackByNumbers).toBe("function");
	});

	it("recordFeedbackByNumbers resolves useful entries by number with session", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		const sid = "test-session-1";
		const reg = new Map();
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

	it("recordFeedbackByNumbers resolves not-useful entries with session", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		const sid = "test-session-2";
		const reg = new Map();
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

	it("recordFeedbackByNumbers ignores unknown session without error", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(() => agent.recordFeedbackByNumbers("nonexistent", [99, 100])).not.toThrow();
	});
});
