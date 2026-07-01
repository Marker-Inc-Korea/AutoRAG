import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-agent-test-"));
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

	it("includes caller-provided search tools in system prompt", () => {
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

	it("system prompt references provided tools", () => {
		const prompt = buildSystemPrompt({
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
		expect(
			memory.getMethodHints("find typescript files").find((hint) => hint.method === "grep")?.score,
		).toBeGreaterThan(0);
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

	it("system prompt routes output through emit_autorag_results without an internal_mapping channel", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("emit_autorag_results");
		expect(prompt).toContain("[1]");
		expect(prompt).toContain("curate");
		expect(prompt).not.toContain("<internal_mapping>");
		expect(prompt).not.toContain("internal_mapping");
	});

	it("system prompt includes behavioral constraints", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("READ-ONLY");
		expect(prompt).toContain("No raw paths");
		expect(prompt).not.toContain("internal_mapping");
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
		const hints = memory.getMethodHints("test query");
		expect(hints.find((hint) => hint.method === "grep")?.score).toBeGreaterThan(0);
		expect(hints.find((hint) => hint.method === "find")?.score).toBeGreaterThan(0);
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
		expect(memory.getMethodHints("q").find((hint) => hint.method === "grep")?.score).toBeGreaterThan(0);
	});

	it("getResultRegistry returns empty map initially", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(agent.getResultRegistry().size).toBe(0);
	});
});
