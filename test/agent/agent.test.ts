import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = join(tmpdir(), `autorag-agent-test-${Date.now()}`);
	mkdirSync(tmpDir, { recursive: true });
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("AutoRAGAgent", () => {
	it("creates with default config", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(agent).toBeDefined();
	});

	it("registers all retrieval methods as tools", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const registry = agent.getRegistry();
		const methods = registry.list();
		expect(methods.length).toBe(5);
		const names = methods.map((m) => m.describe().name);
		expect(names).toContain("posix");
		expect(names).toContain("vector");
		expect(names).toContain("bm25");
		expect(names).toContain("hybrid");
		expect(names).toContain("visual");
	});

	it("system prompt mentions available retrieval methods", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("posix");
		expect(prompt).toContain("retrieval");
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

	it("submitFeedback resolves pending entries and saves to disk", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		agent["lastQuery"] = "find typescript files";
		agent["memory"].append({ query: "find typescript files", method: "posix", outcome: "pending" });
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

	it("system prompt separates active methods from stubs", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("Active Retrieval Methods");
		expect(prompt).toContain("search_posix");
		expect(prompt).toContain("NOT AVAILABLE");
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
		expect(prompt).toContain("read_file");
		expect(prompt).toContain("curate");
	});

	it("system prompt warns when manifest requires unavailable method", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			manifestDir: "test/fixtures/manifests",
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("codebase-vectors");
		expect(prompt).toContain("currently unavailable");
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

	it("system prompt tool reference includes active tools and read_file", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("Tool Quick Reference");
		expect(prompt).toContain("search_posix");
		expect(prompt).toContain("read_file");
		const toolRefSection = prompt.split("Tool Quick Reference")[1];
		expect(toolRefSection).not.toContain("| search_vector");
		expect(toolRefSection).not.toContain("| search_bm25");
	});

	it("system prompt includes Memory & Strategy section", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("Memory & Strategy");
		expect(prompt).toContain("check_memory");
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
		agent["lastQuery"] = "test query";
		agent["memory"].append({ query: "test query", method: "posix", outcome: "pending" });
		agent["memory"].append({ query: "test query", method: "vector", outcome: "pending" });
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
		const entry = agent["memory"].append({ query: "q", method: "posix", outcome: "pending" });
		agent["memory"].registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
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

	it("getResultRegistry is a public method", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(typeof agent.getResultRegistry).toBe("function");
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
		reg.set(1, { index: 1, source: "src/a.ts", content: "", method: "posix" });
		agent["sessions"].set(sid, { query: "q", registry: reg });
		const entry = agent["memory"].append({ query: "q", method: "posix", outcome: "pending" });
		agent["memory"].registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
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
		reg.set(1, { index: 1, source: "src/b.ts", content: "", method: "posix" });
		agent["sessions"].set(sid, { query: "q", registry: reg });
		const entry = agent["memory"].append({ query: "q", method: "posix", outcome: "pending" });
		agent["memory"].registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
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
