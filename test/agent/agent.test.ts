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

	it("submitFeedback updates memory and saves to disk", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		agent["lastQuery"] = "find typescript files";
		agent.submitFeedback(true);
		expect(existsSync(memPath)).toBe(true);
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

	it("system prompt includes structured output format", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("<results>");
		expect(prompt).toContain("<answer>");
		expect(prompt).toContain("<search_summary>");
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
		expect(prompt).toContain("Cite evidence");
	});

	it("system prompt tool reference only lists active tools", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("Tool Quick Reference");
		expect(prompt).toContain("search_posix");
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

	it("submitFeedback uses tracked method when available", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		agent["lastQuery"] = "test query";
		agent["lastMethod"] = "vector";
		agent.submitFeedback(true);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		const entries = memory.getEntries();
		expect(entries.length).toBe(1);
		expect(entries[0].method).toBe("vector");
	});

	it("submitFeedback falls back to posix when no method tracked", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		agent["lastQuery"] = "test query";
		agent.submitFeedback(true);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		const entries = memory.getEntries();
		expect(entries.length).toBe(1);
		expect(entries[0].method).toBe("posix");
	});
});
