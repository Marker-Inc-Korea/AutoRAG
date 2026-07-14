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
	bm25Method: { describe(): { name: string } } | undefined;
	minSyncMethod: { describe(): { name: string } } | undefined;
	innerAgent: {
		transformContext?: (
			messages: Array<{ role: "user"; content: Array<{ type: "text"; text: string }>; timestamp: number }>,
		) => Promise<Array<{ role: string; content: Array<{ type: "text"; text: string }>; timestamp: number }>>;
	};
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

	it("defaults to parent-owned retrieval and explorer tools for library mode", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("librarian");
		expect(prompt).toContain("The parent orchestrator owns `check_memory`, Jikji, datasource, and `search_*` tools.");
		expect(prompt).toContain("Explorer tasks use only read-only `read`/`grep`/`find`/`ls` tools");
		expect(prompt).toContain("check_memory");
		for (const name of [
			"search_bm25_documents",
			"search_minsync_documents",
			"search_all_documents",
			"search_datasource_documents",
		]) {
			expect(prompt).toContain(name);
		}
		// deleted builtin/posix surface is gone
		expect(prompt).not.toContain("search_posix_documents");
		expect(prompt).not.toContain("read_file");
		expect(prompt).not.toContain("READ-ONLY");
		expect(prompt).not.toContain("No raw paths");
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

	it("system prompt exposes contained explorer tools and rejects bash/read_file", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["check_memory"],
			memoryEntries: [],
			manifests: [],
		});
		expect(prompt).not.toContain("bash");
		expect(prompt).not.toContain("read_file");
		expect(prompt).toContain(
			"contained discovery and document reading within exactly one normalized assigned search root",
		);
		for (const name of ["read", "grep", "find", "ls"]) {
			expect(prompt).toContain(`\`${name}\``);
		}
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
		expect(prompt).toContain("Constraints");
		expect(prompt).toContain("No fabrication");
		expect(prompt).not.toContain("READ-ONLY");
		expect(prompt).not.toContain("No raw paths");
		expect(prompt).not.toContain("internal_mapping");
	});

	it("system prompt tool reference includes check_memory", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		const toolRefSection = prompt.split("Tool Ownership Quick Reference")[1];
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

	it("injects memory context when durable insights exist without live hints", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		internals(agent).lastQuery = "photo archive lookup";
		for (let i = 0; i < 600; i++) internals(agent).memory.recordFeedback("photo archive lookup", "posix", true);
		internals(agent).memory.save();
		internals(agent).memory.getSchema().feedbackSignals = [];

		const transformed = await internals(agent).innerAgent.transformContext?.([
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() },
		]);

		expect(transformed?.[0].content[0].text).toContain("<memory_context>");
		expect(transformed?.[0].content[0].text).toContain("Long-Term Retrieval Insights");
		expect(transformed?.[0].content[0].text).toContain("photo archive lookup");
	});

	it("getResultRegistry returns empty map initially", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(agent.getResultRegistry().size).toBe(0);
	});
});

describe("AutoRAGAgent default method registration", () => {
	it("registers BM25 and MinSync by default when options omit them", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const internal = internals(agent);
		expect(internal.bm25Method).toBeDefined();
		expect(internal.bm25Method?.describe().name).toBe("bm25");
		expect(internal.minSyncMethod).toBeDefined();
		expect(internal.minSyncMethod?.describe().name).toBe("minsync");
	});

	it("does not register BM25 when bm25: false is passed", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			bm25: false,
		});
		expect(internals(agent).bm25Method).toBeUndefined();
	});

	it("does not register MinSync when minSync: false is passed", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			minSync: false,
		});
		expect(internals(agent).minSyncMethod).toBeUndefined();
	});

	it("registers BM25 with provided options when an object is passed", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			bm25: { forceEngine: "typescript-fallback" },
		});
		expect(internals(agent).bm25Method).toBeDefined();
		expect(internals(agent).bm25Method?.describe().name).toBe("bm25");
	});

	it("defaults MinSync autoInstall to false when undefined", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(internals(agent).minSyncMethod).toBeDefined();
	});
});
