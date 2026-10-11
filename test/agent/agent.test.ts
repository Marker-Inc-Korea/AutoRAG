import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import type { JudgedEvidenceRecord } from "../../src/memory/judged-evidence.ts";
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
	conversationId: string;
	memory: RetrievalMemory;
	minSyncMethod:
		| {
				describe(): { name: string };
				isBinaryMissing(): boolean;
		  }
		| undefined;
	innerAgent: {
		transformContext?: (
			messages: Array<{ role: "user"; content: Array<{ type: "text"; text: string }>; timestamp: number }>,
		) => Promise<Array<{ role: string; content: Array<{ type: "text"; text: string }>; timestamp: number }>>;
	};
	tools: readonly AgentTool[];
}

function internals(agent: AutoRAGAgent): AgentInternals {
	return agent as unknown as AgentInternals;
}

function fakeModel() {
	return { id: "test-model", provider: "test-provider", api: "test-api" } as never;
}

let recordCounter = 0;

function judgedRecord(overrides: Partial<JudgedEvidenceRecord> = {}): JudgedEvidenceRecord {
	recordCounter += 1;
	const sessionId = overrides.sessionId ?? `session-${recordCounter}`;
	const conversationId = overrides.conversationId ?? "past-conversation";
	return {
		id: `${conversationId}:${sessionId}:e${recordCounter}`,
		sessionId,
		conversationId,
		question: "how to deploy the service",
		searchQuery: "how to deploy the service",
		method: "grep",
		source: `${resolve(FIXTURE_DIR)}/docs/guide.md`,
		stableEvidenceId: `e${recordCounter}`,
		resultNumber: 1,
		title: `Evidence ${recordCounter}`,
		excerpt: `excerpt ${recordCounter}`,
		probability: 0.9,
		createdAt: recordCounter,
		...overrides,
	};
}

describe("AutoRAGAgent", () => {
	it("rejects a non-positive search timeout", () => {
		expect(
			() =>
				new AutoRAGAgent({
					searchPaths: [FIXTURE_DIR],
					memoryPath: join(tmpDir, "memory.json"),
					searchTimeoutMs: 0,
				}),
		).toThrow("searchTimeoutMs must be a positive finite number");
	});

	it("rejects a non-positive tool-call limit", () => {
		expect(
			() =>
				new AutoRAGAgent({
					searchPaths: [FIXTURE_DIR],
					memoryPath: join(tmpDir, "memory.json"),
					maxSearchToolCalls: 1.5,
				}),
		).toThrow("maxSearchToolCalls must be a positive integer");
	});

	it("aborts and rejects when a search exceeds its timeout", async () => {
		let abortCalls = 0;
		const session = {
			agent: { subscribe: () => () => undefined, state: { messages: [] } },
			prompt: async () => await new Promise<void>(() => undefined),
			abort: async () => {
				abortCalls += 1;
			},
			dispose: () => undefined,
		};
		const agent = new AutoRAGAgent({
			model: fakeModel(),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			searchTimeoutMs: 10,
		});
		(agent as unknown as { createSearchSession: () => typeof session }).createSearchSession = () => session;

		const search = agent.searchDocuments("timeout query");
		await expect(search).rejects.toThrow("search timed out after 10ms");
		expect(abortCalls).toBe(1);
	});

	it("aborts a search after the configured retrieval tool-call limit", async () => {
		let abortCalls = 0;
		const session = {
			agent: {
				subscribe: (listener: (event: unknown) => void) => {
					listener({
						type: "tool_execution_end",
						toolName: "semantic_search_local_docs",
						result: { details: { method: "minsync" } },
					});
					return () => undefined;
				},
				state: { messages: [] },
			},
			prompt: async () => undefined,
			abort: async () => {
				abortCalls += 1;
			},
			dispose: () => undefined,
		};
		const agent = new AutoRAGAgent({
			model: fakeModel(),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			maxSearchToolCalls: 1,
			jikji: false,
			minSync: { autoInstall: false },
		});
		(agent as unknown as { createSearchSession: () => typeof session }).createSearchSession = () => session;

		const response = await agent.searchDocuments("cap query");
		expect(response.results).toEqual([]);
		expect(
			response.diagnostics?.some(
				(diagnostic) => diagnostic.code === "no-final-answer" && diagnostic.severity === "warning",
			),
		).toBe(true);
		expect(response.retrievalTrace).toEqual([]);
		expect(abortCalls).toBe(1);
	});

	it("does not cap retrieval tools at three executions per source", async () => {
		// A prior sync creates the MinSync workspace; without it the query's spawn
		// fails on the missing working directory, and CI has no repo-local one.
		mkdirSync(join(tmpDir, ".autorag", "minsync"), { recursive: true });
		const agent = new AutoRAGAgent({
			model: fakeModel(),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			minSync: { autoInstall: false },
			jikji: false,
		});
		const tool = internals(agent).tools.find((entry) => entry.name === "semantic_search_local_docs");
		expect(tool).toBeDefined();
		const execute = tool?.execute as (id: string, params: { query: string }) => Promise<{ details?: unknown }>;
		for (let i = 0; i < 4; i++) {
			const result = await execute(`call-${i}`, { query: "same source" });
			// Every call reaches the underlying tool; no per-source cap short-circuits it.
			expect(result.details).not.toMatchObject({ limitReached: true });
		}
		const fifth = await execute("call-5", { query: "same source" });
		expect(fifth.details).toMatchObject({ method: "semantic_search_local_docs", resultCount: 0 });
	});

	it("creates with default config", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(agent).toBeDefined();
	});

	it("accepts MinSync chunk size configuration", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { maxChunkSize: 1000, autoInstall: false },
		});

		expect((internals(agent).minSyncMethod as unknown as { maxChunkSize?: number }).maxChunkSize).toBe(1000);
	});

	it("accepts and retains configured languages", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			languages: ["ja", "en"],
		});

		expect(agent.languages).toEqual(["ja", "en"]);
	});

	it("registers the dupey duplicate scan tool by default", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(internals(agent).tools.map((tool) => tool.name)).toContain("scan_duplicate_documents");
	});

	it("can disable the dupey duplicate scan tool", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			dupey: false,
		});
		expect(internals(agent).tools.map((tool) => tool.name)).not.toContain("scan_duplicate_documents");
	});

	it("defaults to direct retrieval and reading tools for library mode", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("librarian");
		expect(prompt).toContain("read the relevant source material directly");
		expect(prompt).toContain("Use `bash` to open and verify relevant local files");
		expect(prompt).toContain("check_memory");
		for (const name of ["semantic_search_local_docs", "search_all_documents"]) {
			expect(prompt).toContain(name);
		}
		// deleted builtin/posix surface is gone
		expect(prompt).not.toContain("search_datasource_documents");
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

	it("system prompt exposes bash and omits removed read_file", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["check_memory"],
			memoryEntries: [],
			manifests: [],
		});
		expect(prompt).toContain("bash");
		expect(prompt).not.toContain("read_file");
		expect(prompt).toContain("find/grep");
	});

	it("subscribe returns an unsubscribe function", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const unsubscribe = agent.subscribe(() => undefined);
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
		expect(prompt).toContain("timeout");
		expect(prompt).not.toContain("Fallback Chain");
	});

	it("system prompt routes output through a plain reply with inline evidence ids, not an emit tool", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("End with the final answer as a plain reply");
		expect(prompt).toContain("[e3]");
		expect(prompt).toContain("curate");
		expect(prompt).not.toContain("emit_autorag_results");
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
		expect(prompt).toContain("check_memory");
	});

	it("injects durable long-term insights into the memory context", async () => {
		const memPath = join(tmpDir, "memory.json");
		const seeded = new RetrievalMemory({ storagePath: memPath });
		seeded.recordJudgedEvidence(
			Array.from({ length: 100 }, (_, index) =>
				judgedRecord({
					sessionId: index % 2 === 0 ? "session-a" : "session-b",
					createdAt: index,
				}),
			),
		);
		seeded.save();

		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
			memoryEmbedder: false,
		});
		internals(agent).lastQuery = "deploy the service";

		const transformed = await internals(agent).innerAgent.transformContext?.([
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() },
		]);

		expect(transformed?.[0].content[0].text).toContain("<memory_context>");
		expect(transformed?.[0].content[0].text).toContain("Long-Term Retrieval Insights");
	});

	it("injects current-conversation evidence into the memory context", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			memoryEmbedder: false,
		});
		internals(agent).memory.recordJudgedEvidence([
			judgedRecord({
				conversationId: internals(agent).conversationId,
				question: "alpha topic",
				searchQuery: "alpha topic",
				title: "Alpha title",
			}),
		]);
		internals(agent).lastQuery = "alpha topic";

		const transformed = await internals(agent).innerAgent.transformContext?.([
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() },
		]);

		expect(transformed?.[0].content[0].text).toContain("<memory_context>");
		expect(transformed?.[0].content[0].text).toContain("Current Conversation Memory");
		expect(transformed?.[0].content[0].text).toContain("Alpha title");
	});

	it("returns messages unchanged when only another conversation has evidence", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			memoryEmbedder: false,
		});
		internals(agent).memory.recordJudgedEvidence([
			judgedRecord({
				conversationId: "other-conversation",
				question: "alpha topic",
				searchQuery: "alpha topic",
			}),
		]);
		internals(agent).lastQuery = "photosynthesis cellular respiration";

		const messages = [{ role: "user" as const, content: [{ type: "text" as const, text: "hello" }], timestamp: 1 }];
		const transformed = await internals(agent).innerAgent.transformContext?.(messages);

		expect(transformed).toEqual(messages);
		expect(transformed?.[0].content[0].text).not.toContain("<memory_context>");
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
	it("registers MinSync and hybrid by default when options omit them", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const internal = internals(agent);
		expect(internal.minSyncMethod).toBeDefined();
		expect(internal.minSyncMethod?.describe().name).toBe("minsync");
		expect(agent.getMethodRegistry().getByType("hybrid")).toHaveLength(1);
	});

	it("defaults MinSync autoInstall to true when undefined", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(internals(agent).minSyncMethod).toBeDefined();
	});
});

describe("AutoRAGAgent.getRetrievalEngine delegation", () => {
	it("getRetrievalEngine() registers all agent methods", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const engine = agent.getRetrievalEngine();
		// Should include the registered methods (minsync, hybrid, etc.)
		expect(engine.getMethodRegistry().get("minsync")).toBeDefined();
		expect(engine.getMethodRegistry().list().length).toBeGreaterThanOrEqual(1);
	});

	it("getRetrievalEngine() is cached (same instance on second call)", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const first = agent.getRetrievalEngine();
		const second = agent.getRetrievalEngine();
		expect(first).toBe(second);
	});
});
