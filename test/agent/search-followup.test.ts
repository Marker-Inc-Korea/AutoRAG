import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentEvent, AgentTool } from "@earendil-works/pi-agent-core";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { createEverythingSearchTool } from "../../src/agent/everything-search-tool.ts";
import { createJikjiFindTool, type JikjiFindDetails } from "../../src/agent/jikji-find-tool.ts";
import { createSearchAllDocumentsTool } from "../../src/agent/search-all-tool.ts";
import type { SearchDocumentRetrievalTraceEntry } from "../../src/agent/search-documents.ts";
import { createSearchMinSyncDocumentsTool } from "../../src/agent/search-minsync-tool.ts";
import { createSingleDatasourceSearchTools } from "../../src/agent/search-single-datasource-tool.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";
import { jikjiFindDiagnostic } from "../../src/jikji/diagnostics.ts";
import type { JikjiAnswerPack, JikjiDiagnostic, JikjiFindResult } from "../../src/jikji/index.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";
import type { MinSyncVectorMethod } from "../../src/minsync/method.ts";
import type { RetrievalDiagnostic, RetrievalResult } from "../../src/retrieval/types.ts";

let root: string;
let registrations: FauxProviderRegistration[];
const query = "refund approval";
const hit: RetrievalResult = {
	id: "refund",
	source: "/docs/refund.txt",
	content: "Director approval is required.",
	score: 1,
	metadata: {},
};
const diagnostic: RetrievalDiagnostic = {
	code: "retrieval-method-failed",
	severity: "warning",
	source: "unavailable-method",
	message: "local retrieval failed",
};

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-search-followup-")));
	registrations = [];
	vi.stubEnv("AUTORAG_HOME", join(root, "home"));
	vi.stubEnv("AUTORAG_CONFIG", join(root, "home", "config.json"));
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	vi.restoreAllMocks();
	vi.unstubAllEnvs();
	rmSync(root, { recursive: true, force: true });
});

// Register a real configured connection so its generated name is tracked by
// AutoRAG. The tests below replace only the retrieval provider, not the hooks.
const skill: DatasourceSkill = {
	describe: () => ({
		name: "fixture",
		type: "chat",
		description: "Local fixture",
		capabilities: [],
		tags: ["fixture"],
		status: "active",
		datasourceId: "fixture",
	}),
	polling: () => ({ mode: "none" }),
	index: async () => ({
		ok: true,
		skill: "fixture",
		instanceId: "default",
		chunkCount: 0,
		indexedAt: 1,
		diagnostics: [],
	}),
	retrievalMethods: () => [],
	describeSources: () => [],
	skillManifest: () => ({ name: "fixture", description: "Local fixture", content: "Local fixture" }),
};

interface Internals {
	innerAgent: { state: { tools: { name: string }[] } };
	tools: AgentTool[];
	memory: RetrievalMemory;
	searchToolCallCount: number;
	retrievalTrace: SearchDocumentRetrievalTraceEntry[];
	activeSession?: { abort(): Promise<void> };
	jikjiReady: boolean;
	jikjiClient?: { find(root: string): Promise<JikjiFindResult> };
}

function setup(
	toolOrFactory: AgentTool | ((agent: AutoRAGAgent) => AgentTool),
	maxSearchToolCalls = 64,
	searchPaths = [root],
) {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "followup" }] });
	registrations.push(registration);
	const agent = new AutoRAGAgent({
		model: registration.getModel(),
		searchPaths,
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: false,
		jikji: false,
		everything: false,
		webSearch: false,
		datasourceSkills: [skill],
		maxSearchToolCalls,
	});
	const tool = typeof toolOrFactory === "function" ? toolOrFactory(agent) : toolOrFactory;
	const internal = agent as unknown as Internals;
	// Use the production tool wrappers with local provider fakes on every OS.
	// The production session executes them normally, including error handling.
	const index = internal.tools.findIndex((entry) => entry.name === tool.name);
	if (index >= 0) internal.tools[index] = tool;
	else internal.tools.push(tool);
	internal.innerAgent.state.tools = [...internal.tools];
	return { agent, internal, registration, tool };
}

function allTool(results: RetrievalResult[], diagnostics: RetrievalDiagnostic[] = []) {
	return createSearchAllDocumentsTool({ searchAllDocuments: async () => ({ results, diagnostics }) });
}

function metadataTool(details: unknown): AgentTool {
	return {
		...allTool([]),
		execute: async () => ({ content: [{ type: "text", text: "Fixture evidence" }], details }),
	};
}

function datasourceTool(results: RetrievalResult[], diagnostics: RetrievalDiagnostic[] = []) {
	return createSingleDatasourceSearchTools(
		{ searchSingleDatasourceDocuments: async () => ({ results, diagnostics }) },
		[{ datasourceId: "fixture", description: "Local fixture", instanceScopes: [] }],
	)[0]!;
}

function minSyncTool(results: RetrievalResult[], failure = false) {
	return createSearchMinSyncDocumentsTool(
		() =>
			({
				isBinaryMissing: () => false,
				retrieve: async () => {
					if (failure) throw new Error("local MinSync failed");
					return results;
				},
			}) as unknown as MinSyncVectorMethod,
	);
}

function jikjiTool(paths: string[] | undefined, diagnostics: readonly JikjiDiagnostic[] = []) {
	return createJikjiFindTool({
		findJikji: async () => ({
			answerPack:
				paths === undefined
					? undefined
					: {
							answerPaths: paths,
							paths,
							candidates: paths.map((path) => ({ path, nextRead: "original" as const })),
							evidencePack: [],
							handoffAction: "direct_use",
							toolCallPolicy: { stopAfterFind: false, forbiddenTools: [], allowedFollowups: [] },
							agentShouldNotRerank: false,
						},
			policy: undefined,
			diagnostics,
			roots: [root],
			perRoot: [],
		}),
	});
}

const cases: { name: string; tool: () => AgentTool; positive: boolean; blank?: boolean; error?: boolean }[] = [
	{ name: "merged evidence", tool: () => allTool([hit]), positive: true },
	{ name: "empty merged search", tool: () => allTool([]), positive: false },
	{ name: "blank query", tool: () => allTool([hit]), positive: false, blank: true },
	{ name: "all methods failed", tool: () => allTool([], [diagnostic]), positive: false },
	{ name: "partial merged evidence", tool: () => allTool([hit], [diagnostic]), positive: false },
	{
		name: "thrown search failure",
		tool: () =>
			createSearchAllDocumentsTool({
				searchAllDocuments: async () => {
					throw new Error("local failure");
				},
			}),
		positive: false,
		error: true,
	},
	{
		name: "aborted retrieval",
		tool: () =>
			createSearchAllDocumentsTool({
				searchAllDocuments: async () => {
					throw new DOMException("cancelled", "AbortError");
				},
			}),
		positive: false,
		error: true,
	},
	{ name: "datasource evidence", tool: () => datasourceTool([hit]), positive: true },
	{ name: "empty datasource", tool: () => datasourceTool([]), positive: false },
	{ name: "failed datasource", tool: () => datasourceTool([], [diagnostic]), positive: false },
	{ name: "partial datasource evidence", tool: () => datasourceTool([hit], [diagnostic]), positive: false },
	{ name: "semantic evidence", tool: () => minSyncTool([hit]), positive: true },
	{ name: "empty semantic search", tool: () => minSyncTool([]), positive: false },
	{ name: "failed semantic search", tool: () => minSyncTool([], true), positive: false },
	{
		name: "unavailable semantic search",
		tool: () => createSearchMinSyncDocumentsTool(() => undefined),
		positive: false,
	},
	{
		name: "legacy Jikji details without diagnostics",
		tool: () => ({ ...metadataTool({ answerCount: 1, sources: [hit.source] }), name: "jikji_find" }),
		positive: true,
	},
	{ name: "Jikji answer paths", tool: () => jikjiTool([hit.source]), positive: true },
	{ name: "empty Jikji answer pack", tool: () => jikjiTool([]), positive: false },
	{ name: "unavailable Jikji", tool: () => jikjiTool(undefined), positive: false },
	{
		name: "Everything evidence",
		tool: () =>
			createEverythingSearchTool({
				searchEverything: async () => ({
					ok: true,
					results: [{ path: hit.source, type: "file", size: undefined, dateModified: undefined }],
				}),
			}),
		positive: true,
	},
	{
		name: "empty Everything",
		tool: () => createEverythingSearchTool({ searchEverything: async () => ({ ok: true, results: [] }) }),
		positive: false,
	},
	{
		name: "failed Everything",
		tool: () =>
			createEverythingSearchTool({
				searchEverything: async () => ({ ok: false, reason: "search-failed", message: "local failure" }),
			}),
		positive: false,
	},
];

// Exercise compatibility boundaries through the real session execution path.
for (const resultCount of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, "1", null, undefined]) {
	cases.push({
		name: `explicit invalid/empty count ${String(resultCount)} cannot use legacy evidence`,
		tool: () => metadataTool({ resultCount, sources: [hit.source], results: [hit] }),
		positive: false,
	});
}
cases.push(
	{ name: "legacy source identity", tool: () => metadataTool({ sources: [hit.source] }), positive: true },
	{ name: "legacy result identity", tool: () => metadataTool({ results: [hit] }), positive: true },
	{ name: "invalid legacy sources", tool: () => metadataTool({ sources: [null, " ", 1] }), positive: false },
	{ name: "invalid legacy results", tool: () => metadataTool({ results: [null, {}, 1] }), positive: false },
	{
		name: "positive count with harmless information",
		tool: () => metadataTool({ resultCount: 1, diagnostics: [{ code: "cache-hit", severity: "info" }] }),
		positive: true,
	},
	{
		name: "positive count with harmless warning",
		tool: () => metadataTool({ resultCount: 1, diagnostics: [{ code: "cache-miss", severity: "warning" }] }),
		positive: true,
	},
	{
		name: "positive count with explicit diagnostic error",
		tool: () => metadataTool({ resultCount: 1, diagnostics: [{ code: "other", severity: "error" }] }),
		positive: false,
	},
	{
		name: "positive count with unavailable MinSync warning",
		tool: () => allTool([hit], [{ ...diagnostic, code: "minsync-unavailable" }]),
		positive: false,
	},
	{
		name: "legacy source with failed method",
		tool: () => metadataTool({ sources: [hit.source], diagnostics: [diagnostic] }),
		positive: false,
	},
);

describe("search followup via recordSearchToolEvent", () => {
	it.each(cases)("$name", async ({ name, tool: makeTool, positive, blank, error }) => {
		const tool = makeTool();
		const { agent, internal, registration } = setup(tool);
		registration.setResponses([
			fauxAssistantMessage([fauxToolCall(tool.name, { query: blank ? " " : query })], { stopReason: "toolUse" }),
			fauxAssistantMessage([fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, { answer: "Done", results: [] })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage([{ type: "text", text: "Done" }], { stopReason: "stop" }),
		]);
		const events: AgentEvent[] = [];
		agent.subscribe((event) => {
			events.push(event);
		});
		await agent.searchDocuments(query);
		expect(internal.searchToolCallCount).toBe(1);
		const ends = events.filter(
			(event): event is Extract<AgentEvent, { type: "tool_execution_end" }> =>
				event.type === "tool_execution_end" && event.toolName === tool.name,
		);
		expect(ends).toHaveLength(1);
		expect(ends[0]).toMatchObject({ isError: error ?? false });
		if (name.startsWith("partial ")) {
			expect(ends[0]).toMatchObject({
				result: { details: { resultCount: 1, sources: [hit.source], diagnostics: [diagnostic] } },
			});
		}
		const signals = internal.memory.getSchema().feedbackSignals;
		expect(signals).toHaveLength(positive ? 1 : 0);
		if (positive) {
			expect(signals[0]).toMatchObject({
				query,
				method: tool.name.startsWith("search_datasource_") ? "datasource" : tool.name,
				source: "followup",
				weight: internal.memory.getSchema().signalDefaults.followupWeight,
			});
		}
		const persisted = new RetrievalMemory({ storagePath: join(root, "memory.json") });
		persisted.load();
		expect(persisted.getSchema().feedbackSignals).toEqual(signals);
		const details = ends[0]?.result.details as
			| { resultCount?: number; results?: SearchDocumentRetrievalTraceEntry["results"] }
			| undefined;
		if (Array.isArray(details?.results)) {
			expect(internal.retrievalTrace).toEqual([
				expect.objectContaining({
					tool: tool.name,
					resultCount: typeof details.resultCount === "number" ? details.resultCount : details.results.length,
					results: details.results,
				}),
			]);
		}
	});
});

it("preserves Jikji provider diagnostics for empty packs and unavailable providers", async () => {
	const diagnostics: readonly JikjiDiagnostic[] = [
		{ code: "jikji-unavailable", severity: "warning", source: "jikji", message: "Original provider diagnostic" },
	];
	for (const paths of [undefined, []]) {
		const tool = jikjiTool(paths, diagnostics);
		const result = await tool.execute("fixture", { query });
		expect(result.details.answerCount).toBe(0);
		expect(result.details.diagnostics).toBe(diagnostics);
		expect(result.details.diagnostics?.[0]).toBe(diagnostics[0]);
		if (paths === undefined) {
			expect(result.content).toEqual([
				{ type: "text", text: `jikji-unavailable; use bash to explore.\n${diagnostics[0]?.message}` },
			]);
		}
	}
	const blank = await jikjiTool([hit.source], diagnostics).execute("fixture", { query: " " });
	expect(blank.details.diagnostics).toEqual([]);
	expect(blank.details.answerCount).toBe(0);
});

// Keep aggregation and the production wrapper in the path: a synthetic details
// object alone cannot catch a wrapper that discards failed-root diagnostics.
describe("multi-root Jikji followup via recordSearchToolEvent", () => {
	it.each(["spawn-error", "nonzero-exit", "aborted", "success"] as const)(
		"preserves healthy evidence with second root outcome %s",
		async (outcome) => {
			const healthyRoot = join(root, "healthy");
			const otherRoot = join(root, "other");
			mkdirSync(healthyRoot);
			mkdirSync(otherRoot);
			writeFileSync(join(healthyRoot, "refund.txt"), hit.content);
			const { agent, internal, registration, tool } = setup(createJikjiFindTool, 64, [healthyRoot, otherRoot]);
			const pack: JikjiAnswerPack = {
				answerPaths: ["refund.txt"],
				paths: ["refund.txt"],
				candidates: [{ path: "refund.txt", nextRead: "original", label: "Refund", score: 1 }],
				evidencePack: [{ path: "refund.txt", nextRead: "original" }],
				handoffAction: "direct_use",
				toolCallPolicy: { stopAfterFind: false, forbiddenTools: ["bash"], allowedFollowups: [] },
				agentShouldNotRerank: true,
			};
			const healthy: JikjiFindResult = { ok: true, answerPack: pack, stdout: "", stderr: "", code: 0 };
			const other: JikjiFindResult =
				outcome === "success"
					? { ...healthy, answerPack: { ...pack, answerPaths: [], paths: [], candidates: [], evidencePack: [] } }
					: { ok: false, reason: outcome, stdout: "", stderr: "fixture failure", code: null };
			const expectedDiagnostic = jikjiFindDiagnostic(other);
			const find = vi.fn(async (sourceRoot: string) => (sourceRoot === healthyRoot ? healthy : other));
			internal.jikjiClient = { find };
			internal.jikjiReady = true;
			// A spy calls through to the actual multi-root aggregation and captures
			// the original diagnostic objects for identity checks after wrapping.
			const aggregate = vi.spyOn(agent, "findJikji");
			registration.setResponses([
				fauxAssistantMessage([fauxToolCall(tool.name, { query })], { stopReason: "toolUse" }),
				fauxAssistantMessage([fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, { answer: "Done", results: [] })], {
					stopReason: "toolUse",
				}),
				fauxAssistantMessage([{ type: "text", text: "Done" }], { stopReason: "stop" }),
			]);
			const ends: Extract<AgentEvent, { type: "tool_execution_end" }>[] = [];
			const collect = (event: AgentEvent) => {
				if (event.type === "tool_execution_end" && event.toolName === tool.name) ends.push(event);
			};
			agent.subscribe(collect);
			await agent.searchDocuments(query);
			expect(internal.searchToolCallCount).toBe(1);
			expect(ends).toHaveLength(1);
			expect(ends[0]?.isError).toBe(false);
			const details = ends[0]?.result.details as JikjiFindDetails;
			const provider = await aggregate.mock.results.at(-1)!.value;
			expect(new Set(find.mock.calls.map(([sourceRoot]) => sourceRoot))).toEqual(new Set([healthyRoot, otherRoot]));
			expect(details.answerCount).toBe(1);
			expect(details.sources).toEqual([join(healthyRoot, "refund.txt")]);
			expect(details.candidates).toEqual([{ ...pack.candidates[0], path: join(healthyRoot, "refund.txt") }]);
			expect(details.evidencePack).toEqual(provider.answerPack?.evidencePack);
			expect(details.perRoot).toEqual(provider.perRoot);
			expect(details.handoffAction).toBe(provider.policy?.handoffAction);
			expect(details.stopAfterFind).toBe(provider.policy?.stopAfterFind);
			expect(details.rawFallbackAllowed).toBe(provider.policy?.rawFallbackAllowed);
			expect(details.forbiddenTools).toEqual(provider.policy?.forbiddenTools);
			expect(details.allowedFollowups).toEqual(provider.policy?.allowedFollowups);
			const signals = internal.memory.getSchema().feedbackSignals;
			expect(signals).toHaveLength(outcome === "success" ? 1 : 0);
			if (outcome === "success") {
				expect(signals[0]).toMatchObject({ query, method: "jikji_find", source: "followup" });
			}
			expect(details.diagnostics).toEqual(expectedDiagnostic ? [expectedDiagnostic] : []);
			expect(details.diagnostics).toBe(provider.diagnostics);
			if (expectedDiagnostic) expect(details.diagnostics?.[0]).toBe(provider.diagnostics[0]);
			const persisted = new RetrievalMemory({ storagePath: join(root, "memory.json") });
			persisted.load();
			expect(persisted.getSchema().feedbackSignals).toEqual(signals);
		},
	);
});

it("still exhausts the tool budget and preserves the empty trace without rewarding a failed search", async () => {
	const tool = allTool([], [diagnostic]);
	const { agent, internal, registration } = setup(tool, 1);
	registration.setResponses([
		fauxAssistantMessage([fauxToolCall(tool.name, { query })], { stopReason: "toolUse" }),
		fauxAssistantMessage([{ type: "text", text: "No evidence" }], { stopReason: "stop" }),
	]);
	let abortCalls = 0;
	agent.subscribe((event) => {
		if (event.type !== "tool_execution_start" || !internal.activeSession) return;
		const session = internal.activeSession;
		const abort = session.abort.bind(session);
		vi.spyOn(session, "abort").mockImplementation(async () => {
			abortCalls += 1;
			await abort();
		});
	});
	const response = await agent.searchDocuments(query);
	expect(abortCalls).toBe(1);
	expect(internal.searchToolCallCount).toBe(1);
	expect(response.retrievalTrace).toEqual([{ tool: tool.name, resultCount: 0, results: [] }]);
	expect(internal.memory.getSchema().feedbackSignals).toEqual([]);
});

it("preserves explicit numbered feedback after an empty search", async () => {
	const tool = allTool([]);
	const { agent, internal, registration } = setup(tool);
	// The search returned nothing, so the agent curated a local file it read
	// itself; a real absolute path is the only evidence a non-remote run may
	// cite without a tool having returned it.
	const refundSource = join(root, "refund.txt");
	writeFileSync(refundSource, hit.content);
	registration.setResponses([
		fauxAssistantMessage([fauxToolCall(tool.name, { query })], { stopReason: "toolUse" }),
		fauxAssistantMessage(
			[
				fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
					answer: "[1] Director approval is required.",
					results: [
						{
							number: 1,
							title: "Refund",
							summary: hit.content,
							evidence: [{ excerpt: hit.content }],
							confidence: 1,
							refs: [refundSource],
						},
					],
				}),
			],
			{ stopReason: "toolUse" },
		),
	]);
	const response = await agent.searchDocuments(query);
	expect(internal.memory.getSchema().feedbackSignals).toEqual([]);
	agent.recordFeedbackByNumbers(response.sessionId, [1]);
	const signals = internal.memory.getSchema().feedbackSignals;
	expect(signals.length).toBeGreaterThan(0);
	expect(signals.every((signal) => signal.source === "explicit" && signal.weight > 0)).toBe(true);
	const persisted = new RetrievalMemory({ storagePath: join(root, "memory.json") });
	persisted.load();
	expect(persisted.getSchema().feedbackSignals).toEqual(signals);
});
