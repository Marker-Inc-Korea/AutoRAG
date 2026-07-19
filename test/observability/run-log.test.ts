import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { Agent, type AgentTool, type AgentToolResult } from "@earendil-works/pi-agent-core";
import {
	type Context,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
	registerFauxProvider,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGSessionFactory } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { type AutoRAGRunEvent, AutoRAGRunLogger } from "../../src/observability/run-log.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-run-log-test-"));
	registrations = [];
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(tmpDir, { recursive: true, force: true });
});

function explorerAssignment(originalQuery: string): string {
	return [
		`Original query: ${originalQuery}`,
		"Selected retrieval method: POSIX",
		`Query variants: ${originalQuery}; ${originalQuery} evidence`,
		"Required handoff fields: Retrieved at and Temporal metadata.",
	].join("\n");
}

const SEARCH_QUERY_PREFIX = "Find and curate information for this original query: ";

function searchQueryFromContext(context: Context): string {
	for (let index = context.messages.length - 1; index >= 0; index -= 1) {
		const message = context.messages[index];
		if (message?.role !== "user") continue;
		const content =
			typeof message.content === "string"
				? message.content
				: message.content
						.filter((item) => item.type === "text")
						.map((item) => item.text)
						.join("");
		const queryMatch = new RegExp(
			`^${SEARCH_QUERY_PREFIX}(.*?)(?= Return at most \\d+ curated results\\.| Restrict search to virtual path scope |\\n\\n)`,
			"s",
		).exec(content);
		if (queryMatch?.[1] !== undefined) return queryMatch[1].trim();
	}
	throw new Error("Faux AutoRAG model did not receive a searchDocuments prompt");
}

interface EmitArgs {
	answer: string;
	results: Array<{
		number: number;
		title: string;
		summary: string;
		evidence: Array<{ excerpt: string }>;
		confidence: number;
	}>;
	mapping: Array<{ number: number; source: string; method: string; content: string }>;
}

function emitResults(args: EmitArgs): FauxResponseStep {
	return fauxAssistantMessage([fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, args)], { stopReason: "toolUse" });
}

/**
 * Build a faux model that emits a single subagent call with the given args
 * builder followed by structured emit-results. Mirrors the search-documents
 * test harness so dispatch events are produced end-to-end.
 */
function fauxModelWithSubagentArgs(
	createArgs: (provider: string, originalQuery: string) => Record<string, unknown>,
	...responses: FauxResponseStep[]
) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	const model = reg.getModel();
	reg.setResponses([
		(context) =>
			fauxAssistantMessage([fauxToolCall("subagent", createArgs(model.provider, searchQueryFromContext(context)))], {
				stopReason: "toolUse",
			}),
		...responses,
	]);
	registrations.push(reg);
	return model;
}
/**
 * Build a faux model that repeats the subagent call + each response pair for
 * every search, so multiple searches against the same agent succeed.
 */
function fauxModelForRepeatedSearches(
	createArgs: (provider: string, originalQuery: string) => Record<string, unknown>,
	...responses: FauxResponseStep[]
) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	const model = reg.getModel();
	reg.setResponses(
		responses.flatMap((response) => [
			(context: Context) =>
				fauxAssistantMessage(
					[fauxToolCall("subagent", createArgs(model.provider, searchQueryFromContext(context)))],
					{ stopReason: "toolUse" },
				),
			response,
		]),
	);
	registrations.push(reg);
	return model;
}

function groundedExplorerResult(provider: string): AgentToolResult<unknown> {
	return {
		content: [{ type: "text", text: "Explorer run completed successfully." }],
		details: {
			mode: "single",
			runId: "run-grounded",
			results: [
				{
					agent: "autorag-explorer",
					task: "Investigate the assigned query and return the required handoff.",
					exitCode: 0,
					usage: { input: 100, output: 50, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 1 },
					model: `${provider}/gpt-5.6-luna`,
					finalOutput: [
						"Source: /docs/a.md",
						"Evidence: grounded probe appears in the validation note.",
						"Retrieved at: 2026-07-14T05:30:00.000Z",
						"Temporal metadata: source date is unknown.",
					].join("\n"),
				},
			],
		},
	};
}

function readRunEvents(logPath: string): Array<Record<string, unknown>> {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => JSON.parse(line) as Record<string, unknown>);
}

function readRunEventsFromMemory(memoryPath: string): Array<Record<string, unknown>> {
	return readRunEvents(join(dirname(memoryPath), "logs", "runs.jsonl"));
}

function makeSessionFactory(subagentResult?: AgentToolResult<unknown>): AutoRAGSessionFactory {
	return async (options) => {
		const explorerModel = options.explorerModel;
		if (explorerModel === undefined) throw new Error("Test session requires an explorer model");
		const subagentTool: AgentTool = {
			name: "subagent",
			label: "Subagent",
			description: "Test explorer",
			parameters: Type.Object({
				agent: Type.Optional(Type.String()),
				agentScope: Type.Optional(Type.String()),
				model: Type.Optional(Type.String()),
				task: Type.Optional(Type.String()),
				cwd: Type.Optional(Type.String()),
				artifacts: Type.Optional(Type.Boolean()),
				tasks: Type.Optional(Type.Array(Type.Unknown())),
				chain: Type.Optional(Type.Array(Type.Unknown())),
				parallel: Type.Optional(Type.Array(Type.Unknown())),
			}),
			execute: async () => subagentResult ?? groundedExplorerResult(explorerModel.provider),
		};
		const agent = new Agent({
			initialState: {
				systemPrompt: options.systemPrompt,
				model: options.model,
				tools: [subagentTool, ...options.tools],
			},
			convertToLlm: (messages) =>
				messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult"),
		});
		return {
			agent,
			prompt: async (prompt: string) => agent.prompt(prompt),
			abort: async () => agent.abort(),
			dispose: () => {},
		};
	};
}

function makeAgent(
	model: ReturnType<typeof fauxModelWithSubagentArgs>,
	memoryPath = join(tmpDir, "memory.json"),
	subagentResult?: AgentToolResult<unknown>,
): AutoRAGAgent {
	return new AutoRAGAgent({
		model,
		searchPaths: [FIXTURE_DIR],
		memoryPath,
		workspacePath: tmpDir,
		sessionFactory: makeSessionFactory(subagentResult),
	});
}

describe("AutoRAGRunLogger serialization", () => {
	it("writes dispatch_rejected with schemaVersion 1", () => {
		const logPath = join(tmpDir, "logs", "runs.jsonl");
		const logger = new AutoRAGRunLogger(logPath);
		logger.write({
			event: "dispatch_rejected",
			schemaVersion: 1,
			sessionId: "sess-1",
			toolCallId: "call-1",
			sequence: 1,
			timestamp: "2026-07-16T00:00:00.000Z",
			dispatchKind: "launch",
			code: "DISPATCH_ARTIFACTS_INVALID",
			field: "artifacts",
			forceCorrectable: true,
		});
		const events = readRunEvents(logPath);
		expect(events).toHaveLength(1);
		expect(events[0]).toMatchObject({ event: "dispatch_rejected", schemaVersion: 1, sequence: 1 });
	});

	it("writes dispatch_autofilled with schemaVersion 1", () => {
		const logPath = join(tmpDir, "logs", "runs.jsonl");
		const logger = new AutoRAGRunLogger(logPath);
		logger.write({
			event: "dispatch_autofilled",
			schemaVersion: 1,
			sessionId: "sess-1",
			toolCallId: "call-1",
			sequence: 1,
			timestamp: "2026-07-16T00:00:00.000Z",
			dispatchKind: "launch",
			fields: { artifacts: true, agentScope: false, leafModelFillCount: 0 },
		});
		const events = readRunEvents(logPath);
		expect(events).toHaveLength(1);
		expect(events[0]).toMatchObject({ event: "dispatch_autofilled", schemaVersion: 1 });
	});

	it("serializes each line as a single JSON object with no forbidden raw fields", () => {
		const logPath = join(tmpDir, "logs", "runs.jsonl");
		const logger = new AutoRAGRunLogger(logPath);
		const secretQuery = "secret query content that must not leak";
		const secretTask = "Original query: secret query content that must not leak";
		const events: AutoRAGRunEvent[] = [
			{
				event: "dispatch_rejected",
				schemaVersion: 1,
				sessionId: "sess-secret",
				toolCallId: "call-secret",
				sequence: 1,
				timestamp: "2026-07-16T00:00:00.000Z",
				dispatchKind: "launch",
				code: "DISPATCH_QUERY_MISMATCH",
				field: ".task",
				forceCorrectable: true,
			},
			{
				event: "dispatch_autofilled",
				schemaVersion: 1,
				sessionId: "sess-secret",
				toolCallId: "call-secret-2",
				sequence: 2,
				timestamp: "2026-07-16T00:00:01.000Z",
				dispatchKind: "launch",
				fields: { artifacts: true, agentScope: true, leafModelFillCount: 1 },
			},
		];
		for (const event of events) logger.write(event);
		const raw = readFileSync(logPath, "utf8");
		// The serialized JSON must not carry query text, task/assignment content,
		// or a raw args blob — only the frozen telemetry contract fields.
		expect(raw).not.toContain(secretQuery);
		expect(raw).not.toContain(secretTask);
		expect(raw).not.toContain('"args"');
		expect(raw).not.toContain('"query"');
		expect(raw).not.toContain('"task"');
		expect(raw).not.toContain('"originalQuery"');
		expect(raw).not.toContain('"assignment"');
		// Each non-empty line must be a valid standalone JSON object.
		const lines = raw.trim().split("\n").filter(Boolean);
		expect(lines).toHaveLength(2);
		for (const line of lines) {
			expect(() => JSON.parse(line)).not.toThrow();
		}
	});

	it("never changes search behavior when the logger filesystem fails", () => {
		// Point the logger at a path whose parent is a file, so mkdir/append fails.
		const blockedDir = join(tmpDir, "blocked");
		writeFileSync(blockedDir, "not a directory");
		const logPath = join(blockedDir, "logs", "runs.jsonl");
		const logger = new AutoRAGRunLogger(logPath);
		// Writing must not throw and must not produce any side effect.
		expect(() =>
			logger.write({
				event: "dispatch_rejected",
				schemaVersion: 1,
				sessionId: "sess-1",
				toolCallId: "call-1",
				sequence: 1,
				timestamp: "2026-07-16T00:00:00.000Z",
				dispatchKind: "launch",
				code: "DISPATCH_MALFORMED",
				field: "args",
				forceCorrectable: true,
			}),
		).not.toThrow();
		expect(existsSync(logPath)).toBe(false);
	});
});

describe("AutoRAGAgent dispatch telemetry", () => {
	it("emits exactly one dispatch_rejected event for a rejected dispatch", async () => {
		const memoryPath = join(tmpDir, "rejected", "memory.json");
		// artifacts: true is rejected pre-schema → exactly one dispatch_rejected event.
		const model = fauxModelWithSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				agentScope: "user",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
				artifacts: true,
			}),
			emitResults({
				answer: "[1] rejected telemetry",
				results: [
					{ number: 1, title: "Rejected", summary: "telemetry", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/rejected.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments("rejected telemetry")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
		const events = readRunEventsFromMemory(memoryPath);
		const rejected = events.filter((e) => e.event === "dispatch_rejected");
		expect(rejected).toHaveLength(1);
		expect(rejected[0]).toMatchObject({
			event: "dispatch_rejected",
			schemaVersion: 1,
			code: "DISPATCH_ARTIFACTS_INVALID",
			dispatchKind: "launch",
			forceCorrectable: true,
		});
	});

	it("starts dispatch sequence at 1 for a rejected dispatch", async () => {
		const memoryPath = join(tmpDir, "sequence", "memory.json");
		const model = fauxModelWithSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				agentScope: "user",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
				artifacts: true,
			}),
			emitResults({
				answer: "[1] sequence telemetry",
				results: [
					{ number: 1, title: "Sequence", summary: "telemetry", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/sequence.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments("sequence telemetry")).rejects.toThrow();
		const events = readRunEventsFromMemory(memoryPath);
		const dispatchEvents = events.filter((e) => e.event === "dispatch_rejected" || e.event === "dispatch_autofilled");
		expect(dispatchEvents.length).toBeGreaterThanOrEqual(1);
		expect(dispatchEvents[0].sequence).toBe(1);
	});

	it("resets the dispatch sequence to 1 for each new search", async () => {
		const memoryPath = join(tmpDir, "sequence-reset", "memory.json");
		// A valid dispatch with omitted fields → autofilled event at sequence 1.
		const response = emitResults({
			answer: "[1] reset telemetry",
			results: [{ number: 1, title: "Reset", summary: "telemetry", evidence: [{ excerpt: "ok" }], confidence: 1 }],
			mapping: [{ number: 1, source: "/data/reset.txt", method: "posix", content: "ok" }],
		});
		const model = fauxModelForRepeatedSearches(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
			}),
			response,
			response,
		);
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments("first reset telemetry")).resolves.toMatchObject({
			answer: "[1] reset telemetry",
		});
		const firstEvents = readRunEventsFromMemory(memoryPath);
		const firstAutofilled = firstEvents.filter((e) => e.event === "dispatch_autofilled");
		expect(firstAutofilled.length).toBeGreaterThanOrEqual(1);
		expect(firstAutofilled[0].sequence).toBe(1);

		// Second search in the same agent resets the sequence counter.
		await expect(agent.searchDocuments("second reset telemetry")).resolves.toMatchObject({
			answer: "[1] reset telemetry",
		});
		const allEvents = readRunEventsFromMemory(memoryPath);
		const allAutofilled = allEvents.filter((e) => e.event === "dispatch_autofilled");
		// The second search's first autofilled event must again start at sequence 1.
		const secondSessionStart = allAutofilled.find((_e, i) => i >= firstAutofilled.length);
		expect(secondSessionStart?.sequence).toBe(1);
	});

	it("emits dispatch_autofilled only when a field actually changed", async () => {
		const memoryPath = join(tmpDir, "changed-only", "memory.json");
		// All fields explicitly set to their canonical values: artifacts=false,
		// agentScope="user", model set. Nothing is autofilled, so no
		// dispatch_autofilled event should appear.
		const model = fauxModelWithSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				agentScope: "user",
				artifacts: false,
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
			}),
			emitResults({
				answer: "[1] no autofill",
				results: [
					{ number: 1, title: "NoAutofill", summary: "telemetry", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/no-autofill.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments("no autofill")).resolves.toMatchObject({
			answer: "[1] no autofill",
		});
		const events = readRunEventsFromMemory(memoryPath);
		const autofilled = events.filter((e) => e.event === "dispatch_autofilled");
		expect(autofilled).toHaveLength(0);
	});

	it("emits dispatch_autofilled when omitted fields are filled", async () => {
		const memoryPath = join(tmpDir, "autofilled", "memory.json");
		// Omit artifacts and agentScope → both autofilled → event emitted.
		const model = fauxModelWithSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
			}),
			emitResults({
				answer: "[1] autofill event",
				results: [{ number: 1, title: "Autofill", summary: "event", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/autofill.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments("autofill event")).resolves.toMatchObject({
			answer: "[1] autofill event",
		});
		const events = readRunEventsFromMemory(memoryPath);
		const autofilled = events.filter((e) => e.event === "dispatch_autofilled");
		expect(autofilled).toHaveLength(1);
		expect(autofilled[0]).toMatchObject({
			event: "dispatch_autofilled",
			schemaVersion: 1,
			dispatchKind: "launch",
		});
		expect(autofilled[0].fields).toMatchObject({
			artifacts: true,
			agentScope: true,
		});
	});

	it("keeps forbidden raw fields absent from the persisted dispatch event JSON", async () => {
		const memoryPath = join(tmpDir, "no-raw-fields", "memory.json");
		const secretQuery = "raw field leak probe unique marker";
		const model = fauxModelWithSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				agentScope: "user",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
				artifacts: true,
			}),
			emitResults({
				answer: "[1] raw field leak",
				results: [{ number: 1, title: "RawLeak", summary: "probe", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/raw-leak.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments(secretQuery)).rejects.toThrow();
		const raw = readFileSync(join(dirname(memoryPath), "logs", "runs.jsonl"), "utf8");
		// The dispatch_rejected event must not leak the query, task, or raw args.
		expect(raw).not.toContain(secretQuery);
		expect(raw).not.toContain("Original query");
		expect(raw).not.toContain('"task"');
		expect(raw).not.toContain('"args"');
		expect(raw).not.toContain('"originalQuery"');
	});

	it("continues search when the run logger filesystem fails mid-search", async () => {
		const memoryPath = join(tmpDir, "logger-failure", "memory.json");
		const logPath = join(dirname(memoryPath), "logs", "runs.jsonl");
		// Pre-create the log path as a directory so appendFileSync fails on every
		// write attempt during the search.
		mkdirSync(logPath, { recursive: true });
		const model = fauxModelWithSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				agentScope: "user",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
				artifacts: true,
			}),
			emitResults({
				answer: "[1] logger failure nonfatal",
				results: [
					{ number: 1, title: "LoggerFail", summary: "nonfatal", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/logger-fail.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model, memoryPath);

		// The search must still reject for the real dispatch reason, not crash
		// from the logger filesystem failure.
		await expect(agent.searchDocuments("logger failure nonfatal")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
		// The log path remains a directory (appendFileSync cannot write a file
		// there), proving the write failure was swallowed without changing
		// search behavior.
		expect(existsSync(logPath)).toBe(true);
	});
});
