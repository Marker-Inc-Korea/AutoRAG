import { randomUUID } from "node:crypto";
import {
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	symlinkSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { Agent, type AgentTool, type AgentToolResult } from "@earendil-works/pi-agent-core";
import {
	type Context,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent, type AutoRAGSessionFactory } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;
let registrations: FauxProviderRegistration[];

const SEARCH_QUERY_PREFIX = "Find and curate information for this original query: ";

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-search-documents-test-"));
	registrations = [];
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(tmpDir, { recursive: true, force: true });
});

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

function explorerAssignment(originalQuery: string): string {
	return [
		`Original query: ${originalQuery}`,
		"Selected retrieval method: POSIX",
		`Query variants: ${originalQuery}; ${originalQuery} evidence`,
		"Required handoff fields: Retrieved at and Temporal metadata.",
	].join("\n");
}

function fauxModelWithExplorer(explorerId: string, ...responses: FauxResponseStep[]) {
	return fauxModelWithExplorerCwd(explorerId, resolve(FIXTURE_DIR), ...responses);
}

function fauxModelWithExplorerCwd(explorerId: string, cwd: string | undefined, ...responses: FauxResponseStep[]) {
	return fauxModelWithExplorerArtifacts(explorerId, cwd, false, ...responses);
}

function fauxModelWithExplorerArtifacts(
	explorerId: string,
	cwd: string | undefined,
	artifacts: boolean | undefined,
	...responses: FauxResponseStep[]
) {
	return fauxModelFromSubagentArgs(
		(provider, originalQuery) => ({
			agent: "autorag-explorer",
			agentScope: "user",
			model: `${provider}/${explorerId}`,
			task: explorerAssignment(originalQuery),
			...(cwd !== undefined ? { cwd } : {}),
			...(artifacts !== undefined ? { artifacts } : {}),
		}),
		...responses,
	);
}

function fauxModelWithNestedExplorerTasks(explorerId: string, cwd: string, ...responses: FauxResponseStep[]) {
	return fauxModelFromSubagentArgs(
		(provider, originalQuery) => {
			const task = {
				agent: "autorag-explorer",
				model: `${provider}/${explorerId}`,
				task: explorerAssignment(originalQuery),
				cwd,
			};
			return { agentScope: "user", artifacts: false, tasks: [task, { ...task }] };
		},
		...responses,
	);
}

function fauxModelFromSubagentArgs(
	createArgs: (provider: string, originalQuery: string) => Record<string, unknown>,
	...responses: FauxResponseStep[]
) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	const model = reg.getModel();
	reg.setResponses([
		(context: Context) =>
			fauxAssistantMessage([fauxToolCall("subagent", createArgs(model.provider, searchQueryFromContext(context)))], {
				stopReason: "toolUse",
			}),
		...responses,
	]);
	registrations.push(reg);
	return model;
}

function fauxModel(...responses: FauxResponseStep[]) {
	return fauxModelWithExplorer("gpt-5.6-luna", ...responses);
}

function fauxModelForRepeatedSearches(explorerId: string, ...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	const model = reg.getModel();
	reg.setResponses(
		responses.flatMap((response) => [
			(context: Context) =>
				fauxAssistantMessage(
					[
						fauxToolCall("subagent", {
							agent: "autorag-explorer",
							agentScope: "user",
							model: `${model.provider}/${explorerId}`,
							task: explorerAssignment(searchQueryFromContext(context)),
							cwd: resolve(FIXTURE_DIR),
							artifacts: false,
						}),
					],
					{ stopReason: "toolUse" },
				),
			response,
		]),
	);
	registrations.push(reg);
	return model;
}

interface EmitArgs {
	answer: string;
	results: Array<{
		number: number;
		title: string;
		summary: string;
		evidence: Array<{ excerpt: string; lineNumber?: number }>;
		confidence: number;
	}>;
	mapping: Array<{
		number: number;
		source: string;
		method: string;
		content: string;
		evidenceRefs?: Array<{ method: string; source: string; content?: string; excerpt?: string }>;
	}>;
	warnings?: string[];
}

function emitResults(args: EmitArgs): FauxResponseStep {
	return fauxAssistantMessage([fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, args)], { stopReason: "toolUse" });
}

interface ExplorerHandoff {
	readonly finalOutput?: string;
	readonly structuredOutput?: unknown;
}

function canonicalExplorerReport(
	evidenceCandidates: readonly Record<string, unknown>[] = [
		{
			source: "/docs/a.md",
			method: "posix",
			evidence: "QZ-ORCHID appears in the validation note.",
			retrievedAt: "2026-07-14T05:30:00.000Z",
			sourceTemporal: { status: "unknown" },
		},
	],
	assignment: {
		readonly originalQuery?: string;
		readonly method?: string;
		readonly queryVariant?: string;
	} = {},
): Record<string, unknown> {
	return {
		assignment: {
			originalQuery: assignment.originalQuery ?? "grounding probe",
			method: assignment.method ?? "posix",
			queryVariant: assignment.queryVariant ?? "QZ-ORCHID validation note",
		},
		evidenceCandidates,
		summary: "The validation note contains the requested marker.",
	};
}

function piSubagentsExplorerResult(provider: string, handoff: ExplorerHandoff): AgentToolResult<unknown> {
	return {
		content: [{ type: "text", text: "Explorer run completed successfully." }],
		details: {
			mode: "single",
			runId: "run-realistic",
			results: [
				{
					agent: "autorag-explorer",
					task: "Investigate the assigned query and return the required handoff.",
					exitCode: 0,
					usage: { input: 100, output: 50, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 1 },
					model: `${provider}/gpt-5.6-luna`,
					...handoff,
				},
			],
		},
	};
}

function makeGroundingProbeAgent(handoff: ExplorerHandoff): AutoRAGAgent {
	const model = fauxModel(
		emitResults({
			answer: "[1] grounding probe",
			results: [{ number: 1, title: "Grounding", summary: "probe", evidence: [{ excerpt: "fact" }], confidence: 1 }],
			mapping: [{ number: 1, source: "/docs/a.md", method: "posix", content: "fact" }],
		}),
	);
	return new AutoRAGAgent({
		model,
		searchPaths: [FIXTURE_DIR],
		memoryPath: join(tmpDir, "memory.json"),
		workspacePath: tmpDir,
		sessionFactory: fauxSessionFactory(undefined, piSubagentsExplorerResult(model.provider, handoff)),
	});
}

function callerTool(name: string): AgentTool {
	return {
		name,
		label: name,
		description: `${name} caller tool`,
		parameters: Type.Object({ query: Type.String() }),
		async execute() {
			return { content: [{ type: "text", text: "caller" }], details: { method: name, resultCount: 1, sources: [] } };
		},
	};
}

function fauxSessionFactory(
	onSubagent?: (args: unknown) => void,
	subagentResult?: AgentToolResult<unknown>,
): AutoRAGSessionFactory {
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
			execute: async (_toolCallId, params) => {
				onSubagent?.(params);
				return (
					subagentResult ??
					piSubagentsExplorerResult(explorerModel.provider, {
						finalOutput:
							"source: /docs/a evidence: grounded retrievedAt: 2026-07-13T00:00:00.000Z temporal metadata: unknown",
					})
				);
			},
		};
		const agent = new Agent({
			initialState: {
				systemPrompt: options.systemPrompt,
				model: options.model,
				tools: [subagentTool, ...options.tools],
			},
			convertToLlm: (messages) =>
				messages.filter(
					(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
				),
		});
		return {
			agent,
			prompt: async (prompt: string) => agent.prompt(prompt),
			abort: async () => agent.abort(),
			dispose: () => {},
		};
	};
}

function makeAgent(model: ReturnType<typeof fauxModel>, memoryPath = join(tmpDir, "memory.json")) {
	return new AutoRAGAgent({
		model,
		searchPaths: [FIXTURE_DIR],
		memoryPath,
		workspacePath: tmpDir,
		sessionFactory: fauxSessionFactory(),
	});
}

interface CleanupCounts {
	unsubscribe: number;
	dispose: number;
}

function cleanupProbeSessionFactory(
	counts: CleanupCounts,
	options: {
		readonly promptError?: Error;
		readonly unsubscribeError?: Error;
		readonly disposeError?: Error;
	},
): AutoRAGSessionFactory {
	let sessionCount = 0;
	return async (sessionOptions) => {
		const baseSession = await fauxSessionFactory()(sessionOptions);
		sessionCount += 1;
		if (sessionCount !== 1) return baseSession;

		const subscribe = baseSession.agent.subscribe.bind(baseSession.agent);
		vi.spyOn(baseSession.agent, "subscribe").mockImplementation((listener) => {
			const unsubscribe = subscribe(listener);
			return () => {
				counts.unsubscribe += 1;
				if (options.unsubscribeError !== undefined) throw options.unsubscribeError;
				unsubscribe();
			};
		});

		return {
			...baseSession,
			prompt: async (prompt) => {
				if (options.promptError !== undefined) throw options.promptError;
				await baseSession.prompt(prompt);
			},
			dispose: () => {
				counts.dispose += 1;
				if (options.disposeError !== undefined) throw options.disposeError;
				baseSession.dispose();
			},
		};
	};
}

function readRunEvents(memoryPath: string): Array<Record<string, unknown>> {
	const logPath = join(dirname(memoryPath), "logs", "runs.jsonl");
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => JSON.parse(line) as Record<string, unknown>);
}

function makeRunLogPath(memoryPath: string): string {
	return join(dirname(memoryPath), "logs", "runs.jsonl");
}

function replaceRunLogWithDirectory(memoryPath: string): void {
	const logPath = makeRunLogPath(memoryPath);
	rmSync(logPath, { recursive: true, force: true });
	mkdirSync(logPath);
}

describe("AutoRAGAgent searchDocuments", () => {
	it("keeps an explicit memoryPath unchanged when AUTORAG_HOME is configured", async () => {
		const explicitMemoryPath = join(tmpDir, "explicit-state", "memory.json");
		const autoragHome = join(tmpDir, "configured-home");
		const previousAutoragHome = process.env.AUTORAG_HOME;
		process.env.AUTORAG_HOME = autoragHome;
		const model = fauxModel(
			emitResults({
				answer: "[1] explicit memory path",
				results: [
					{ number: 1, title: "Explicit", summary: "memory path", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/explicit.txt", method: "posix", content: "ok" }],
			}),
		);

		try {
			const agent = new AutoRAGAgent({
				model,
				searchPaths: [FIXTURE_DIR],
				memoryPath: explicitMemoryPath,
				workspacePath: tmpDir,
				sessionFactory: fauxSessionFactory(),
			});

			await agent.searchDocuments("explicit memory path");

			expect(existsSync(explicitMemoryPath)).toBe(true);
			expect(existsSync(makeRunLogPath(explicitMemoryPath))).toBe(true);
			expect(existsSync(join(autoragHome, "memory.json"))).toBe(false);
			expect(existsSync(join(autoragHome, "logs", "runs.jsonl"))).toBe(false);
		} finally {
			if (previousAutoragHome === undefined) delete process.env.AUTORAG_HOME;
			else process.env.AUTORAG_HOME = previousAutoragHome;
		}
	});

	it("uses AUTORAG_HOME for memory and run logs when memoryPath is omitted", async () => {
		const autoragHome = join(tmpDir, "configured-home");
		const fallbackHome = join(tmpDir, "fallback-home");
		const previousHome = process.env.HOME;
		const previousAutoragHome = process.env.AUTORAG_HOME;
		process.env.HOME = fallbackHome;
		process.env.AUTORAG_HOME = autoragHome;
		const model = fauxModel(
			emitResults({
				answer: "[1] default memory path",
				results: [
					{ number: 1, title: "Default", summary: "memory path", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/default.txt", method: "posix", content: "ok" }],
			}),
		);

		try {
			const agent = new AutoRAGAgent({
				model,
				searchPaths: [FIXTURE_DIR],
				workspacePath: tmpDir,
				sessionFactory: fauxSessionFactory(),
			});

			await agent.searchDocuments("default memory path");

			expect(existsSync(join(autoragHome, "memory.json"))).toBe(true);
			expect(existsSync(join(autoragHome, "logs", "runs.jsonl"))).toBe(true);
			expect(existsSync(join(fallbackHome, ".autorag", "memory.json"))).toBe(false);
		} finally {
			if (previousHome === undefined) delete process.env.HOME;
			else process.env.HOME = previousHome;
			if (previousAutoragHome === undefined) delete process.env.AUTORAG_HOME;
			else process.env.AUTORAG_HOME = previousAutoragHome;
		}
	});

	it("writes redacted started/completed run events beside durable memory", async () => {
		const stateDir = join(tmpDir, "state");
		const memoryPath = join(stateDir, "memory.json");
		const apiKey = "TOP_SECRET_AUTORAG_KEY";
		const query = "AUTORAG_QUERY_SECRET_7f4c9d2e_DO_NOT_PERSIST";
		const model = fauxModel(
			emitResults({
				answer: "[1] logged",
				results: [{ number: 1, title: "Logged", summary: "run", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/logged.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			apiKey,
			searchPaths: [FIXTURE_DIR],
			memoryPath,
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(),
		});

		await agent.searchDocuments(query);
		const rawRunLog = readFileSync(join(stateDir, "logs", "runs.jsonl"), "utf8");
		expect(rawRunLog.includes(query), "raw runs.jsonl persisted the unique query fixture").toBe(false);
		const events = readRunEvents(memoryPath);
		expect(events.map((event) => event.event)).toEqual(["search_started", "search_completed"]);
		expect(events[0]).toMatchObject({
			queryLength: query.length,
			orchestratorModel: model.id,
			explorerModel: "gpt-5.6-luna",
		});
		expect(rawRunLog.includes(apiKey), "raw runs.jsonl persisted the API key fixture").toBe(false);
	});

	it("continues search when search_started logging cannot create its directory", async () => {
		const stateDir = join(tmpDir, "started-log-failure");
		const memoryPath = join(stateDir, "memory.json");
		mkdirSync(stateDir, { recursive: true });
		writeFileSync(join(stateDir, "logs"), "blocked");
		const model = fauxModel(
			emitResults({
				answer: "[1] started logging failed",
				results: [
					{
						number: 1,
						title: "Started log",
						summary: "directory failure",
						evidence: [{ excerpt: "ok" }],
						confidence: 1,
					},
				],
				mapping: [{ number: 1, source: "/data/started-log.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments("started log failure")).resolves.toMatchObject({
			answer: "[1] started logging failed",
		});
	});

	it("continues successful search when search_completed logging cannot append", async () => {
		const memoryPath = join(tmpDir, "completed-log-failure", "memory.json");
		const model = fauxModel(
			emitResults({
				answer: "[1] completed logging failed",
				results: [
					{
						number: 1,
						title: "Completed log",
						summary: "append failure",
						evidence: [{ excerpt: "ok" }],
						confidence: 1,
					},
				],
				mapping: [{ number: 1, source: "/data/completed-log.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath,
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(() => replaceRunLogWithDirectory(memoryPath)),
		});

		await expect(agent.searchDocuments("completed log failure")).resolves.toMatchObject({
			answer: "[1] completed logging failed",
		});
	});

	it("continues search when run event serialization fails", async () => {
		const memoryPath = join(tmpDir, "serialization-log-failure", "memory.json");
		const model = fauxModel(
			emitResults({
				answer: "[1] serialization logging failed",
				results: [
					{
						number: 1,
						title: "Serialization log",
						summary: "serialization failure",
						evidence: [{ excerpt: "ok" }],
						confidence: 1,
					},
				],
				mapping: [{ number: 1, source: "/data/serialization-log.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model, memoryPath);
		const stringify = JSON.stringify;
		const stringifySpy = vi.spyOn(JSON, "stringify").mockImplementation((value) => {
			if (typeof value === "object" && value !== null && "event" in value) {
				throw new Error("run event serialization failure");
			}
			return stringify(value);
		});

		try {
			await expect(agent.searchDocuments("serialization log failure")).resolves.toMatchObject({
				answer: "[1] serialization logging failed",
			});
		} finally {
			stringifySpy.mockRestore();
		}
	});

	it("preserves the original search error when search_failed logging cannot append", async () => {
		const memoryPath = join(tmpDir, "failed-log-failure", "memory.json");
		const originalError = new Error("original search failure");
		const model = fauxModel();
		const baseSessionFactory = fauxSessionFactory();
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath,
			workspacePath: tmpDir,
			sessionFactory: async (options) => {
				const session = await baseSessionFactory(options);
				return {
					...session,
					prompt: async () => {
						replaceRunLogWithDirectory(memoryPath);
						throw originalError;
					},
				};
			},
		});

		await expect(agent.searchDocuments("failed log failure")).rejects.toBe(originalError);
	});

	it("preserves a primary search error when unsubscribe cleanup throws and attempts every cleanup action", async () => {
		const primaryError = new Error("primary search failure");
		const counts = { unsubscribe: 0, dispose: 0 } satisfies CleanupCounts;
		const memoryPath = join(tmpDir, "unsubscribe-cleanup-error", "memory.json");
		const unsubscribeError = new Error("cleanup secret message /private/cleanup-path");
		const model = fauxModel(
			emitResults({
				answer: "search after unsubscribe cleanup",
				results: [
					{ number: 1, title: "After cleanup", summary: "search", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/after-cleanup.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath,
			workspacePath: tmpDir,
			sessionFactory: cleanupProbeSessionFactory(counts, {
				promptError: primaryError,
				unsubscribeError,
			}),
		});
		agent.subscribe(() => {});

		await expect(agent.searchDocuments("cleanup failure query secret")).rejects.toBe(primaryError);
		expect(counts).toEqual({ unsubscribe: 2, dispose: 1 });
		const rawRunLog = readFileSync(makeRunLogPath(memoryPath), "utf8");
		expect(rawRunLog).not.toContain(unsubscribeError.message);
		expect(rawRunLog).not.toContain("cleanup failure query secret");
		expect(readRunEvents(memoryPath).map((event) => event.event)).toEqual([
			"search_started",
			"search_failed",
			"cleanup_failed",
		]);
		expect(readRunEvents(memoryPath)).toContainEqual(
			expect.objectContaining({
				event: "cleanup_failed",
				sessionId: expect.stringMatching(/^[0-9a-f-]{36}$/),
				failureCount: 2,
				errorTypes: ["Error"],
			}),
		);
		await expect(agent.searchDocuments("search after unsubscribe cleanup")).resolves.toMatchObject({
			answer: "search after unsubscribe cleanup",
		});
	});

	it("preserves a primary search error when dispose cleanup throws", async () => {
		const primaryError = new Error("primary search failure");
		const counts = { unsubscribe: 0, dispose: 0 } satisfies CleanupCounts;
		const memoryPath = join(tmpDir, "dispose-cleanup-error", "memory.json");
		const model = fauxModel(
			emitResults({
				answer: "search after dispose cleanup",
				results: [
					{ number: 1, title: "After dispose", summary: "search", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/after-dispose.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath,
			workspacePath: tmpDir,
			sessionFactory: cleanupProbeSessionFactory(counts, {
				promptError: primaryError,
				disposeError: new Error("dispose cleanup failure"),
			}),
		});

		await expect(agent.searchDocuments("dispose cleanup failure")).rejects.toBe(primaryError);
		expect(counts).toEqual({ unsubscribe: 1, dispose: 1 });
		expect(readRunEvents(memoryPath)).toContainEqual(
			expect.objectContaining({
				event: "cleanup_failed",
				failureCount: 1,
				errorTypes: ["Error"],
			}),
		);
		await expect(agent.searchDocuments("search after dispose cleanup")).resolves.toMatchObject({
			answer: "search after dispose cleanup",
		});
	});

	it("does not fail a successful search when cleanup throws and clears state for the next search", async () => {
		const counts = { unsubscribe: 0, dispose: 0 } satisfies CleanupCounts;
		const memoryPath = join(tmpDir, "successful-cleanup-error", "memory.json");
		const model = fauxModelForRepeatedSearches(
			"gpt-5.6-luna",
			emitResults({
				answer: "first successful search",
				results: [{ number: 1, title: "First", summary: "search", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/first.txt", method: "posix", content: "ok" }],
			}),
			emitResults({
				answer: "second successful search",
				results: [{ number: 1, title: "Second", summary: "search", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/second.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath,
			workspacePath: tmpDir,
			sessionFactory: cleanupProbeSessionFactory(counts, {
				unsubscribeError: new Error("unsubscribe cleanup failure"),
				disposeError: new Error("dispose cleanup failure"),
			}),
		});
		agent.subscribe(() => {});

		await expect(agent.searchDocuments("first search")).resolves.toMatchObject({ answer: "first successful search" });
		expect(counts).toEqual({ unsubscribe: 2, dispose: 1 });
		const events = readRunEvents(memoryPath);
		expect(events.map((event) => event.event)).toEqual(["search_started", "search_completed", "cleanup_failed"]);
		expect(events).toContainEqual(
			expect.objectContaining({
				event: "cleanup_failed",
				sessionId: expect.stringMatching(/^[0-9a-f-]{36}$/),
				failureCount: 3,
				errorTypes: ["Error"],
			}),
		);
		await expect(agent.searchDocuments("second search")).resolves.toMatchObject({
			answer: "second successful search",
		});
	});

	it("uses the configured explorer model for dispatch and grounded handoff validation", async () => {
		let sessionPrompt = "";
		let sessionSystemPrompt = "";
		let sessionExplorerModel: { provider: string; id: string } | undefined;
		let subagentArgs: unknown;
		const model = fauxModelWithExplorerCwd(
			"custom-explorer",
			resolve(FIXTURE_DIR),
			emitResults({
				answer: "[1] configured explorer",
				results: [
					{ number: 1, title: "Configured", summary: "explorer", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/configured.txt", method: "posix", content: "ok" }],
			}),
		);
		const baseFactory = fauxSessionFactory((args) => {
			subagentArgs = args;
		});
		const agent = new AutoRAGAgent({
			model,
			explorerModel: { ...model, id: "custom-explorer", name: "Custom Explorer" },
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: async (options) => {
				sessionSystemPrompt = options.systemPrompt;
				sessionExplorerModel = options.explorerModel;
				const session = await baseFactory(options);
				return {
					...session,
					prompt: async (prompt) => {
						sessionPrompt = prompt;
						await session.prompt(prompt);
					},
				};
			},
		});

		const response = await agent.searchDocuments("configured model query");
		expect(response.answer).toContain("configured explorer");
		expect(sessionExplorerModel).toMatchObject({ provider: model.provider, id: "custom-explorer" });
		expect(subagentArgs).toMatchObject({
			agent: "autorag-explorer",
			agentScope: "user",
			model: `${model.provider}/custom-explorer`,
			cwd: resolve(FIXTURE_DIR),
			artifacts: false,
		});
		expect(sessionPrompt).toContain(`${model.provider}/custom-explorer`);
		expect(sessionSystemPrompt).toContain(model.id);
		expect(sessionSystemPrompt).toContain("custom-explorer");
		expect(sessionSystemPrompt).not.toContain("gpt-5.6-sol");
		expect(sessionSystemPrompt).not.toContain("gpt-5.6-luna");
	});

	it("credits a required explorer task when artifacts is omitted (autofilled to false)", async () => {
		let capturedArgs: Record<string, unknown> | undefined;
		const model = fauxModelWithExplorerArtifacts(
			"gpt-5.6-luna",
			resolve(FIXTURE_DIR),
			undefined,
			emitResults({
				answer: "[1] omitted artifacts autofill",
				results: [
					{ number: 1, title: "Omitted", summary: "artifacts", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/omitted-artifacts.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory((args) => {
				capturedArgs = args as Record<string, unknown>;
			}),
		});

		await expect(agent.searchDocuments("omitted artifacts autofill")).resolves.toMatchObject({
			answer: "[1] omitted artifacts autofill",
		});
		// The normalized args that reached execution must carry the autofilled artifacts=false.
		expect(capturedArgs).toMatchObject({ artifacts: false });
	});

	it("rejects a required explorer task with artifacts true", async () => {
		let explorerExecutions = 0;
		const model = fauxModelWithExplorerArtifacts(
			"gpt-5.6-luna",
			resolve(FIXTURE_DIR),
			true,
			emitResults({
				answer: "[1] invalid artifacts",
				results: [
					{ number: 1, title: "Invalid", summary: "artifacts", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/invalid-artifacts.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(() => {
				explorerExecutions += 1;
			}),
		});

		await expect(agent.searchDocuments("invalid artifacts")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
		expect(explorerExecutions).toBe(0);
	});

	it("credits a required explorer task when agentScope is omitted (autofilled to user)", async () => {
		let capturedArgs: Record<string, unknown> | undefined;
		const model = fauxModelFromSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
			}),
			emitResults({
				answer: "[1] omitted agentScope autofill",
				results: [
					{ number: 1, title: "Omitted", summary: "agentScope", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/omitted-agentScope.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory((args) => {
				capturedArgs = args as Record<string, unknown>;
			}),
		});

		await expect(agent.searchDocuments("omitted agentScope autofill")).resolves.toMatchObject({
			answer: "[1] omitted agentScope autofill",
		});
		// The normalized args that reached execution must carry the autofilled agentScope=user.
		expect(capturedArgs).toMatchObject({ agentScope: "user" });
	});

	it("rejects a required explorer task with agentScope project", async () => {
		let explorerExecutions = 0;
		const model = fauxModelFromSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				agentScope: "project",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
				artifacts: false,
			}),
			emitResults({
				answer: "[1] invalid agent scope",
				results: [
					{ number: 1, title: "Invalid", summary: "agent scope", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/invalid-agent-scope.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(() => {
				explorerExecutions += 1;
			}),
		});

		await expect(agent.searchDocuments("invalid agent scope")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
		expect(explorerExecutions).toBe(0);
	});

	it("credits nested explorer tasks when artifacts is only set on the top-level invocation", async () => {
		const model = fauxModelWithNestedExplorerTasks(
			"gpt-5.6-luna",
			resolve(FIXTURE_DIR),
			emitResults({
				answer: "[1] valid nested artifacts",
				results: [
					{
						number: 1,
						title: "Valid nested",
						summary: "top-level artifacts",
						evidence: [{ excerpt: "ok" }],
						confidence: 1,
					},
				],
				mapping: [{ number: 1, source: "/data/valid-nested-artifacts.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model);

		await expect(agent.searchDocuments("valid nested artifacts")).resolves.toMatchObject({
			answer: "[1] valid nested artifacts",
		});
	});

	it("credits explorer-only fanout nested in a chain parallel step", async () => {
		const model = fauxModelFromSubagentArgs(
			(provider, originalQuery) => {
				const task = {
					agent: "autorag-explorer",
					model: `${provider}/gpt-5.6-luna`,
					task: explorerAssignment(originalQuery),
					cwd: resolve(FIXTURE_DIR),
				};
				return {
					agentScope: "user",
					artifacts: false,
					chain: [{ parallel: [task, { ...task }] }],
				};
			},
			emitResults({
				answer: "[1] valid chain fanout",
				results: [
					{ number: 1, title: "Valid", summary: "chain fanout", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/valid-chain-fanout.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model);

		await expect(agent.searchDocuments("valid chain fanout")).resolves.toMatchObject({
			answer: "[1] valid chain fanout",
		});
	});

	it.each([
		[
			"an unknown single agent",
			(task: Record<string, unknown>) => ({
				...task,
				agent: "unrestricted-helper",
				agentScope: "user",
				artifacts: false,
			}),
		],
		[
			"mixed top-level tasks",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				tasks: [task, { ...task, agent: "unrestricted-helper" }],
			}),
		],
		[
			"a mixed sequential chain",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				chain: [task, { ...task, agent: "unrestricted-helper" }],
			}),
		],
		[
			"a mixed static parallel chain step",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				chain: [{ parallel: [task, { ...task, agent: "unrestricted-helper" }] }],
			}),
		],
		[
			"a mixed dynamic parallel chain step",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				chain: [
					task,
					{
						expand: { from: { output: "targets", path: "/items" }, maxItems: 1 },
						parallel: { ...task, agent: "unrestricted-helper" },
						collect: { as: "findings" },
					},
				],
			}),
		],
		[
			"a malformed top-level task without an agent",
			(task: Record<string, unknown>) => {
				const malformedTask = { ...task };
				delete malformedTask.agent;
				return { agentScope: "user", artifacts: false, tasks: [task, malformedTask] };
			},
		],
		[
			"a second explorer using a non-configured model",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				tasks: [task, { ...task, model: "other/gpt-5.6-luna" }],
			}),
		],
		[
			"a second explorer outside the configured root",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				tasks: [task, { ...task, cwd: tmpDir }],
			}),
		],
		[
			"a second explorer without required role metadata",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				tasks: [task, { ...task, task: "Original query: test" }],
			}),
		],
	] as const)("blocks %s before any child executes", async (_label, createDispatch) => {
		let explorerExecutions = 0;
		const model = fauxModelFromSubagentArgs(
			(provider, originalQuery) => {
				const explorerTask = {
					agent: "autorag-explorer",
					model: `${provider}/gpt-5.6-luna`,
					task: explorerAssignment(originalQuery),
					cwd: resolve(FIXTURE_DIR),
				};
				return createDispatch(explorerTask);
			},
			emitResults({
				answer: "[1] unsafe dispatch",
				results: [
					{ number: 1, title: "Unsafe", summary: "dispatch", evidence: [{ excerpt: "bad" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/unsafe.txt", method: "posix", content: "bad" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(() => {
				explorerExecutions += 1;
			}),
		});

		await expect(agent.searchDocuments("unsafe dispatch")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
		expect(explorerExecutions).toBe(0);
	});

	it.each([
		[
			"tasks",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				tasks: [task, { task: task.task, model: task.model, cwd: task.cwd }],
			}),
		],
		[
			"chain",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				chain: [task, { task: task.task, model: task.model, cwd: task.cwd }],
			}),
		],
		[
			"parallel",
			(task: Record<string, unknown>) => ({
				agentScope: "user",
				artifacts: false,
				parallel: [task, { task: task.task, model: task.model, cwd: task.cwd }],
			}),
		],
	] as const)(
		"blocks a %s leaf with task/model/cwd but no agent before any child executes",
		async (_label, createDispatch) => {
			let explorerExecutions = 0;
			const model = fauxModelFromSubagentArgs(
				(provider, originalQuery) => {
					const explorerTask = {
						agent: "autorag-explorer",
						model: `${provider}/gpt-5.6-luna`,
						task: explorerAssignment(originalQuery),
						cwd: resolve(FIXTURE_DIR),
					};
					return createDispatch(explorerTask);
				},
				emitResults({
					answer: "[1] no-agent leaf",
					results: [
						{ number: 1, title: "NoAgent", summary: "leaf", evidence: [{ excerpt: "bad" }], confidence: 1 },
					],
					mapping: [{ number: 1, source: "/data/no-agent-leaf.txt", method: "posix", content: "bad" }],
				}),
			);
			const agent = new AutoRAGAgent({
				model,
				searchPaths: [FIXTURE_DIR],
				memoryPath: join(tmpDir, "memory.json"),
				workspacePath: tmpDir,
				sessionFactory: fauxSessionFactory(() => {
					explorerExecutions += 1;
				}),
			});

			await expect(agent.searchDocuments("no-agent leaf")).rejects.toThrow(
				"requires a successful autorag-explorer subagent call",
			);
			expect(explorerExecutions).toBe(0);
		},
	);

	it.each([
		["a mismatched", "unrelated prior search"],
		["an empty", ""],
		["a placeholder", "<original query>"],
	] as const)("blocks %s declared original query before any child executes", async (_label, declaredQuery) => {
		let explorerExecutions = 0;
		const activeQuery = "dispatch grounding probe";
		const model = fauxModelFromSubagentArgs(
			(provider) => ({
				agent: "autorag-explorer",
				agentScope: "user",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(declaredQuery),
				cwd: resolve(FIXTURE_DIR),
				artifacts: false,
			}),
			emitResults({
				answer: "[1] mismatched dispatch",
				results: [
					{ number: 1, title: "Mismatch", summary: "dispatch", evidence: [{ excerpt: "bad" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/docs/a.md", method: "posix", content: "bad" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(
				() => {
					explorerExecutions += 1;
				},
				piSubagentsExplorerResult(model.provider, {
					structuredOutput: canonicalExplorerReport(undefined, { originalQuery: activeQuery }),
				}),
			),
		});

		await expect(agent.searchDocuments(activeQuery)).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
		expect(explorerExecutions).toBe(0);
	});

	it("revalidates the declared original query during success accounting", async () => {
		let explorerExecutions = 0;
		const activeQuery = "success accounting grounding probe";
		const changedQuery = "mutated unrelated search";
		const model = fauxModel(
			emitResults({
				answer: "[1] mutated dispatch",
				results: [
					{ number: 1, title: "Mutation", summary: "dispatch", evidence: [{ excerpt: "bad" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/docs/a.md", method: "posix", content: "bad" }],
			}),
		);
		let agent: AutoRAGAgent;
		agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: async (options) => {
				const session = await fauxSessionFactory(
					() => {
						explorerExecutions += 1;
					},
					piSubagentsExplorerResult(model.provider, {
						structuredOutput: canonicalExplorerReport(undefined, { originalQuery: changedQuery }),
					}),
				)(options);
				session.agent.beforeToolCall = async () => {
					(agent as unknown as { lastQuery: string }).lastQuery = changedQuery;
					return undefined;
				};
				return session;
			},
		});

		await expect(agent.searchDocuments(activeQuery)).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
		expect(explorerExecutions).toBe(1);
	});

	it("keeps explorer result association bound to the exact dispatched task index", async () => {
		const activeQuery = "exact task index grounding probe";
		const model = fauxModel(
			emitResults({
				answer: "[1] unmatched result",
				results: [
					{ number: 1, title: "Unmatched", summary: "result", evidence: [{ excerpt: "bad" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/docs/a.md", method: "posix", content: "bad" }],
			}),
		);
		const subagentResult: AgentToolResult<unknown> = {
			content: [{ type: "text", text: "Explorer fanout completed." }],
			details: {
				mode: "parallel",
				results: [
					{
						agent: "autorag-explorer",
						exitCode: 1,
						finalOutput: "failed",
					},
					{
						agent: "autorag-explorer",
						exitCode: 0,
						structuredOutput: canonicalExplorerReport(undefined, { originalQuery: activeQuery }),
					},
				],
			},
		};
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(undefined, subagentResult),
		});

		await expect(agent.searchDocuments(activeQuery)).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it("credits a realistic pi-subagents successful explorer result envelope", async () => {
		const model = fauxModel(
			emitResults({
				answer: "[1] realistic explorer handoff",
				results: [
					{ number: 1, title: "Realistic", summary: "handoff", evidence: [{ excerpt: "fact" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/docs/a.md", method: "posix", content: "fact" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(undefined, {
				content: [{ type: "text", text: "Explorer run completed successfully." }],
				details: {
					mode: "single",
					runId: "run-realistic",
					results: [
						{
							agent: "autorag-explorer",
							task: "Investigate the assigned query and return the required handoff.",
							exitCode: 0,
							usage: { input: 100, output: 50, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 1 },
							model: `${model.provider}/gpt-5.6-luna`,
							finalOutput: [
								"Source: /docs/a.md",
								"Evidence: QZ-ORCHID appears in the validation note.",
								"Retrieved at: 2026-07-14T05:30:00.000Z",
								"Temporal metadata: source date is unknown.",
							].join("\n"),
						},
					],
				},
			}),
		});

		await expect(agent.searchDocuments("realistic explorer handoff")).resolves.toMatchObject({
			answer: "[1] realistic explorer handoff",
		});
	});

	it("accepts the sanitized C001 Luna Markdown handoff when the dispatch query matches", async () => {
		const query =
			"What is the exact unique sentinel in the only fixture file, and what is its full source path? Quote the sentinel exactly and name only that source.";
		const model = fauxModel(
			emitResults({
				answer: "[1] C001 sentinel",
				results: [
					{ number: 1, title: "C001", summary: "sentinel", evidence: [{ excerpt: "fact" }], confidence: 1 },
				],
				mapping: [
					{
						number: 1,
						source: "/private/tmp/autorag-precommit-real-cli-final-20260714/fixture/only-fixture.txt",
						method: "posix",
						content: "fact",
					},
				],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(undefined, {
				...piSubagentsExplorerResult(model.provider, {
					finalOutput: [
						"Candidate finding:",
						"",
						"- **Source:** `/private/tmp/autorag-precommit-real-cli-final-20260714/fixture/only-fixture.txt`",
						"- **Retrieval method:** Contained POSIX discovery and reading (`find`, `ls`, `read`)",
						"- **Query variant:** “list all files, identify the only fixture file, and inspect its complete contents for a unique literal”",
						"- **Relevance:** strong",
						"- **Exact supporting excerpt:** Line 2: `Unique sentinel: AUTORAG_QA_C001_1783993879279_3e2da15a8b9f`",
						"- **RetrievedAt:** 2026-07-14; exact clock time unavailable",
						"- **Source temporal metadata:** unknown; no creation/modification/publication metadata available through the permitted tools",
						"- **Temporal basis:** retrieval date only",
						"- **Uncertainty:** Low for file content and uniqueness within the discovered root; temporal metadata unavailable",
						"- **Discovery result:** `find` and `ls` returned exactly one file: `only-fixture.txt`",
					].join("\n"),
				}),
			}),
		});

		await expect(agent.searchDocuments(query)).resolves.toMatchObject({ answer: "[1] C001 sentinel", query });
	});

	it("accepts the latest Luna plain-text candidate findings label shape", async () => {
		const query = "identify the exact sentinel and source path from the only fixture";
		const agent = makeGroundingProbeAgent({
			finalOutput: [
				"Candidate findings:",
				"1. **Strong candidate**",
				"**Exact source path:** /private/tmp/autorag-precommit-real-cli-final-20260714/fixture/only-fixture.txt",
				"**Retrieval method:** Contained POSIX discovery and reading (find, ls, read)",
				"**Query variants used:** identify the only fixture and inspect its complete contents",
				"**Verbatim evidence:** Unique sentinel: AUTORAG_QA_C001_1783993879279_3e2da15a8b9f",
				"**Retrieved at:** 2026-07-14; exact clock time unavailable",
				"**Source temporal metadata:** unknown; no creation/modification/publication metadata available",
				"**Temporal-basis uncertainty:** retrieval date only",
				"**Duplicate/uniqueness evidence:** find and ls returned exactly one file",
				"No weak or conflicting candidates were returned.",
			].join("\n"),
		});

		await expect(agent.searchDocuments(query)).resolves.toMatchObject({ answer: "[1] grounding probe", query });
	});

	it.each(["**", "__"] as const)("accepts %s-emphasized Luna prose labels", async (emphasis) => {
		const label = (value: string) => `${emphasis}${value}${emphasis}`;
		const agent = makeGroundingProbeAgent({
			finalOutput: [
				`- ${label("Original query")}: grounding probe`,
				`- ${label("Selected retrieval method")}: posix`,
				`- ${label("Query variant")}: QZ-ORCHID validation note`,
				`- ${label("Source")}: /docs/a.md`,
				`- ${label("Exact supporting excerpt")}: QZ-ORCHID appears in the validation note.`,
				`- ${label("Retrieved at")}: 2026-07-14T05:30:00.000Z`,
				`- ${label("Source temporal metadata")}: unknown`,
				`- ${label("Diagnostics")}: one weak candidate omitted a page number.`,
			].join("\n"),
		});

		await expect(agent.searchDocuments("grounding probe")).resolves.toMatchObject({ answer: "[1] grounding probe" });
	});

	it.each(["## **Diagnostics**", "**Diagnostics**", "__Diagnostics__"] as const)(
		"rejects fields hidden behind a colonless emphasized Diagnostics heading %s",
		async (heading) => {
			const agent = makeGroundingProbeAgent({
				finalOutput: [
					heading,
					"Source: /docs/diagnostic.md",
					"Evidence: This diagnostic mirrors the handoff schema.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			});

			await expect(agent.searchDocuments("grounding probe")).rejects.toThrow(
				"requires a successful autorag-explorer subagent call",
			);
		},
	);

	it.each(["**", "__"] as const)("rejects a mismatched %s-emphasized Original query", async (emphasis) => {
		const agent = makeGroundingProbeAgent({
			finalOutput: [
				`- ${emphasis}Original query${emphasis}: unrelated prior search`,
				"Source: /docs/a.md",
				"Evidence: QZ-ORCHID appears in the validation note.",
				"Retrieved at: 2026-07-14T05:30:00.000Z",
				"Temporal metadata: source date is unknown.",
			].join("\n"),
		});

		await expect(agent.searchDocuments("grounding probe")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it.each([
		["colon mismatch", "Original query: unrelated prior search"],
		["equals mismatch", "Original query = unrelated prior search"],
		["dash mismatch", "Original query - unrelated prior search"],
		["pipe mismatch", "Original query | unrelated prior search"],
		["Markdown mismatch", "| Original query | unrelated prior search |"],
		["bold Markdown mismatch", "| **Original query:** | unrelated prior search |"],
		["colon placeholder", "Original query: original query goes here"],
		["equals placeholder", "Original query = <original query>"],
		["dash placeholder", "Original query - query placeholder"],
		["pipe placeholder", "Original query | same as above"],
		["Markdown placeholder", "| Original query | original query goes below |"],
		["bold Markdown placeholder", "| **Original query:** | original query goes below |"],
	] as const)("rejects an explicit %s declaration", async (_label, declaration) => {
		const agent = makeGroundingProbeAgent({
			finalOutput: [
				declaration,
				"Source: /docs/a.md",
				"Evidence: QZ-ORCHID appears in the validation note.",
				"Retrieved at: 2026-07-14T05:30:00.000Z",
				"Temporal metadata: source date is unknown.",
			].join("\n"),
		});

		await expect(agent.searchDocuments("grounding probe")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it("accepts an active query containing inline canonical assignment labels", async () => {
		const query =
			"Explain how Selected retrieval method: POSIX and Original query: values are treated as literal text while preserving this exact query.";
		const agent = makeGroundingProbeAgent({
			finalOutput: [
				"Source: /docs/a.md",
				"Evidence: QZ-ORCHID appears in the validation note.",
				"Retrieved at: 2026-07-14T05:30:00.000Z",
				"Temporal metadata: source date is unknown.",
			].join("\n"),
		});

		await expect(agent.searchDocuments(query)).resolves.toMatchObject({ answer: "[1] grounding probe", query });
	});

	it.each(["**", "__"] as const)(
		"rejects fields hidden behind a %s-emphasized Diagnostics section",
		async (emphasis) => {
			const agent = makeGroundingProbeAgent({
				finalOutput: [
					`- ${emphasis}Diagnostics${emphasis}: generated diagnostic metadata follows.`,
					"Source: /docs/diagnostic.md",
					"Evidence: This diagnostic mirrors the handoff schema.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			});

			await expect(agent.searchDocuments("grounding probe")).rejects.toThrow(
				"requires a successful autorag-explorer subagent call",
			);
		},
	);

	it.each([
		["<source path>", "QZ-ORCHID appears in the validation note."],
		["[source]", "QZ-ORCHID appears in the validation note."],
		["/docs/a.md", "{exact supporting excerpt}"],
		["/docs/a.md", "<evidence>"],
		["source here", "QZ-ORCHID appears in the validation note."],
		["/docs/a.md", "evidence here"],
	] as const)("rejects source/evidence template placeholders %s / %s", async (source, evidence) => {
		const agent = makeGroundingProbeAgent({
			finalOutput: [
				`Source: ${source}`,
				`Evidence: ${evidence}`,
				"Retrieved at: 2026-07-14T05:30:00.000Z",
				"Temporal metadata: unknown",
			].join("\n"),
		});

		await expect(agent.searchDocuments("grounding probe")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it.each([
		[
			"prose instructions",
			{
				finalOutput: [
					"Source: source path goes here",
					"Evidence: evidence excerpt goes below",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			},
		],
		[
			"Markdown table instructions",
			{
				finalOutput: [
					"| Source | Evidence | Retrieved at | Temporal metadata |",
					"| --- | --- | --- | --- |",
					"| source path to be provided | evidence excerpt placeholder | 2026-07-14T05:30:00.000Z | unknown |",
				].join("\n"),
			},
		],
	] as const)("rejects compositional source/evidence placeholders in %s", async (_label, handoff) => {
		const agent = makeGroundingProbeAgent(handoff);

		await expect(agent.searchDocuments("grounding probe")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it.each([
		[
			"prose source path here",
			{
				finalOutput: [
					"Source: path here",
					"Evidence: QZ-ORCHID appears in the validation note.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			},
		],
		[
			"prose source path below",
			{
				finalOutput: [
					"Source: path below",
					"Evidence: QZ-ORCHID appears in the validation note.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			},
		],
		[
			"prose evidence excerpt here",
			{
				finalOutput: [
					"Source: /docs/a.md",
					"Evidence: excerpt here",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			},
		],
		[
			"prose evidence excerpt below",
			{
				finalOutput: [
					"Source: /docs/a.md",
					"Evidence: excerpt below",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			},
		],
		[
			"Markdown source path here",
			{
				finalOutput: [
					"| Source | Evidence | Retrieved at | Temporal metadata |",
					"| --- | --- | --- | --- |",
					"| path here | QZ-ORCHID appears in the validation note. | 2026-07-14T05:30:00.000Z | unknown |",
				].join("\n"),
			},
		],
		[
			"Markdown source path below",
			{
				finalOutput: [
					"| Source | Evidence | Retrieved at | Temporal metadata |",
					"| --- | --- | --- | --- |",
					"| path below | QZ-ORCHID appears in the validation note. | 2026-07-14T05:30:00.000Z | unknown |",
				].join("\n"),
			},
		],
		[
			"Markdown evidence excerpt here",
			{
				finalOutput: [
					"| Source | Evidence | Retrieved at | Temporal metadata |",
					"| --- | --- | --- | --- |",
					"| /docs/a.md | excerpt here | 2026-07-14T05:30:00.000Z | unknown |",
				].join("\n"),
			},
		],
		[
			"Markdown evidence excerpt below",
			{
				finalOutput: [
					"| Source | Evidence | Retrieved at | Temporal metadata |",
					"| --- | --- | --- | --- |",
					"| /docs/a.md | excerpt below | 2026-07-14T05:30:00.000Z | unknown |",
				].join("\n"),
			},
		],
	] as const)("rejects short source/evidence placeholders in %s", async (_label, handoff) => {
		const agent = makeGroundingProbeAgent(handoff);

		await expect(agent.searchDocuments("grounding probe")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it("accepts a substantive datasource id and explicit temporal unknown", async () => {
		const agent = makeGroundingProbeAgent({
			finalOutput: [
				"Source: katok://room-123/message-456",
				"Evidence: QZ-ORCHID appears in the validation note.",
				"Retrieved at: 2026-07-14T05:30:00.000Z",
				"Temporal metadata: unknown",
			].join("\n"),
		});

		await expect(agent.searchDocuments("grounding probe")).resolves.toMatchObject({ answer: "[1] grounding probe" });
	});

	it.each([
		["canonical ExplorerReport structuredOutput", { structuredOutput: canonicalExplorerReport() }],
		[
			"canonical ExplorerReport JSON structuredOutput string",
			{ structuredOutput: JSON.stringify(canonicalExplorerReport()) },
		],
		[
			"ExplorerReport JSON embedded in finalOutput",
			{
				finalOutput: ["Explorer report:", "```json", JSON.stringify(canonicalExplorerReport()), "```"].join("\n"),
			},
		],
		[
			"later ExplorerReport JSON after an unmatched fragment",
			{
				finalOutput: [
					"Malformed diagnostic fragment: { not valid JSON",
					"Explorer report:",
					JSON.stringify(canonicalExplorerReport()),
				].join("\n"),
			},
		],
		[
			"grounded prose with an inline balanced locator object",
			{
				finalOutput: [
					'Locator: {"page":3,"section":"validation"}',
					"Source: /docs/a.md",
					"Evidence: QZ-ORCHID appears in the validation note.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: source date is unknown.",
				].join("\n"),
			},
		],
		[
			"grounded Markdown table handoff",
			{
				finalOutput: [
					"| Source | Evidence | Retrieved at | Temporal metadata |",
					"| --- | --- | --- | --- |",
					"| /docs/a.md | QZ-ORCHID appears in the validation note. | 2026-07-14T05:30:00.000Z | source date is unknown |",
				].join("\n"),
			},
		],
		[
			"grounded Markdown table handoff using Path as the source column",
			{
				finalOutput: [
					"| Path | Relevance | Evidence | retrievedAt | Temporal metadata |",
					"| --- | --- | --- | --- | --- |",
					"| /docs/a.md | strong | QZ-ORCHID appears in the validation note. | 2026-07-14 | unknown |",
				].join("\n"),
			},
		],
		[
			"complete documented prose handoff with intervening metadata",
			{
				finalOutput: [
					"Original query: grounding probe",
					"Method: posix",
					"Query variant: QZ-ORCHID validation note",
					"Source: /docs/a.md",
					"Relevance: strong",
					"Evidence: QZ-ORCHID appears in the validation note.",
					"Location context: page 3, validation section",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal basis: no source timestamp was available.",
					"Temporal metadata: source date is unknown.",
					"Uncertainty: the source does not state a publication date.",
				].join("\n"),
			},
		],
		[
			"grounded prose followed by a nonfatal Diagnostics section",
			{
				finalOutput: [
					"Original query: grounding probe",
					"Source: /docs/a.md",
					"Evidence: QZ-ORCHID appears in the validation note.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: source date is unknown.",
					"Diagnostics:",
					"One weak candidate omitted a page number.",
				].join("\n"),
			},
		],
	] as const)("accepts %s through searchDocuments", async (_label, handoff) => {
		const agent = makeGroundingProbeAgent(handoff);

		await expect(agent.searchDocuments("grounding probe")).resolves.toMatchObject({ answer: "[1] grounding probe" });
	});

	it.each([
		[
			"placeholder source and evidence text",
			{
				finalOutput: [
					"Source: none",
					"Evidence: unknown",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			},
		],
		[
			"Markdown handoff with an invalid qualified timestamp",
			{
				finalOutput: [
					"- **Source:** `/docs/a.md`",
					"- **Exact supporting excerpt:** QZ-ORCHID appears in the validation note.",
					"- **RetrievedAt:** 2026-99-99; exact clock time unavailable",
					"- **Source temporal metadata:** source date is unknown.",
				].join("\n"),
			},
		],
		[
			"Markdown diagnostics-only handoff",
			{
				finalOutput: [
					"- **Diagnostics:** no explorer handoff was returned.",
					"- **Source:** `/docs/diagnostic.md`",
					"- **Exact supporting excerpt:** This is diagnostic text, not source evidence.",
					"- **RetrievedAt:** 2026-07-14T05:30:00.000Z",
					"- **Source temporal metadata:** unknown",
				].join("\n"),
			},
		],
		[
			"scalar boolean and number fields",
			{
				structuredOutput: {
					source: true,
					evidence: 1,
					retrievedAt: false,
					temporalMetadata: 0,
				},
			},
		],
		["canonical report with empty candidates", { structuredOutput: canonicalExplorerReport([]) }],
		[
			"canonical report with placeholder candidate fields",
			{
				structuredOutput: canonicalExplorerReport([
					{
						source: "none",
						method: "posix",
						evidence: "unknown",
						retrievedAt: "2026-07-14T05:30:00.000Z",
						sourceTemporal: { status: "unknown" },
					},
				]),
			},
		],
		[
			"canonical report with a mismatched assignment query",
			{
				structuredOutput: canonicalExplorerReport(undefined, {
					originalQuery: "unrelated prior search",
				}),
				finalOutput: [
					"Source: /docs/a.md",
					"Evidence: QZ-ORCHID appears in the validation note.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: source date is unknown.",
				].join("\n"),
			},
		],
		[
			"canonical report with a placeholder query variant",
			{
				structuredOutput: canonicalExplorerReport(undefined, {
					queryVariant: "unknown",
				}),
			},
		],
		[
			"prose report with a mismatched original query",
			{
				finalOutput: [
					"Original query: unrelated prior search",
					"Source: /docs/a.md",
					"Evidence: QZ-ORCHID appears in the validation note.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: source date is unknown.",
				].join("\n"),
			},
		],
		[
			"Markdown prose report with a mismatched original query",
			{
				finalOutput: [
					"- **Original query:** unrelated prior search",
					"- **Source:** `/docs/a.md`",
					"- **Exact supporting excerpt:** QZ-ORCHID appears in the validation note.",
					"- **RetrievedAt:** 2026-07-14T05:30:00.000Z",
					"- **Source temporal metadata:** source date is unknown.",
				].join("\n"),
			},
		],
		[
			"canonical report with a non-substantive assignment method",
			{
				structuredOutput: canonicalExplorerReport(undefined, {
					method: "none",
				}),
			},
		],
		[
			"explicit diagnostic labels without a handoff",
			{
				finalOutput: [
					"Diagnostic only: no explorer handoff was returned.",
					"Source: /docs/diagnostic.md",
					"Evidence: This is a schema example, not source evidence.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			},
		],
		[
			"valid-looking fields only inside a Diagnostics section",
			{
				finalOutput: [
					"Diagnostics:",
					"Source: /docs/diagnostic.md",
					"Evidence: This diagnostic mirrors the handoff schema.",
					"Retrieved at: 2026-07-14T05:30:00.000Z",
					"Temporal metadata: unknown",
				].join("\n"),
			},
		],
		[
			"table-shaped diagnostic without a Markdown delimiter",
			{
				finalOutput: [
					"Diagnostic schema example; no handoff was returned.",
					"| Source | Evidence | Retrieved at | Temporal metadata |",
					"| /docs/diagnostic.md | Example only | 2026-07-14T05:30:00.000Z | unknown |",
				].join("\n"),
			},
		],
		[
			"Markdown table with a noncontiguous data row",
			{
				finalOutput: [
					"| Source | Evidence | Retrieved at | Temporal metadata |",
					"| --- | --- | --- | --- |",
					"Diagnostic note interrupts the table.",
					"| /docs/a.md | QZ-ORCHID appears in the validation note. | 2026-07-14T05:30:00.000Z | unknown |",
				].join("\n"),
			},
		],
		[
			"arbitrary nested diagnostic object",
			{
				structuredOutput: {
					diagnostics: {
						grounding: {
							source: "/docs/diagnostic.md",
							evidence: "Diagnostic text is not an explorer evidence handoff.",
							retrievedAt: "2026-07-14T05:30:00.000Z",
							temporalMetadata: "unknown",
						},
					},
				},
			},
		],
		[
			"arbitrary nested diagnostic JSON in finalOutput",
			{
				finalOutput: JSON.stringify({
					diagnostics: {
						source: "/docs/diagnostic.md",
						evidence: "Diagnostic text is not an explorer evidence handoff.",
						retrievedAt: "2026-07-14T05:30:00.000Z",
						temporalMetadata: "unknown",
					},
				}),
			},
		],
		[
			"nested valid ExplorerReport diagnostic object",
			{ structuredOutput: { diagnostics: canonicalExplorerReport() } },
		],
		[
			"nested valid ExplorerReport diagnostic JSON",
			{ finalOutput: JSON.stringify({ diagnostics: canonicalExplorerReport() }) },
		],
	] as const)("rejects %s through searchDocuments", async (_label, handoff) => {
		const agent = makeGroundingProbeAgent(handoff);

		await expect(agent.searchDocuments("grounding probe")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it("does not credit pi-subagents assignment or diagnostics without a substantive handoff", async () => {
		const model = fauxModel(
			emitResults({
				answer: "[1] diagnostic-only",
				results: [
					{ number: 1, title: "Diagnostic", summary: "only", evidence: [{ excerpt: "bad" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/docs/bad.md", method: "posix", content: "bad" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(undefined, {
				content: [
					{
						type: "text",
						text: "Diagnostic mentions source evidence retrievedAt temporal metadata, but no handoff was returned.",
					},
				],
				details: {
					mode: "single",
					results: [
						{
							agent: "autorag-explorer",
							task: "Return source evidence retrievedAt temporal metadata.",
							exitCode: 0,
							usage: { input: 100, output: 5, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 1 },
							finalOutput: "Explorer completed without a source handoff.",
							error: "Missing evidence and retrievedAt temporal metadata fields.",
						},
					],
				},
			}),
		});

		await expect(agent.searchDocuments("diagnostic-only handoff")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it("routes search through the mandatory session factory with the Luna explorer contract", async () => {
		let sessionPrompt = "";
		let sessionSystemPrompt = "";
		let sessionToolNames: readonly string[] = [];
		const model = fauxModel(
			emitResults({
				answer: "[1] Mandatory session result",
				results: [
					{ number: 1, title: "Mandatory", summary: "session", evidence: [{ excerpt: "session" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/session.txt", method: "bash", content: "session" }],
			}),
			fauxAssistantMessage("done", { stopReason: "stop" }),
		);
		const baseFactory = fauxSessionFactory();
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: async (options) => {
				sessionSystemPrompt = options.systemPrompt;
				sessionToolNames = options.tools.map((tool) => tool.name);
				const session = await baseFactory(options);
				return {
					...session,
					prompt: async (prompt) => {
						sessionPrompt = prompt;
						await session.prompt(prompt);
					},
				};
			},
		});

		const response = await agent.searchDocuments("What changed in the renewal policy?");

		expect(response.answer).toContain("Mandatory session result");
		expect(sessionToolNames).toContain(EMIT_AUTORAG_RESULTS_TOOL_NAME);
		expect(sessionSystemPrompt).toContain("pi-subagents");
		expect(sessionPrompt).toContain("faux/gpt-5.6-luna");
		expect(sessionPrompt).toMatch(/must use.*subagent/i);
		expect(sessionPrompt).toMatch(/original query/i);
		expect(sessionPrompt).toMatch(/retrieval method/i);
		expect(sessionPrompt).toMatch(/query variants/i);
		expect(sessionPrompt).toContain("<<<AUTORAG_ASSIGNMENT_V1>>>");
		expect(sessionPrompt).toContain('"originalQuery"');
		expect(sessionPrompt).toContain('"queryVariants"');
		expect(sessionPrompt).toMatch(/weak.*candidate/i);
		expect(sessionPrompt).toContain("retrievedAt");
		expect(sessionPrompt).toMatch(/temporal metadata/i);
		expect(sessionPrompt).toMatch(/orchestrator.*final/i);
	});

	it("forwards provider-scoped credentials without setting the orchestrator apiKey", async () => {
		let sessionApiKey: string | undefined;
		let sessionProviderApiKeys: Readonly<Record<string, string>> | undefined;
		const model = fauxModelFromSubagentArgs(
			(_provider, originalQuery) => ({
				agent: "autorag-explorer",
				agentScope: "user",
				model: "test-proxy/gpt-5.6-luna",
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
				artifacts: false,
			}),
			emitResults({
				answer: "[1] provider-scoped credential",
				results: [
					{
						number: 1,
						title: "Credential",
						summary: "provider-scoped",
						evidence: [{ excerpt: "ok" }],
						confidence: 1,
					},
				],
				mapping: [{ number: 1, source: "/data/credential.txt", method: "posix", content: "ok" }],
			}),
		);
		const baseFactory = fauxSessionFactory();
		const agent = new AutoRAGAgent({
			model,
			explorerModel: { ...model, provider: "test-proxy", id: "gpt-5.6-luna", name: "GPT-5.6 Luna" },
			providerApiKeys: { "test-proxy": "explorer-secret" },
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: async (options) => {
				sessionApiKey = options.apiKey;
				sessionProviderApiKeys = options.providerApiKeys;
				return baseFactory(options);
			},
		});

		const response = await agent.searchDocuments("provider-scoped credential");

		expect(response.answer).toContain("provider-scoped credential");
		expect(sessionApiKey).toBeUndefined();
		expect(sessionProviderApiKeys).toEqual({ "test-proxy": "explorer-secret" });
	});

	it.each([
		["missing cwd", undefined],
		["workspace root cwd", tmpDir],
	])("rejects grounded explorer results with %s outside configured search roots", async (_label, cwd) => {
		const model = fauxModelWithExplorerCwd(
			"gpt-5.6-luna",
			cwd,
			emitResults({
				answer: "[1] invalid root",
				results: [{ number: 1, title: "Invalid", summary: "root", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/invalid-root.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model);

		await expect(agent.searchDocuments("invalid root")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
	});

	it("accepts a grounded explorer result when cwd is a configured search root", async () => {
		const allowedRoot = resolve(FIXTURE_DIR);
		const model = fauxModelWithExplorerCwd(
			"gpt-5.6-luna",
			allowedRoot,
			emitResults({
				answer: "[1] allowed root",
				results: [{ number: 1, title: "Allowed", summary: "root", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/allowed-root.txt", method: "posix", content: "ok" }],
			}),
		);
		const agent = makeAgent(model);

		await expect(agent.searchDocuments("allowed root")).resolves.toMatchObject({ answer: "[1] allowed root" });
	});

	it("accepts a configured symlink root while it points to a directory", async () => {
		const realRoot = join(tmpDir, "documents-real");
		const linkedRoot = join(tmpDir, "documents-link");
		mkdirSync(realRoot);
		symlinkSync(realRoot, linkedRoot, "dir");
		const model = fauxModelWithExplorerCwd(
			"gpt-5.6-luna",
			linkedRoot,
			emitResults({
				answer: "[1] linked root",
				results: [{ number: 1, title: "Linked", summary: "root", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: join(realRoot, "linked.txt"), method: "posix", content: "ok" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [linkedRoot],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(),
		});

		await expect(agent.searchDocuments("linked root")).resolves.toMatchObject({ answer: "[1] linked root" });
	});

	it("rejects a missing configured search root before creating a search session", () => {
		const missingRoot = join(tmpDir, "missing-root");
		let sessionDispatches = 0;

		expect(
			() =>
				new AutoRAGAgent({
					searchPaths: [missingRoot],
					memoryPath: join(tmpDir, "memory.json"),
					workspacePath: tmpDir,
					sessionFactory: async () => {
						sessionDispatches += 1;
						throw new Error("unexpected session dispatch");
					},
				}),
		).toThrow(/AutoRAG search root does not exist.*missing-root/i);
		expect(sessionDispatches).toBe(0);
	});

	it("rejects a configured search root that is not a directory", () => {
		const fileRoot = join(tmpDir, "root.txt");
		writeFileSync(fileRoot, "not a directory", "utf8");

		expect(
			() =>
				new AutoRAGAgent({
					searchPaths: [fileRoot],
					memoryPath: join(tmpDir, "memory.json"),
					workspacePath: tmpDir,
				}),
		).toThrow(/AutoRAG search root is not a directory.*root\.txt/i);
	});

	it("uses the pinned canonical root in the prompt and explorer dispatch for a configured symlink", async () => {
		const realRoot = join(tmpDir, "canonical-documents");
		const linkedRoot = join(tmpDir, "canonical-documents-link");
		mkdirSync(realRoot);
		symlinkSync(realRoot, linkedRoot, "dir");
		const canonicalRoot = realpathSync(linkedRoot);
		let sessionPrompt = "";
		let dispatchedCwd: unknown;
		const model = fauxModelWithExplorerCwd(
			"gpt-5.6-luna",
			linkedRoot,
			emitResults({
				answer: "[1] canonical root",
				results: [{ number: 1, title: "Canonical", summary: "root", evidence: [{ excerpt: "ok" }], confidence: 1 }],
				mapping: [{ number: 1, source: join(canonicalRoot, "canonical.txt"), method: "posix", content: "ok" }],
			}),
		);
		const baseFactory = fauxSessionFactory((args) => {
			dispatchedCwd = typeof args === "object" && args !== null ? (args as Record<string, unknown>).cwd : undefined;
		});
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [linkedRoot],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: async (options) => {
				const session = await baseFactory(options);
				return {
					...session,
					prompt: async (prompt) => {
						sessionPrompt = prompt;
						await session.prompt(prompt);
					},
				};
			},
		});

		await expect(agent.searchDocuments("canonical root")).resolves.toMatchObject({ answer: "[1] canonical root" });
		expect(sessionPrompt).toContain(`Allowed explorer roots (normalized):\n- ${canonicalRoot}`);
		expect(sessionPrompt).not.toContain(`- ${linkedRoot}`);
		expect(dispatchedCwd).toBe(canonicalRoot);
	});

	it("blocks a retargeted configured symlink before explorer execution", async () => {
		const originalRoot = join(tmpDir, "original-root");
		const redirectedRoot = join(tmpDir, "redirected-root");
		const linkedRoot = join(tmpDir, "mutable-root");
		mkdirSync(originalRoot);
		mkdirSync(redirectedRoot);
		symlinkSync(originalRoot, linkedRoot, "dir");
		let explorerExecutions = 0;
		const model = fauxModelWithExplorerCwd(
			"gpt-5.6-luna",
			linkedRoot,
			emitResults({
				answer: "[1] redirected root",
				results: [
					{ number: 1, title: "Redirected", summary: "root", evidence: [{ excerpt: "bad" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: join(redirectedRoot, "redirected.txt"), method: "posix", content: "bad" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [linkedRoot],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(() => {
				explorerExecutions += 1;
			}),
		});
		unlinkSync(linkedRoot);
		symlinkSync(redirectedRoot, linkedRoot, "dir");

		await expect(agent.searchDocuments("retargeted root")).rejects.toThrow(
			"requires a successful autorag-explorer subagent call",
		);
		expect(explorerExecutions).toBe(0);
	});

	it("normalizes a lexical /tmp explorer cwd to the pinned canonical root", async () => {
		const tmpAliasRoot = mkdtempSync(join("/tmp", "autorag-pinned-root-"));
		try {
			const canonicalRoot = realpathSync(tmpAliasRoot);
			let dispatchedCwd: unknown;
			const model = fauxModelWithExplorerCwd(
				"gpt-5.6-luna",
				tmpAliasRoot,
				emitResults({
					answer: "[1] tmp root",
					results: [{ number: 1, title: "Tmp", summary: "root", evidence: [{ excerpt: "ok" }], confidence: 1 }],
					mapping: [{ number: 1, source: join(canonicalRoot, "tmp.txt"), method: "posix", content: "ok" }],
				}),
			);
			const agent = new AutoRAGAgent({
				model,
				searchPaths: [tmpAliasRoot],
				memoryPath: join(tmpDir, "memory.json"),
				workspacePath: tmpDir,
				sessionFactory: fauxSessionFactory((args) => {
					dispatchedCwd =
						typeof args === "object" && args !== null ? (args as Record<string, unknown>).cwd : undefined;
				}),
			});

			await expect(agent.searchDocuments("tmp root")).resolves.toMatchObject({ answer: "[1] tmp root" });
			expect(dispatchedCwd).toBe(canonicalRoot);
		} finally {
			rmSync(tmpAliasRoot, { recursive: true, force: true });
		}
	});

	it("includes virtual path scope in the agent search prompt", () => {
		const agent = makeAgent(fauxModel());
		const prompt = agent.buildSearchPrompt("refund policy", { topK: 3, scope: "/docs/policies" });
		expect(prompt).toContain("Return at most 3 curated results");
		expect(prompt).toContain("Restrict search to virtual path scope /docs/policies");
		expect(prompt).toContain(`Allowed explorer roots (normalized):\n- ${realpathSync(FIXTURE_DIR)}`);
		expect(prompt).toMatch(/every autorag-explorer task must set an explicit cwd/i);
		expect(prompt).toMatch(/missing or null top-level artifacts.*safely autofilled/i);
		expect(prompt).toMatch(/explicit wrong values remain rejected/i);
		expect(prompt).toMatch(/one task per root/i);
		expect(prompt).toMatch(/never use the workspace root.*outside these roots/i);
	});
	it("returns structured curated results emitted by the agent loop without leaking paths", async () => {
		const model = fauxModel(
			emitResults({
				answer: "[1] Meeting notes summary",
				results: [
					{
						number: 1,
						title: "Meeting notes from 2024-01-15",
						summary: "Planning sync covering deadlines",
						evidence: [{ excerpt: "Meeting notes from 2024-01-15", lineNumber: 1 }],
						confidence: 1,
					},
				],
				mapping: [
					{ number: 1, source: "/data/notes.txt", method: "grep", content: "Meeting notes from 2024-01-15" },
				],
			}),
		);
		const agent = makeAgent(model);

		const response = await agent.searchDocuments("Meeting", { topK: 1 });

		expect(response.query).toBe("Meeting");
		expect(response.sessionId).toMatch(/[0-9a-f-]{36}/);
		expect(response.results).toEqual([
			{
				number: 1,
				title: "Meeting notes from 2024-01-15",
				summary: "Planning sync covering deadlines",
				evidence: [{ excerpt: "Meeting notes from 2024-01-15", lineNumber: 1 }],
				confidence: 1,
				feedbackId: `${response.sessionId}:1`,
			},
		]);
		expect(response.answer).toBe("[1] Meeting notes summary");
		expect(response.searched).toBe(1);
		expect(response.warnings).toEqual([]);
		// Source paths live only in the internal registry, never in the public response.
		expect(JSON.stringify(response)).not.toContain("/data/notes.txt");
		expect(JSON.stringify(response)).not.toContain("/Users/");
	});

	it("surfaces a path-free diagnostic when caller tools are dropped", async () => {
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 0.5 }],
				mapping: [{ number: 1, source: "/data/a.txt", method: "grep", content: "a" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			tools: [callerTool("bash"), callerTool("grep"), callerTool("search_all_documents")],
			sessionFactory: fauxSessionFactory(),
		});

		const response = await agent.searchDocuments("Meeting");
		const diagnostic = response.diagnostics?.find((item) => item.code === "caller-tool-dropped");
		expect(diagnostic).toMatchObject({ severity: "info", source: "tools" });
		expect(JSON.stringify(diagnostic)).not.toContain(tmpDir);
		expect(JSON.stringify(diagnostic)).not.toContain("bash");
	});

	it("populates the session registry from the structured tool mapping", async () => {
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [
					{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 0.5 },
					{ number: 2, title: "B", summary: "b", evidence: [{ excerpt: "b" }], confidence: 0.5 },
				],
				mapping: [
					{ number: 1, source: "/data/a.txt", method: "grep", content: "a" },
					{ number: 2, source: "/data/b.txt", method: "posix", content: "b" },
				],
			}),
		);
		const agent = makeAgent(model);

		const response = await agent.searchDocuments("Meeting");
		const registry = agent.getResultRegistry(response.sessionId);

		expect(registry.get(1)).toMatchObject({ index: 1, source: "/data/a.txt", method: "grep", content: "a" });
		expect(registry.get(1)?.evidenceRefs?.[0]).toMatchObject({ method: "grep", source: "/data/a.txt", content: "a" });
		expect(registry.get(2)).toMatchObject({ index: 2, source: "/data/b.txt", method: "posix", content: "b" });
		expect(registry.get(2)?.evidenceRefs?.[0]).toMatchObject({
			method: "posix",
			source: "/data/b.txt",
			content: "b",
		});
	});

	it("returns an empty structured response when query is blank without running the agent", async () => {
		// Missing model configuration would fail if retrieval ran; blank query must short-circuit.
		const memoryPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [tmpDir],
			memoryPath,
			workspacePath: tmpDir,
		});

		const response = await agent.searchDocuments("   ", { topK: 1 });

		expect(response.query).toBe("");
		expect(response.results).toEqual([]);
		expect(response.answer).toBe("");
		expect(response.searched).toBe(0);
		expect(response.warnings).toEqual(["empty-query"]);
		expect(readRunEvents(memoryPath)).toEqual([]);
	});

	it("throws when the agent completes without emitting structured results", async () => {
		const memoryPath = join(tmpDir, "memory.json");
		const model = fauxModel(fauxAssistantMessage("I could not find anything.", { stopReason: "stop" }));
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments("Meeting")).rejects.toThrow(
			"AutoRAG agent completed without emitting structured results",
		);
		expect(readRunEvents(memoryPath).map((event) => event.event)).toEqual(["search_started", "search_failed"]);
	});

	it("throws when result numbers and mapping numbers are not one-to-one", async () => {
		const memoryPath = join(tmpDir, "memory.json");
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
				mapping: [
					{ number: 1, source: "/data/a.txt", method: "grep", content: "a" },
					{ number: 2, source: "/data/b.txt", method: "grep", content: "b" },
				],
			}),
		);
		const agent = makeAgent(model, memoryPath);

		await expect(agent.searchDocuments("Meeting")).rejects.toThrow(/one-to-one/);
		const events = readRunEvents(memoryPath);
		expect(events.map((event) => event.event)).toEqual(["search_started", "search_failed"]);
		expect(events.filter((event) => event.event === "search_failed")).toHaveLength(1);
	});

	it("does not leak prior-run results into a later no-output run", async () => {
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/a.txt", method: "grep", content: "a" }],
			}),
			fauxAssistantMessage("No structured output this time.", { stopReason: "stop" }),
		);
		const agent = makeAgent(model);

		const first = await agent.searchDocuments("Meeting");
		expect(agent.getResultRegistry(first.sessionId).size).toBe(1);

		await expect(agent.searchDocuments("Second")).rejects.toThrow(
			"AutoRAG agent completed without emitting structured results",
		);
		// The earlier session registry is untouched by the failed run.
		expect(agent.getResultRegistry(first.sessionId).size).toBe(1);
	});

	it("rejects a concurrent search without mutating the in-flight session", async () => {
		const memoryPath = join(tmpDir, "memory.json");
		let release: (() => void) | undefined;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		const model = fauxModel(async () => {
			await gate;
			return fauxAssistantMessage(
				[
					fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
						answer: "answer",
						results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
						mapping: [{ number: 1, source: "/data/a.txt", method: "grep", content: "a" }],
					}),
				],
				{ stopReason: "toolUse" },
			);
		});
		const agent = makeAgent(model, memoryPath);

		const inFlight = agent.searchDocuments("first");
		await expect(agent.searchDocuments("second")).rejects.toThrow(/busy/);

		release?.();
		const response = await inFlight;
		expect(response.query).toBe("first");
		expect(response.results).toHaveLength(1);
		expect(readRunEvents(memoryPath).map((event) => event.event)).toEqual(["search_started", "search_completed"]);
	});

	it("records search result numbers for feedback resolution", async () => {
		const memPath = join(tmpDir, "memory.json");
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/a.txt", method: "grep", content: "a" }],
			}),
		);
		const agent = makeAgent(model, memPath);
		const response = await agent.searchDocuments("Meeting");

		agent.recordFeedbackByNumbers(response.sessionId, [1]);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getMethodHints("Meeting").find((hint) => hint.method === "grep")?.score).toBeGreaterThan(0);
	});

	it("keeps feedback state independent for each returned feedback id", async () => {
		const memPath = join(tmpDir, "memory.json");
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [
					{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 },
					{ number: 2, title: "B", summary: "b", evidence: [{ excerpt: "b" }], confidence: 1 },
					{ number: 3, title: "C", summary: "c", evidence: [{ excerpt: "c" }], confidence: 1 },
				],
				mapping: [
					{ number: 1, source: "/data/a.txt", method: "grep", content: "a" },
					{ number: 2, source: "/data/b.txt", method: "grep", content: "b" },
					{ number: 3, source: "/data/c.txt", method: "grep", content: "c" },
				],
			}),
		);
		const agent = makeAgent(model, memPath);
		const response = await agent.searchDocuments("function|Meeting");

		agent.recordFeedbackByNumbers(response.sessionId, [1]);
		agent.recordFeedbackByNumbers(response.sessionId, [2]);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getSchema().curatedResults).toHaveLength(3);
		expect(memory.getSchema().feedbackSignals.filter((signal) => signal.source === "explicit")).toHaveLength(4);
		expect(memory.getMethodHints("function|Meeting").find((hint) => hint.method === "grep")?.score).toBeGreaterThan(
			0,
		);
	});
	it("preserves real paths verbatim in answer, title, summary, and evidence (#6)", async () => {
		const home = homedir();
		const dotAutorag = join(tmpDir, ".autorag", "parsed", "x.md");
		const model = fauxModel(
			emitResults({
				answer: `Located at ${tmpDir}/data/notes.txt and home ${home}/priv and id /data/notes.txt`,
				results: [
					{
						number: 1,
						title: `Notes under ${tmpDir}/data`,
						summary: `Cached at ${dotAutorag} and home ${home}/secret`,
						evidence: [{ excerpt: `line from ${tmpDir}/data/notes.txt` }],
						confidence: 1,
					},
				],
				mapping: [{ number: 1, source: "/data/notes.txt", method: "grep", content: "notes" }],
			}),
		);
		const agent = makeAgent(model);

		const response = await agent.searchDocuments("Meeting");
		const blob = JSON.stringify(response);

		// Real paths flow through verbatim — path opacity is removed.
		expect(blob).toContain(tmpDir);
		expect(blob).toContain(home);
		expect(blob).toContain("/data/notes.txt");
		// Hidden registry keeps the original source untouched for feedback.
		expect(agent.getResultRegistry(response.sessionId).get(1)?.source).toBe("/data/notes.txt");
	});

	it("does not redact benign path-like text (#6 negative cases)", async () => {
		const benignAnswer =
			"See docs/policy and https://example.com/a/b and C:\\Temp\\note.txt; run `cat docs/policy`; ref [1]; product Notes.txt";
		const model = fauxModel(
			emitResults({
				answer: benignAnswer,
				results: [
					{
						number: 1,
						title: "docs/policy overview",
						summary: "Visit https://example.com/a/b for details",
						evidence: [{ excerpt: "cat docs/policy" }],
						confidence: 1,
					},
				],
				mapping: [{ number: 1, source: "/internal/opaque-source-id", method: "grep", content: "x" }],
			}),
		);
		const agent = makeAgent(model);

		const response = await agent.searchDocuments("Meeting");

		expect(response.answer).toBe(benignAnswer);
		expect(response.results[0]?.title).toBe("docs/policy overview");
		expect(response.results[0]?.summary).toBe("Visit https://example.com/a/b for details");
		expect(response.results[0]?.evidence[0]?.excerpt).toBe("cat docs/policy");
	});

	it("always populates diagnostics at runtime even with nothing to redact (#6/#21)", async () => {
		const model = fauxModel(
			emitResults({
				answer: "clean answer",
				results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/a.txt", method: "grep", content: "a" }],
			}),
		);
		const agent = makeAgent(model);
		const response = await agent.searchDocuments("Meeting");
		expect(Array.isArray(response.diagnostics)).toBe(true);
	});

	it("empty-query response still includes an empty diagnostics array (#6/#21)", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [tmpDir],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
		});
		const response = await agent.searchDocuments("   ");
		expect(response.warnings).toEqual(["empty-query"]);
		expect(response.diagnostics).toEqual([]);
	});
	it("routes unknown emitted warnings to diagnostics instead of dropping them (#21)", async () => {
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/a.txt", method: "grep", content: "a" }],
				warnings: ["empty-query", "some-unusual-warning"],
			}),
		);
		const agent = makeAgent(model);
		const response = await agent.searchDocuments("Meeting");

		expect(response.warnings).toEqual(["empty-query"]);
		const unknown = response.diagnostics?.find((d) => d.code === "unknown-warning");
		expect(unknown?.severity).toBe("info");
		expect(unknown?.message).toContain("some-unusual-warning");
	});

	it("surfaces a bm25-degraded-fallback diagnostic when BM25 runs in fallback mode (#21)", async () => {
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/a.txt", method: "grep", content: "a" }],
			}),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			bm25: { forceEngine: "typescript-fallback" },
			sessionFactory: fauxSessionFactory(),
		});
		await agent.refresh(true);

		const response = await agent.searchDocuments("Meeting");
		const bm25 = response.diagnostics?.find((d) => d.source === "bm25");
		expect(bm25?.code).toBe("bm25-degraded-fallback");
		expect(JSON.stringify(response.diagnostics)).not.toContain(tmpDir);
	});

	it("[red-team] preserves every real path verbatim alongside benign lookalikes (#6)", async () => {
		const home = homedir();
		const dot = join(tmpDir, ".autorag");
		const model = fauxModel(
			emitResults({
				answer: `x${tmpDir}/a mid-word, ${home}/h, ${dot}/p, id/data/secret.txt, but keep docs/policy and https://ex.com/a`,
				results: [
					{
						number: 1,
						title: `t ${tmpDir}/z and product Report.txt`,
						summary: `s ${home}/q and ref [2]`,
						evidence: [{ excerpt: `e ${dot}/idx and cat docs/policy` }],
						confidence: 1,
					},
				],
				mapping: [{ number: 1, source: "/data/secret.txt", method: "grep", content: "c" }],
			}),
		);
		const agent = makeAgent(model);
		const response = await agent.searchDocuments("Meeting");
		const blob = JSON.stringify(response);

		// Real paths are preserved verbatim — path opacity is removed.
		expect(blob).toContain(tmpDir);
		expect(blob).toContain(home);
		expect(blob).toContain("/data/secret.txt");
		// Benign lookalikes are still preserved.
		expect(response.answer).toContain("docs/policy");
		expect(response.answer).toContain("https://ex.com/a");
		expect(response.results[0]?.title).toContain("product Report.txt");
		expect(response.results[0]?.summary).toContain("[2]");
		expect(response.results[0]?.evidence[0]?.excerpt).toContain("cat docs/policy");
	});
	it("does not parse or index in the query path (no refresh/syncParsedMirrors during searchDocuments) (#22)", async () => {
		const model = fauxModel(
			emitResults({
				answer: "answer",
				results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
				mapping: [{ number: 1, source: "/data/a.txt", method: "grep", content: "a" }],
			}),
		);
		const agent = makeAgent(model);
		const refreshSpy = vi.spyOn(agent, "refresh");
		const syncSpy = vi.spyOn(agent, "syncParsedMirrors");

		await agent.searchDocuments("Meeting");

		expect(refreshSpy).not.toHaveBeenCalled();
		expect(syncSpy).not.toHaveBeenCalled();
	});

	it("composes an existing subagent prepareArguments before AutoRAG preparation", async () => {
		// The AutoRAG clone must call the original tool's prepareArguments first
		// (if it exists), then pass its result into autoragPrepare. This test
		// proves the composition order: the original prepare sets a harmless
		// marker field, and AutoRAG preparation still normalizes the launch
		// envelope (artifacts=false, agentScope=user) without losing the marker.
		let capturedArgs: Record<string, unknown> | undefined;
		let prepareMarkerSeen = false;

		const model = fauxModelFromSubagentArgs(
			(provider, originalQuery) => ({
				agent: "autorag-explorer",
				model: `${provider}/gpt-5.6-luna`,
				task: explorerAssignment(originalQuery),
				cwd: resolve(FIXTURE_DIR),
				// Intentionally omit artifacts and agentScope so AutoRAG autofill runs.
			}),
			emitResults({
				answer: "[1] composed prepare",
				results: [
					{ number: 1, title: "Composed", summary: "prepare", evidence: [{ excerpt: "ok" }], confidence: 1 },
				],
				mapping: [{ number: 1, source: "/data/composed.txt", method: "posix", content: "ok" }],
			}),
		);

		// Session factory that adds a prepareArguments to the subagent tool.
		// The marker proves the original prepare ran before AutoRAG preparation.
		const composedSessionFactory: AutoRAGSessionFactory = async (options) => {
			const explorerModel = options.explorerModel;
			if (explorerModel === undefined) throw new Error("Test session requires an explorer model");
			const subagentTool: AgentTool = {
				name: "subagent",
				label: "Subagent",
				description: "Test explorer with prepareArguments",
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
				prepareArguments: (rawArgs: unknown) => {
					if (rawArgs !== null && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
						const args = rawArgs as Record<string, unknown>;
						// Harmless marker: supply a safe field the upstream tool would set.
						if (!("artifacts" in args)) {
							args.__prepareMarker = "upstream-prepare-ran";
						}
					}
					return rawArgs as Record<string, unknown>;
				},
				execute: async (_toolCallId, params) => {
					capturedArgs = params as Record<string, unknown>;
					if (
						typeof capturedArgs === "object" &&
						capturedArgs !== null &&
						capturedArgs.__prepareMarker === "upstream-prepare-ran"
					) {
						prepareMarkerSeen = true;
					}
					return piSubagentsExplorerResult(explorerModel.provider, {
						finalOutput:
							"source: /docs/a evidence: grounded retrievedAt: 2026-07-13T00:00:00.000Z temporal metadata: unknown",
					});
				},
			};
			const agent = new Agent({
				initialState: {
					systemPrompt: options.systemPrompt,
					model: options.model,
					tools: [subagentTool, ...options.tools],
				},
				convertToLlm: (messages) =>
					messages.filter(
						(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
					),
			});
			return {
				agent,
				prompt: async (prompt: string) => agent.prompt(prompt),
				abort: async () => agent.abort(),
				dispose: () => {},
			};
		};

		const agent = new AutoRAGAgent({
			model,
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: composedSessionFactory,
		});

		await expect(agent.searchDocuments("composed prepare")).resolves.toMatchObject({
			answer: "[1] composed prepare",
		});

		// The original prepareArguments ran (marker survived composition).
		expect(prepareMarkerSeen).toBe(true);
		// AutoRAG preparation normalized the launch envelope.
		expect(capturedArgs).toMatchObject({ artifacts: false, agentScope: "user" });
	});
});
