import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-lifecycle-test-"));
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

function fauxModel(originalQuery: string, ...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	reg.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall("subagent", {
					agent: "autorag-explorer",
					agentScope: "user",
					model: "faux/gpt-5.6-luna",
					task: explorerAssignment(originalQuery),
					cwd: resolve(FIXTURE_DIR),
					artifacts: false,
				}),
			],
			{ stopReason: "toolUse" },
		),
		...responses,
	]);
	registrations.push(reg);
	return reg.getModel();
}

function emitOne(): FauxResponseStep {
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
}

function fauxSessionFactory(
	originalQuery: string,
): NonNullable<ConstructorParameters<typeof AutoRAGAgent>[0]["sessionFactory"]> {
	return async (options) => {
		const subagentTool: AgentTool = {
			name: "subagent",
			label: "Subagent",
			description: "Test explorer",
			parameters: Type.Object({
				agent: Type.String(),
				agentScope: Type.Literal("user"),
				model: Type.String(),
				task: Type.String(),
				cwd: Type.Optional(Type.String()),
				artifacts: Type.Optional(Type.Boolean()),
			}),
			execute: async () => ({
				content: [{ type: "text", text: "Explorer run completed successfully." }],
				details: {
					mode: "single",
					runId: "run-lifecycle",
					results: [
						{
							agent: "autorag-explorer",
							task: explorerAssignment(originalQuery),
							exitCode: 0,
							usage: { input: 100, output: 50, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 1 },
							model: "faux/gpt-5.6-luna",
							structuredOutput: {
								assignment: {
									originalQuery,
									method: "posix",
									queryVariant: "Meeting notes",
									queryVariants: ["Meeting", "Meeting notes"],
								},
								evidenceCandidates: [
									{
										source: resolve(FIXTURE_DIR, "data/notes.txt"),
										method: "posix",
										evidence:
											"Meeting notes from 2024-01-15 list action items to review the PR and update docs.",
										retrievedAt: "2026-07-14T05:30:00.000Z",
										sourceTemporal: { status: "asOf", asOf: "2024-01-15T00:00:00.000Z" },
										locator: "line 1-2",
									},
								],
								summary: "The meeting notes contain two concrete follow-up action items.",
							},
						},
					],
				},
			}),
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
			prompt: async (prompt) => agent.prompt(prompt),
			abort: async () => agent.abort(),
			dispose: () => {},
		};
	};
}

describe("AutoRAGAgent lifecycle", () => {
	it("subscribe observes events from the in-flight agent run", async () => {
		const agent = new AutoRAGAgent({
			model: fauxModel("Meeting", emitOne()),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory("Meeting"),
		});

		const eventTypes: string[] = [];
		const unsubscribe = agent.subscribe((event) => {
			eventTypes.push(event.type);
		});

		await agent.searchDocuments("Meeting");
		unsubscribe();

		expect(eventTypes.length).toBeGreaterThan(0);
	});

	it("abort cancels the in-flight run and the agent recovers for the next search", async () => {
		const abortAware: FauxResponseStep = (_context, options) =>
			new Promise((_resolve, reject) => {
				options?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
			});
		const agent = new AutoRAGAgent({
			model: fauxModel(
				"first",
				abortAware,
				fauxAssistantMessage(
					[
						fauxToolCall("subagent", {
							agent: "autorag-explorer",
							agentScope: "user",
							model: "faux/gpt-5.6-luna",
							task: explorerAssignment("second"),
							cwd: resolve(FIXTURE_DIR),
							artifacts: false,
						}),
					],
					{ stopReason: "toolUse" },
				),
				emitOne(),
			),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory("second"),
		});

		const inFlight = agent.searchDocuments("first");
		const rejected = expect(inFlight).rejects.toThrow();
		await new Promise((resolve) => setTimeout(resolve, 10));
		agent.abort();
		await rejected;

		// The busy guard is cleared, so a subsequent search runs normally.
		const response = await agent.searchDocuments("second");
		expect(response.query).toBe("second");
		expect(response.results).toHaveLength(1);
	});
});
