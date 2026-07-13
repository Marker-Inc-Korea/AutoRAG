import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
	registerFauxProvider,
} from "@earendil-works/pi-ai";
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

function fauxModel(...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	reg.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall("subagent", {
					agent: "autorag-explorer",
					model: "faux/gpt-5.6-luna",
					task: "Original query: test Selected retrieval method: POSIX query variants: test retrievedAt temporal metadata",
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

function fauxSessionFactory(): NonNullable<ConstructorParameters<typeof AutoRAGAgent>[0]["sessionFactory"]> {
	return async (options) => {
		const subagentTool: AgentTool = {
			name: "subagent",
			label: "Subagent",
			description: "Test explorer",
			parameters: Type.Object({ agent: Type.String(), model: Type.String(), task: Type.String() }),
			execute: async () => ({
				content: [
					{
						type: "text",
						text: "source: /docs/a evidence: grounded retrievedAt: 2026-07-13 temporal metadata: unknown",
					},
				],
				details: {},
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
			model: fauxModel(emitOne()),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(),
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
				abortAware,
				fauxAssistantMessage(
					[
						fauxToolCall("subagent", {
							agent: "autorag-explorer",
							model: "faux/gpt-5.6-luna",
							task: "Original query: test Selected retrieval method: POSIX query variants: test retrievedAt temporal metadata",
						}),
					],
					{ stopReason: "toolUse" },
				),
				emitOne(),
			),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
			sessionFactory: fauxSessionFactory(),
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
