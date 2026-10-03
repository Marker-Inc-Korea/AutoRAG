import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall, type Message } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";

let root: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-llm-transcript-"));
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function groundedEmit(answer: string) {
	return fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
		answer: `[1] ${answer}`,
		results: [
			{
				number: 1,
				title: "Result",
				summary: answer,
				evidence: [{ excerpt: answer }],
				confidence: 0.9,
			},
		],
		mapping: [{ number: 1, source: "/docs/a.txt", method: "bash", content: answer }],
	});
}

function toolNamesDeclaredToModel(messages: readonly Message[]): string[] {
	const names: string[] = [];
	for (const message of messages) {
		if (message.role !== "system") continue;
		for (const tool of message.toolsAdded ?? []) names.push(tool.name);
	}
	return names;
}

const isMissingFinalEmit = (diagnostic: { code: string }): boolean => diagnostic.code === "missing-final-emit";

describe("provider transcript", () => {
	it("declares the system prompt and the tools the agent needs to emit curated results", async () => {
		const registration = registerFauxProvider({
			api: `faux-${randomUUID()}`,
			models: [{ id: "transcript-model" }],
		});
		const providerTranscripts: { roles: string[]; declaredTools: string[] }[] = [];
		registration.setResponses([
			(context) => {
				providerTranscripts.push({
					roles: context.messages.map((message) => message.role),
					declaredTools: toolNamesDeclaredToModel(context.messages),
				});
				return fauxAssistantMessage([groundedEmit("grounded answer")], { stopReason: "toolUse" });
			},
		]);
		registrations.push(registration);

		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
			minSync: false,
		});

		const response = await agent.searchDocuments("find the grounded answer");

		expect(providerTranscripts.length).toBeGreaterThan(0);
		for (const transcript of providerTranscripts) {
			expect(transcript.roles).toContain("system");
			expect(transcript.declaredTools).toContain(EMIT_AUTORAG_RESULTS_TOOL_NAME);
		}
		expect(response.answer).toBe("[1] grounded answer");
		expect(response.results).toHaveLength(1);
		expect(response.diagnostics?.some(isMissingFinalEmit) ?? false).toBe(false);
	});
});
