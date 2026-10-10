import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type AssistantMessage,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
	type Model,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_FAST_ANSWER_TOOL_NAME } from "../../src/agent/fast-answer-tool.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";

// A search that times out during verification must not throw away the first
// answer it already produced.

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-timeout-"));
	docs = join(root, "docs");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "fromis.txt"), "fromis_9 debuted with nine members and now has five.\n");
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]): Model<string> {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model", reasoning: true }] });
	reg.setResponses(responses);
	registrations.push(reg);
	return reg.getModel();
}

const fastAnswer: FauxResponseStep = fauxAssistantMessage(
	[
		fauxToolCall(EMIT_FAST_ANSWER_TOOL_NAME, {
			answer: "- fromis_9 has five members [1]",
			results: [
				{
					number: 1,
					title: "Members",
					summary: "fromis_9 now has five members.",
					evidence: [{ excerpt: "now has five", lineNumber: 1 }],
					confidence: 0.8,
				},
			],
			sources: [{ number: 1, source: join(tmpdir(), "fromis.txt") }],
		}),
	],
	{ stopReason: "toolUse" },
);

/** A model turn that never completes, so verification outlives the timeout. */
const hang: FauxResponseStep = () => new Promise<AssistantMessage>(() => undefined);

function agentFor(model: Model<string>): AutoRAGAgent {
	return new AutoRAGAgent({
		model,
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { autoInstall: false },
		jikji: false,
		searchTimeoutMs: 1_500,
	});
}

describe("search timeout after a first answer", () => {
	it("returns the first answer with a search-timeout diagnostic instead of rejecting", async () => {
		const agent = agentFor(
			fauxModel(fastAnswer, fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }), hang),
		);

		const response = await agent.searchDocuments("프로미스 나인 총 몇명이지.");

		expect(response.answer).toBe("- fromis_9 has five members [1]");
		expect(response.results.map((result) => result.number)).toEqual([1]);
		expect(response.diagnostics).toContainEqual(
			expect.objectContaining({ code: "search-timeout", severity: "warning" }),
		);
	});

	it("streams the preliminary and then a complete event carrying the same answer", async () => {
		const agent = agentFor(
			fauxModel(fastAnswer, fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }), hang),
		);
		const events: SearchDocumentsStreamEvent[] = [];
		for await (const event of agent.searchDocumentsStream("프로미스 나인 총 몇명이지.")) events.push(event);

		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("missing complete event");
		expect(complete.response.answer).toBe("- fromis_9 has five members [1]");
		expect(complete.response.diagnostics?.some((d) => d.code === "search-timeout")).toBe(true);
	});

	it("records the timeout answer so its result registry still resolves", async () => {
		const agent = agentFor(
			fauxModel(fastAnswer, fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }), hang),
		);

		const response = await agent.searchDocuments("프로미스 나인 총 몇명이지.");
		expect(response.diagnostics).toContainEqual(
			expect.objectContaining({ code: "search-timeout", severity: "warning" }),
		);

		const registry = agent.getResultRegistry(response.sessionId);
		expect(registry.get(1)?.content).toBe("now has five");
	});

	it("does not leak a timed-out run's preliminary into a later search", async () => {
		const agent = agentFor(
			fauxModel(fastAnswer, fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }), hang),
		);
		await agent.searchDocuments("프로미스 나인 총 몇명이지.");

		const second = agentFor(fauxModel(fastAnswer, fauxAssistantMessage("done", { stopReason: "stop" })));
		const settled: SearchDocumentsStreamEvent[] = [];
		for await (const event of second.searchDocumentsStream("두번째 검색")) {
			if (event.type === "preliminary" || event.type === "complete") settled.push(event);
		}
		const complete = settled.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("missing complete event");
		expect(complete.response.query).toBe("두번째 검색");
		expect(settled.every((event) => event.type !== "preliminary" || event.response.query === "두번째 검색")).toBe(
			true,
		);
	});

	it("still rejects when the search times out before any answer exists", async () => {
		const agent = agentFor(fauxModel(hang));

		await expect(agent.searchDocuments("프로미스 나인 총 몇명이지.")).rejects.toThrow(
			"search timed out after 1500ms",
		);
	});
});
