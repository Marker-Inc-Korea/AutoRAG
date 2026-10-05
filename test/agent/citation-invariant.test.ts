import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
	type Model,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { EMIT_FAST_ANSWER_TOOL_NAME } from "../../src/agent/fast-answer-tool.ts";
import type { SearchDocumentsResponse, SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import { createAutoRAGLite } from "../../src/core.ts";

// Issue #1788: every bracketed citation in `answer` must resolve to a
// `results[].number` of the same response, in both the preliminary (fast) and
// complete (deep) phase.

const CITATION = /\[(\d+)\](?!\()/gu;

function expectCitationsResolve(response: SearchDocumentsResponse): void {
	const numbers = new Set(response.results.map((result) => result.number));
	for (const match of response.answer.matchAll(CITATION)) expect(numbers.has(Number(match[1]))).toBe(true);
}

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-citations-"));
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

function agentOptions(model: Model<string>): AutoRAGAgentOptions {
	return {
		model,
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: false,
		jikji: false,
	};
}

const unit = (number: number) => ({
	number,
	title: `Unit ${number}`,
	summary: "fromis_9 now has five members.",
	evidence: [{ excerpt: "now has five", lineNumber: 1 }],
	confidence: 0.8,
});

function fastCall(answer: string, numbers: readonly number[]): FauxResponseStep {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_FAST_ANSWER_TOOL_NAME, {
				answer,
				results: numbers.map(unit),
				sources: numbers.map((number) => ({ number, source: join(docs, "fromis.txt") })),
			}),
		],
		{ stopReason: "toolUse" },
	);
}

function finalCall(answer: string, numbers: readonly number[]): FauxResponseStep {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
				answer,
				results: numbers.map(unit),
				mapping: numbers.map((number) => ({
					number,
					source: join(docs, "fromis.txt"),
					method: "bash",
					content: "fromis_9 now has five members.",
				})),
			}),
		],
		{ stopReason: "toolUse" },
	);
}

describe("answer citations resolve to results (#1788)", () => {
	it("rejects mismatched emits in both phases so the model re-emits one numbering", async () => {
		// The issue's real run: fast cites [6] with results [1]; deep cites [1]..[6] with results [1][2].
		const model = fauxModel(
			fastCall("- fromis_9 has five members [6]", [1]),
			fastCall("- fromis_9 has five members [1]", [1]),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalCall("- Correction: debuted with nine [1][2][3][4][5][6]", [1, 2]),
			finalCall("- Correction: debuted with nine [1][2]", [1, 2]),
		);
		const agent = new AutoRAGAgent(agentOptions(model));
		const events: SearchDocumentsStreamEvent[] = [];
		for await (const event of agent.searchDocumentsStream("프로미스 나인 총 몇명이지.")) events.push(event);

		const preliminary = events.find((event) => event.type === "preliminary");
		const complete = events.find((event) => event.type === "complete");
		if (preliminary?.type !== "preliminary" || complete?.type !== "complete") throw new Error("missing events");
		expect(preliminary.response.answer).toBe("- fromis_9 has five members [1]");
		expect(complete.response.answer).toBe("- Correction: debuted with nine [1][2]");
		expectCitationsResolve(preliminary.response);
		expectCitationsResolve(complete.response);
		expect(complete.response.diagnostics?.some((d) => d.code === "citation-without-result")).toBe(false);
	});

	it("strips unmatched citations from externally curated reports and reports a diagnostic", () => {
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [docs],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				minSync: false,
				jikji: false,
				fsearch: false,
			}),
		);
		const lite = createAutoRAGLite({ flags: { config: configPath }, cwd: root });
		const image = "![chart](</data/[7] chart.png>) [2]";
		const response = lite.recordReport("q", {
			answer: `- five members [1][7]\n- debuted with nine [8]\n${image}`,
			results: [unit(1), unit(2)],
			mapping: [1, 2].map((number) => ({
				number,
				source: join(docs, "fromis.txt"),
				method: "fixture",
				content: "c",
				evidenceRefs: [],
			})),
			warnings: [],
		});

		expect(response.answer).toBe(`- five members [1]\n- debuted with nine\n${image}`);
		expect(response.diagnostics).toContainEqual(
			expect.objectContaining({ code: "citation-without-result", severity: "warning" }),
		);
		expect(response.diagnostics?.find((d) => d.code === "citation-without-result")?.message).toContain("[7], [8]");
	});
});
