import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	type Model,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { assertResultsMappingOneToOne } from "../../src/agent/citations.ts";
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
		minSync: { autoInstall: false },
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

describe("answer citations resolve to results (#1788)", () => {
	it("derives results from cited evidence so every remaining number resolves", async () => {
		// The model cites a real file it read plus ids no tool returned; the
		// derivation keeps [1] and drops the rest, so no [n] is left dangling.
		const model = fauxModel(
			fauxAssistantMessage(
				`- fromis_9 has five members [file:${join(docs, "fromis.txt")}] and also [e99] and [7].`,
				{ stopReason: "stop" },
			),
			fauxAssistantMessage(`- verified: five members [file:${join(docs, "fromis.txt")}].`, { stopReason: "stop" }),
		);
		const agent = new AutoRAGAgent(agentOptions(model));
		const events: SearchDocumentsStreamEvent[] = [];
		for await (const event of agent.searchDocumentsStream("프로미스 나인 총 몇명이지.")) events.push(event);

		const preliminary = events.find((event) => event.type === "preliminary");
		const complete = events.find((event) => event.type === "complete");
		if (preliminary?.type !== "preliminary" || complete?.type !== "complete") throw new Error("missing events");
		for (const response of [preliminary.response, complete.response]) {
			expectCitationsResolve(response);
			expect(response.diagnostics).toContainEqual(
				expect.objectContaining({ code: "citation-without-result", severity: "warning" }),
			);
		}
		expect(complete.response.results.map((result) => result.number)).toEqual([1]);
		expect(complete.response.answer).not.toContain("[e99]");
		expect(complete.response.answer).not.toContain("[7]");
	});

	it("strips unmatched citations from externally curated reports and reports a diagnostic", () => {
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [docs],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				minSync: { autoInstall: false },
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

// The structured report path (`recordStructuredResultsSession`) still rejects
// a report whose result numbers and mapping numbers are not one-to-one.
describe("assertResultsMappingOneToOne duplicates", () => {
	const entries = (...numbers: number[]) => numbers.map((number) => ({ number }));

	it("accepts unique, equal number sets in any order", () => {
		expect(() => assertResultsMappingOneToOne("emit", entries(2, 1, 3), entries(3, 1, 2))).not.toThrow();
	});

	it("rejects identical duplicates on both sides and names the repeated numbers", () => {
		expect(() => assertResultsMappingOneToOne("emit", entries(1, 1, 2), entries(1, 1, 2))).toThrow(
			/results repeat \[1\]; mapping repeats \[1\]/u,
		);
	});

	it("rejects a duplicate on one side only", () => {
		expect(() => assertResultsMappingOneToOne("emit", entries(1, 2), entries(1, 2, 2))).toThrow(
			/mapping repeats \[2\]/u,
		);
	});

	it("rejects equal-length sets with different numbers", () => {
		expect(() => assertResultsMappingOneToOne("emit", entries(1, 2), entries(1, 3))).toThrow(/one-to-one/u);
	});
});
