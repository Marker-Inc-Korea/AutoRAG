import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
	registerFauxProvider,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-search-documents-test-"));
	registrations = [];
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(tmpDir, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	reg.setResponses(responses);
	registrations.push(reg);
	return reg.getModel();
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
}

function emitResults(args: EmitArgs): FauxResponseStep {
	return fauxAssistantMessage([fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, args)], { stopReason: "toolUse" });
}

function makeAgent(model: ReturnType<typeof fauxModel>, memoryPath = join(tmpDir, "memory.json")) {
	return new AutoRAGAgent({ model, searchPaths: [FIXTURE_DIR], memoryPath, workspacePath: tmpDir });
}

describe("AutoRAGAgent searchDocuments", () => {
	it("includes virtual path scope in the agent search prompt", () => {
		const agent = makeAgent(fauxModel());
		const prompt = agent.buildSearchPrompt("refund policy", { topK: 3, scope: "/docs/policies" });
		expect(prompt).toContain("Return at most 3 curated results");
		expect(prompt).toContain("Restrict search to virtual path scope /docs/policies");
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
		// Missing source path would fail if retrieval ran; blank query must short-circuit.
		const agent = new AutoRAGAgent({
			searchPaths: [join(tmpDir, "missing-source")],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
		});

		const response = await agent.searchDocuments("   ", { topK: 1 });

		expect(response.query).toBe("");
		expect(response.results).toEqual([]);
		expect(response.answer).toBe("");
		expect(response.searched).toBe(0);
		expect(response.warnings).toEqual(["empty-query"]);
	});

	it("throws when the agent completes without emitting structured results", async () => {
		const model = fauxModel(fauxAssistantMessage("I could not find anything.", { stopReason: "stop" }));
		const agent = makeAgent(model);

		await expect(agent.searchDocuments("Meeting")).rejects.toThrow(
			"AutoRAG agent completed without emitting structured results",
		);
	});

	it("throws when result numbers and mapping numbers are not one-to-one", async () => {
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
		const agent = makeAgent(model);

		await expect(agent.searchDocuments("Meeting")).rejects.toThrow(/one-to-one/);
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
		const agent = makeAgent(model);

		const inFlight = agent.searchDocuments("first");
		await expect(agent.searchDocuments("second")).rejects.toThrow(/busy/);

		release?.();
		const response = await inFlight;
		expect(response.query).toBe("first");
		expect(response.results).toHaveLength(1);
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
});
