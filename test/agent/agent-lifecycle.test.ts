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
	reg.setResponses(responses);
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

describe("AutoRAGAgent lifecycle", () => {
	it("subscribe observes events from the in-flight agent run", async () => {
		const agent = new AutoRAGAgent({
			model: fauxModel(emitOne()),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
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
			model: fauxModel(abortAware, emitOne()),
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
		});

		const inFlight = agent.searchDocuments("first");
		await new Promise((resolve) => setTimeout(resolve, 10));
		agent.abort();
		await expect(inFlight).rejects.toThrow();

		// The busy guard is cleared, so a subsequent search runs normally.
		const response = await agent.searchDocuments("second");
		expect(response.query).toBe("second");
		expect(response.results).toHaveLength(1);
	});
});
