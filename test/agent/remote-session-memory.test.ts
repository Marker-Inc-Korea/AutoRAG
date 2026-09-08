import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import type { RetrievalMemory } from "../../src/memory/memory.ts";

let root: string;
let registration: FauxProviderRegistration;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-remote-memory-"));
	registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "remote-memory" }] });
	registration.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
					answer: "[1] remote result",
					results: [
						{
							number: 1,
							title: "Remote result",
							summary: "remote result",
							evidence: [{ excerpt: "remote result" }],
							confidence: 1,
						},
					],
					mapping: [{ number: 1, source: "/docs/remote.txt", method: "search", content: "remote result" }],
				}),
			],
			{ stopReason: "toolUse" },
		),
	]);
});

afterEach(() => {
	registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

interface AgentInternals {
	memory: RetrievalMemory;
	sessions: Map<string, { transient?: boolean }>;
	withMemoryContext(messages: unknown[]): Promise<unknown[]>;
}

function internals(agent: AutoRAGAgent): AgentInternals {
	return agent as unknown as AgentInternals;
}

describe("AutoRAGAgent remote-session memory isolation", () => {
	it("reads and writes the shared memory during a remote search", async () => {
		const memoryPath = join(root, "memory.json");
		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath,
			remoteSession: true,
			jikji: false,
			minSync: false,
			bm25: false,
		});
		const memory = internals(agent).memory;
		memory.recordWeakSignal("seed", "search", "followup");
		memory.save();
		const before = readFileSync(memoryPath);
		const recordCuratedResultsSession = vi.spyOn(memory, "recordCuratedResultsSession");
		const save = vi.spyOn(memory, "save");

		const response = await agent.searchDocuments("remote query");

		expect(internals(agent).sessions.get(response.sessionId)?.transient).not.toBe(true);
		expect(recordCuratedResultsSession).toHaveBeenCalled();
		expect(save).toHaveBeenCalled();
		expect(existsSync(memoryPath)).toBe(true);
		expect(readFileSync(memoryPath)).not.toEqual(before);
	});
});
