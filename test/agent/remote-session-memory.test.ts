import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
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
	it("does not read or write memory during a remote search", async () => {
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
		const beforeMtime = statSync(memoryPath).mtimeMs;
		const signalCount = memory.getSignalCount();
		const recordWeakSignal = vi.spyOn(memory, "recordWeakSignal");
		const recordCuratedResultsSession = vi.spyOn(memory, "recordCuratedResultsSession");
		const save = vi.spyOn(memory, "save");
		const input = [{ role: "user", content: [{ type: "text", text: "query" }], timestamp: 1 }];
		await expect(internals(agent).withMemoryContext(input)).resolves.toBe(input);

		const response = await agent.searchDocuments("remote query");

		expect(internals(agent).sessions.get(response.sessionId)?.transient).toBe(true);
		expect(recordWeakSignal).not.toHaveBeenCalled();
		expect(recordCuratedResultsSession).not.toHaveBeenCalled();
		expect(save).not.toHaveBeenCalled();
		expect(memory.getSignalCount()).toBe(signalCount);
		expect(existsSync(memoryPath)).toBe(true);
		expect(readFileSync(memoryPath)).toEqual(before);
		expect(statSync(memoryPath).mtimeMs).toBe(beforeMtime);
	});
});
