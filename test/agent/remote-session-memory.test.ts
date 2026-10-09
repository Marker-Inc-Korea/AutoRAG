import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";

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

type TestMessage = { role: "user"; content: { type: "text"; text: string }[]; timestamp: number };

interface AgentInternals {
	memory: RetrievalMemory;
	lastQuery: string | undefined;
	sessions: Map<string, { transient?: boolean }>;
	withMemoryContext(messages: TestMessage[]): Promise<TestMessage[]>;
}

function internals(agent: AutoRAGAgent): AgentInternals {
	return agent as unknown as AgentInternals;
}

describe("AutoRAGAgent remote-session memory isolation", () => {
	it("neither reads nor writes retrieval memory during a remote search", async () => {
		const memoryPath = join(root, "memory.json");
		const seed = new RetrievalMemory({ storagePath: memoryPath });
		seed.recordCuratedResultsSession({
			sessionId: "seed-session",
			query: "seed query",
			results: [
				{
					number: 1,
					title: "Seed result",
					summary: "seed summary",
					content: "seed content",
					method: "posix",
					source: "/docs/seed.md",
					evidenceRefs: [
						normalizeSessionEvidenceRef({ method: "posix", source: "/docs/seed.md", content: "seed content" }),
					],
				},
			],
		});
		seed.recordJudgedEvidence([
			{
				id: "past:remote",
				sessionId: "past-session",
				conversationId: "past-conversation",
				question: "remote query",
				searchQuery: "remote query",
				method: "search",
				source: "/docs/remote.txt",
				stableEvidenceId: "remote",
				resultNumber: 1,
				title: "Past remote evidence",
				excerpt: "past remote evidence",
				probability: 0.9,
				createdAt: 1,
			},
		]);
		seed.save();
		expect(existsSync(memoryPath)).toBe(true);
		const before = readFileSync(memoryPath);

		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath,
			remoteSession: true,
			memoryEmbedder: false,
			jikji: false,
			minSync: false,
		});
		const memory = internals(agent).memory;
		const recordCuratedResultsSession = vi.spyOn(memory, "recordCuratedResultsSession");
		const save = vi.spyOn(memory, "save");

		const response = await agent.searchDocuments("remote query");

		expect(internals(agent).sessions.get(response.sessionId)?.transient).not.toBe(true);
		expect(recordCuratedResultsSession).not.toHaveBeenCalled();
		expect(save).not.toHaveBeenCalled();
		expect(readFileSync(memoryPath)).toEqual(before);
	});

	it("leaves the transformed context unchanged even with matching judged evidence on disk", async () => {
		const memoryPath = join(root, "memory.json");
		const seed = new RetrievalMemory({ storagePath: memoryPath });
		seed.recordCuratedResultsSession({
			sessionId: "seed-session",
			query: "seed query",
			results: [
				{
					number: 1,
					title: "Seed result",
					summary: "seed summary",
					content: "seed content",
					method: "posix",
					source: "/docs/seed.md",
					evidenceRefs: [
						normalizeSessionEvidenceRef({ method: "posix", source: "/docs/seed.md", content: "seed content" }),
					],
				},
			],
		});
		seed.recordJudgedEvidence([
			{
				id: "past:remote",
				sessionId: "past-session",
				conversationId: "past-conversation",
				question: "remote query",
				searchQuery: "remote query",
				method: "search",
				source: "/docs/remote.txt",
				stableEvidenceId: "remote",
				resultNumber: 1,
				title: "Past remote evidence",
				excerpt: "past remote evidence",
				probability: 0.9,
				createdAt: 1,
			},
		]);
		seed.save();

		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath,
			remoteSession: true,
			memoryEmbedder: false,
			jikji: false,
			minSync: false,
		});
		internals(agent).lastQuery = "remote query";

		const messages: TestMessage[] = [{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: 1 }];
		const transformed = await internals(agent).withMemoryContext(messages);

		expect(transformed).toEqual(messages);
		expect(transformed[0].content[0].text).not.toContain("<memory_context>");
	});
});
