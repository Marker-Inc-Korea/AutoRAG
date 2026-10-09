import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import type {
	DatasourceIndexResult,
	DatasourceSkill,
	PollingMetadata,
	SourceDescription,
} from "../../src/datasource/types.ts";
import type { RetrievalMemory } from "../../src/memory/memory.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalResult } from "../../src/retrieval/types.ts";

let root: string;
let registration: FauxProviderRegistration;

const remoteRow: RetrievalResult = {
	id: "remote-1",
	source: "/remote/chunks/1",
	content: "remote result",
	score: 1,
	metadata: { method: "remote.keyword" },
};

class StaticRemoteMethod implements RetrievalMethod {
	describe(): RetrievalMethodDescriptor {
		return {
			name: "remote.keyword",
			type: "bm25",
			description: "Remote test datasource method",
			status: "active",
			capabilities: ["keyword"],
			datasourceId: "remote",
			tags: ["remote"],
		};
	}
	async retrieve(): Promise<RetrievalResult[]> {
		return [remoteRow];
	}
}

// A real datasource retrieval step so the model has evidence a tool returned
// this run: remote sessions may not cite local files, only tool evidence.
const remoteSkill: DatasourceSkill = {
	describe: () => ({
		name: "remote",
		type: "chat",
		description: "Remote fixture",
		capabilities: ["keyword", "polling"],
		tags: ["remote"],
		status: "active",
		datasourceId: "remote",
		instanceId: "default",
		instances: ["default"],
	}),
	polling: (): PollingMetadata => ({ mode: "poll", intervalMs: 60_000 }),
	skillManifest: () => ({
		name: "datasource-remote",
		description: "Remote fixture",
		content: "Search indexed remote chats with search_datasource_remote.",
	}),
	index: async (): Promise<DatasourceIndexResult> => ({
		ok: true,
		instanceId: "default",
		skill: "remote",
		chunkCount: 1,
		indexedAt: 1,
		diagnostics: [],
	}),
	retrievalMethods: () => [new StaticRemoteMethod()],
	describeSources: (): readonly SourceDescription[] => [
		{
			source: "/remote/default",
			datasourceId: "remote",
			skill: "remote",
			instanceId: "default",
			contentType: "chat",
			metadata: { description: "Remote fixture" },
		},
	],
};

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-remote-memory-"));
	registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "remote-memory" }] });
	registration.setResponses([
		fauxAssistantMessage([fauxToolCall("search_datasource_remote", { query: "remote result" })], {
			stopReason: "toolUse",
		}),
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
							refs: [remoteRow.source],
						},
					],
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
			datasourceSkills: [remoteSkill],
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
