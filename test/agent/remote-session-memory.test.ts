import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type {
	DatasourceIndexResult,
	DatasourceSkill,
	PollingMetadata,
	SourceDescription,
} from "../../src/datasource/types.ts";
import type { JudgedEvidenceRecord } from "../../src/memory/judged-evidence.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";
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
		fauxAssistantMessage("Initial pass.", { stopReason: "stop" }),
		fauxAssistantMessage([fauxToolCall("search_datasource_remote", { query: "remote result" })], {
			stopReason: "toolUse",
		}),
		fauxAssistantMessage("remote result [e1]", { stopReason: "stop" }),
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
	tools: AgentTool[];
}

function internals(agent: AutoRAGAgent): AgentInternals {
	return agent as unknown as AgentInternals;
}

function checkMemoryText(agent: AutoRAGAgent, query: string): Promise<string> {
	const tool = internals(agent).tools.find((candidate) => candidate.name === "check_memory");
	if (tool === undefined) throw new Error("expected check_memory tool");
	return tool
		.execute("call-1", { query })
		.then((result) => result.content.map((part) => (part.type === "text" ? part.text : "")).join("\n"));
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
			minSync: { autoInstall: false },
			datasourceSkills: [remoteSkill],
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
			minSync: { autoInstall: false },
		});
		internals(agent).lastQuery = "remote query";

		const messages: TestMessage[] = [{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: 1 }];
		const transformed = await internals(agent).withMemoryContext(messages);

		expect(transformed).toEqual(messages);
		expect(transformed[0].content[0].text).not.toContain("<memory_context>");
	});

	it("returns no judged evidence through check_memory during a remote search", async () => {
		const memoryPath = join(root, "memory.json");
		const searchPath = "test/fixtures/sample-project";
		const seed = new RetrievalMemory({ storagePath: memoryPath });
		const privateRecord: JudgedEvidenceRecord = {
			id: "past:private",
			sessionId: "past-session",
			conversationId: "past-conversation",
			question: "remote query",
			searchQuery: "remote query",
			method: "minsync",
			source: join(resolve(searchPath), "private.md"),
			stableEvidenceId: "private",
			resultNumber: 1,
			title: "Private evidence",
			excerpt: "private local evidence",
			probability: 0.9,
			createdAt: 1,
		};
		seed.recordJudgedEvidence([privateRecord]);
		seed.save();
		const before = readFileSync(memoryPath);
		const options = {
			model: registration.getModel(),
			searchPaths: [searchPath],
			workspacePath: root,
			memoryPath,
			memoryEmbedder: false as const,
			jikji: false as const,
			minSync: { autoInstall: false },
		};

		const local = new AutoRAGAgent(options);
		const remote = new AutoRAGAgent({ ...options, remoteSession: true });

		expect(await checkMemoryText(local, "remote query")).toContain("private local evidence");
		const remoteText = await checkMemoryText(remote, "remote query");
		expect(remoteText).not.toContain("private local evidence");
		expect(remoteText).not.toContain("remote query");
		expect(readFileSync(memoryPath)).toEqual(before);
	});
});
