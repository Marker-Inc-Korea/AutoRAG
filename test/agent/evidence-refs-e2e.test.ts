import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { SEARCH_ALL_DOCUMENTS_TOOL_NAME } from "../../src/agent/search-all-tool.ts";
import type { DatasourceIndexResult, DatasourceSkill, PollingMetadata } from "../../src/datasource/types.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../src/retrieval/types.ts";

let root: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-evidence-e2e-"));
	registrations = [];
	mkdirSync(join(root, "docs"), { recursive: true });
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

const CHUNK = "Refund exceptions require director approval before payout.";

function skill(): DatasourceSkill {
	const method: RetrievalMethod = {
		describe(): RetrievalMethodDescriptor {
			return {
				name: "kakao.keyword",
				type: "bm25",
				description: "test method",
				status: "active",
				capabilities: ["keyword"],
				datasourceId: "kakao",
				tags: ["kakao"],
			};
		},
		async retrieve(_query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> {
			return [
				{
					id: "kakao:chunk-1",
					source: "/kakao/acct-1/chunks/1",
					content: CHUNK,
					score: 1,
					metadata: { method: "kakao-lexical", datasourceId: "kakao" },
				},
			];
		},
	};
	return {
		describe: () => ({
			name: "kakao",
			type: "chat",
			description: "chats",
			capabilities: ["keyword"],
			tags: ["kakao"],
			status: "active",
			datasourceId: "kakao",
			instanceId: "acct-1",
			instances: ["acct-1"],
		}),
		polling: (): PollingMetadata => ({ mode: "none" }),
		skillManifest: () => ({ name: "datasource-kakao", description: "chats", content: "# kakao" }),
		index: async (): Promise<DatasourceIndexResult> => ({
			ok: true,
			instanceId: "acct-1",
			skill: "kakao",
			chunkCount: 1,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [method],
		describeSources: () => [
			{
				source: "/kakao/acct-1",
				datasourceId: "kakao",
				skill: "kakao",
				instanceId: "acct-1",
				contentType: "chat",
				metadata: {},
			},
		],
	};
}

function model(...steps: FauxResponseStep[]) {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "single-agent" }] });
	registration.setResponses(steps);
	registrations.push(registration);
	return registration.getModel();
}

function emit(refs: string[], excerpt = "a loose paraphrase the model wrote") {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
				answer: "- Director approval is required [1]",
				results: [
					{
						number: 1,
						title: "Refund rule",
						summary: "Director approval",
						evidence: [{ excerpt }],
						confidence: 0.9,
						refs,
					},
				],
			}),
		],
		{ stopReason: "toolUse" },
	);
}

function search() {
	return fauxAssistantMessage([fauxToolCall(SEARCH_ALL_DOCUMENTS_TOOL_NAME, { query: "refund approval" })], {
		stopReason: "toolUse",
	});
}

function agentFor(m: AutoRAGAgentOptions["model"]) {
	return new AutoRAGAgent({
		model: m,
		searchPaths: [join(root, "docs")],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		jikji: false,
		minSync: false,
		datasourceSkills: [skill()],
	});
}

describe("evidence refs through the real agent loop", () => {
	it("stores the harness-recorded source and chunk, not what the model wrote", async () => {
		const agent = agentFor(model(search(), emit(["e1"])));

		const response = await agent.searchDocuments("refund approval");

		expect(response.results).toHaveLength(1);
		const entry = agent.getResultRegistry(response.sessionId).get(1);
		expect(entry?.source).toBe("/kakao/acct-1/chunks/1");
		expect(entry?.method).toBe("kakao-lexical");
		expect(entry?.content).toBe(CHUNK);
		expect(entry?.evidenceRefs?.[0]?.retrievalResultId).toBe("kakao:chunk-1");
		// The paraphrase stays the user-facing evidence excerpt only.
		expect(response.results[0]?.evidence[0]?.excerpt).toBe("a loose paraphrase the model wrote");
	});

	it("makes the model re-emit when it cites an id no tool issued, instead of storing a guess", async () => {
		const agent = agentFor(model(search(), emit(["e99"]), emit(["e1"])));

		const response = await agent.searchDocuments("refund approval");

		expect(response.diagnostics?.some((d) => d.code === ("missing-final-emit" as never))).toBe(false);
		expect(agent.getResultRegistry(response.sessionId).get(1)?.source).toBe("/kakao/acct-1/chunks/1");
	});

	it("rejects an invented local path and accepts a real file the model opened itself", async () => {
		const real = join(root, "docs", "real.txt");
		writeFileSync(real, CHUNK);
		const agent = agentFor(model(emit([join(root, "docs", "invented.txt")]), emit([real])));

		const response = await agent.searchDocuments("refund approval");

		const entry = agent.getResultRegistry(response.sessionId).get(1);
		expect(entry?.source).toBe(real);
		expect(entry?.method).toBe("bash");
	});

	it("forgets a previous run's ids, so a stale id cannot resolve in the next run", async () => {
		const agent = agentFor(model(search(), emit(["e1"]), emit(["e1"]), search(), emit(["e1"])));

		await agent.searchDocuments("first run");
		// Second run: the model cites e1 before any tool ran this run -> rejected, then it searches and re-cites.
		const second = await agent.searchDocuments("second run");

		expect(agent.getResultRegistry(second.sessionId).get(1)?.source).toBe("/kakao/acct-1/chunks/1");
	});
});
