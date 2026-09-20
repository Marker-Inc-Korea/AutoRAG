import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	createSingleDatasourceSearchTools,
	type SingleDatasourceSearchProvider,
	singleDatasourceToolName,
} from "../../src/agent/search-single-datasource-tool.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import type { DatasourceIndexResult, DatasourceSkill, PollingMetadata } from "../../src/datasource/types.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-single-datasource-test-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function result(id: string, source: string): RetrievalResult {
	return { id, source, content: `message ${id}`, score: 1, metadata: {} };
}

/** A datasource skill whose method records how often it was executed. */
function makeSpySkill(
	datasourceId: string,
	rows: readonly RetrievalResult[],
	calls: { count: number },
): DatasourceSkill {
	return {
		describe: () => ({
			name: datasourceId,
			type: "chat",
			description: `${datasourceId} test connection`,
			capabilities: ["keyword", "polling"],
			tags: [datasourceId],
			status: "active",
			datasourceId,
			instanceId: "default",
		}),
		polling: (): PollingMetadata => ({ mode: "none" }),
		skillManifest: () => ({
			name: `datasource-${datasourceId}`,
			description: `Search ${datasourceId}.`,
			content: `# ${datasourceId}`,
		}),
		index: async (): Promise<DatasourceIndexResult> => ({
			ok: true,
			instanceId: "default",
			skill: datasourceId,
			chunkCount: rows.length,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [
			{
				describe: () => ({
					name: `${datasourceId}.keyword`,
					type: "bm25" as const,
					description: `${datasourceId} keyword method`,
					status: "active" as const,
					capabilities: ["keyword"],
					datasourceId,
					tags: [datasourceId],
				}),
				retrieve: async (_query: string, options: { readonly topK?: number }) => {
					calls.count += 1;
					return rows.slice(0, options.topK ?? rows.length);
				},
			},
		],
		describeSources: () => [
			{
				source: `/${datasourceId}/default`,
				datasourceId,
				skill: datasourceId,
				instanceId: "default",
				contentType: "chat",
				metadata: {},
			},
		],
	};
}

describe("singleDatasourceToolName", () => {
	it("sanitizes connection aliases into tool names", () => {
		expect(singleDatasourceToolName("discord")).toBe("search_datasource_discord");
		expect(singleDatasourceToolName("kakao-work")).toBe("search_datasource_kakao_work");
	});
});

describe("createSingleDatasourceSearchTools", () => {
	it("routes execution to the provider with the spec's datasource id", async () => {
		const calls: { datasourceId?: string; query?: string; topK?: number } = {};
		const provider: SingleDatasourceSearchProvider = {
			searchSingleDatasourceDocuments: async (datasourceId, query, options) => {
				calls.datasourceId = datasourceId;
				calls.query = query;
				calls.topK = options?.topK;
				return { results: [result("a", "/kakao/default/chunks/a")], diagnostics: [] };
			},
		};
		const [tool] = createSingleDatasourceSearchTools(provider, [
			{ datasourceId: "kakao", description: "KakaoTalk chats", instanceScopes: ["/kakao/default"] },
		]);

		expect(tool?.name).toBe("search_datasource_kakao");
		expect(tool?.description).toContain("/kakao/default");

		const outcome = await tool?.execute("call-1", { query: "deployment", topK: 5 });
		expect(calls).toEqual({ datasourceId: "kakao", query: "deployment", topK: 5 });
		expect(outcome?.details.resultCount).toBe(1);
		expect(outcome?.details.datasource).toBe("kakao");
	});

	it("short-circuits an empty query without calling the provider", async () => {
		let called = false;
		const provider: SingleDatasourceSearchProvider = {
			searchSingleDatasourceDocuments: async () => {
				called = true;
				return { results: [], diagnostics: [] };
			},
		};
		const [tool] = createSingleDatasourceSearchTools(provider, [
			{ datasourceId: "slack", description: "Slack chats", instanceScopes: [] },
		]);

		const outcome = await tool?.execute("call-1", { query: "   " });
		expect(called).toBe(false);
		expect(outcome?.details.resultCount).toBe(0);
	});
});

describe("AutoRAGAgent single-datasource retrieval", () => {
	it("runs only the targeted datasource's methods", async () => {
		const kakaoCalls = { count: 0 };
		const slackCalls = { count: 0 };
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			minSync: false,
			datasourceSkills: [
				makeSpySkill("kakao", [result("a", "/kakao/default/chunks/a")], kakaoCalls),
				makeSpySkill("slack", [result("s", "/slack/default/chunks/s")], slackCalls),
			],
			datasourceAccess: { allowedTags: ["kakao", "slack"] },
		});

		const { results } = await agent.searchSingleDatasourceDocuments("kakao", "message");

		expect(results.map((r) => r.source)).toEqual(["/kakao/default/chunks/a"]);
		expect(kakaoCalls.count).toBe(1);
		expect(slackCalls.count).toBe(0);
	});

	it("returns an empty result set for an unauthorized datasource id", async () => {
		const kakaoCalls = { count: 0 };
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			minSync: false,
			datasourceSkills: [makeSpySkill("kakao", [result("a", "/kakao/default/chunks/a")], kakaoCalls)],
			datasourceAccess: { allowedTags: [] },
		});

		const { results, diagnostics } = await agent.searchSingleDatasourceDocuments("kakao", "message");

		expect(results).toEqual([]);
		expect(diagnostics).toEqual([]);
		expect(kakaoCalls.count).toBe(0);
	});
});

describe("buildSystemPrompt per-datasource tools", () => {
	it("lists generated datasource tools with their connection instead of as caller tools", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["search_datasource_documents", "search_datasource_kakao_work", "my_custom_tool"],
			manifests: [],
			jikjiIndexingEnabled: false,
			modelId: "test-model",
		});

		expect(prompt).toContain("**search_datasource_kakao_work**: search only the kakao-work datasource connection");
		expect(prompt).toContain("**my_custom_tool**: caller-provided tool");
		expect(prompt).not.toContain("**search_datasource_kakao_work**: caller-provided tool");
	});
});
