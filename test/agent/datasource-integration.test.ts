import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createLoadDatasourceSkillTool } from "../../src/agent/datasource-skill.ts";
import type {
	DatasourceIndexResult,
	DatasourceSkill,
	PollingMetadata,
	SourceDescription,
} from "../../src/datasource/types.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../src/retrieval/types.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-agent-datasource-test-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

class StaticMethod implements RetrievalMethod {
	private readonly name: string;
	private readonly rows: readonly RetrievalResult[];

	constructor(name: string, rows: readonly RetrievalResult[]) {
		this.name = name;
		this.rows = rows;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: this.name,
			type: "bm25",
			description: "KakaoTalk test datasource method",
			status: "active",
			capabilities: ["keyword"],
			datasourceId: "kakao",
			tags: ["kakao", "chat"],
		};
	}

	async retrieve(_query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return this.rows.slice(0, options.topK ?? this.rows.length);
	}
}

function makeSkill(rows: readonly RetrievalResult[]): DatasourceSkill {
	const method = new StaticMethod("kakao.keyword", rows);
	return {
		describe() {
			return {
				name: "kakao",
				type: "chat",
				description: "KakaoTalk chats exported through lazykatok",
				capabilities: ["keyword", "polling"],
				tags: ["kakao", "chat"],
				status: "active",
				datasourceId: "kakao",
				instanceId: "acct-1",
				instances: ["acct-1", "acct-2"],
			};
		},
		polling(): PollingMetadata {
			return { mode: "poll", intervalMs: 60_000 };
		},
		skillManifest() {
			return {
				name: "datasource-kakao",
				description: "Search indexed KakaoTalk chats.",
				content: "# KakaoTalk\nSearch with search_datasource_kakao; scope /kakao/acct-1.",
			};
		},
		async index(): Promise<DatasourceIndexResult> {
			return {
				ok: true,
				instanceId: "acct-1",
				skill: "kakao",
				chunkCount: rows.length,
				indexedAt: 1,
				diagnostics: [],
			};
		},
		retrievalMethods() {
			return [method];
		},
		describeSources(): readonly SourceDescription[] {
			return [
				{
					source: "/kakao/acct-1",
					datasourceId: "kakao",
					skill: "kakao",
					instanceId: "acct-1",
					contentType: "chat",
					metadata: { description: "KakaoTalk chat history for acct-1" },
				},
				{
					source: "/kakao/acct-2",
					datasourceId: "kakao",
					skill: "kakao",
					instanceId: "acct-2",
					contentType: "chat",
					metadata: { description: "KakaoTalk chat history for acct-2" },
				},
			];
		},
	};
}

function result(id: string, source: string): RetrievalResult {
	return { id, source, content: `message ${id}`, score: 1, metadata: {} };
}

function makeScopedSkill(rows: readonly RetrievalResult[]): DatasourceSkill {
	const method: RetrievalMethod = {
		describe: () => ({
			name: "slack.keyword",
			type: "bm25",
			description: "Slack test datasource method",
			status: "active",
			capabilities: ["keyword", "scoped"],
			datasourceId: "slack",
			tags: ["slack"],
		}),
		retrieve: async (_query, options) =>
			rows.filter((row) => options.scope === undefined || row.source.startsWith(options.scope.replace("/**", ""))),
	};
	return {
		describe: () => ({
			name: "slack",
			type: "chat",
			description: "Slack chats",
			capabilities: ["keyword", "polling", "scoped"],
			tags: ["slack"],
			status: "active",
			datasourceId: "slack",
			instanceId: "allowed",
		}),
		polling: () => ({ mode: "poll", intervalMs: 60_000 }),
		skillManifest: () => ({
			name: "datasource-slack",
			description: "Search Slack chats.",
			content: "Search with search_datasource_slack.",
		}),
		index: async () => ({
			ok: true,
			instanceId: "allowed",
			skill: "slack",
			chunkCount: rows.length,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [method],
		describeSources: () => [],
	};
}

describe("AutoRAGAgent datasource integration", () => {
	it("passes datasource results of a configured skill through to merge", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
			datasourceSkills: [
				makeSkill([result("a", "/kakao/personal/chunks/a"), result("b", "/kakao/personal/chunks/b")]),
			],
		});

		const { results } = await agent.searchSingleDatasourceDocuments("kakao", "message");

		expect(results.map((r) => r.source)).toEqual(["/kakao/personal/chunks/a", "/kakao/personal/chunks/b"]);
	});

	it("narrows configured datasource results to the per-query scope", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: { autoInstall: false },
			jikji: false,
			everything: false,
			fsearch: false,
			datasourceSkills: [
				makeScopedSkill([
					result("allowed", "/slack/allowed/channel/message"),
					result("secret", "/slack/secret/channel/message"),
				]),
			],
		});

		const { results } = await agent.retrieveWithDiagnostics("message", {
			scope: "/slack/allowed/**",
		});

		expect(results.map((row) => row.source)).toEqual(["/slack/allowed/channel/message"]);
	});

	it("announces configured datasource skills in the system prompt (progressive disclosure) without raw paths", () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
			datasourceSkills: [makeSkill([])],
		});

		const prompt = agent.getSystemPrompt();

		expect(prompt).toContain("<available_skills>");
		expect(prompt).toContain("datasource-kakao");
		expect(prompt).toContain("Search indexed KakaoTalk chats.");
		expect(prompt).toContain("load_datasource_skill");
		expect(prompt).toContain("search_datasource_kakao");
		expect(prompt).not.toContain("search_datasource_documents");
		// Full skill content (with example scopes) is loaded on demand, not in the prompt.
		expect(prompt).not.toContain("/kakao/acct-1");
		expect(prompt).not.toContain("/Users/");
	});

	it("indexes datasource skills during refresh and surfaces path-opaque diagnostics", async () => {
		const skill = makeSkill([]);
		const failingSkill: DatasourceSkill = {
			...skill,
			describe: () => ({ ...skill.describe(), name: "kakao", instanceId: "acct-1" }),
			index: async () => ({
				ok: false,
				instanceId: "acct-1",
				skill: "kakao",
				indexedAt: 1,
				error: "failed",
				code: "datasource-index-failed",
				message: "failed at /Users/me/Library/Containers/com.kakao",
				diagnostics: [
					{
						code: "datasource-index-failed",
						severity: "error",
						message: "failed at /Users/me/Library/Containers/com.kakao",
						source: "/Users/me/Library/Containers/com.kakao",
						instanceId: "acct-1",
					},
				],
			}),
		};
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
			datasourceSkills: [failingSkill],
		});

		const refreshResult = await agent.refresh(true);
		const status = await agent.getRefreshStatus();
		const serialized = JSON.stringify(status);

		expect(status.components.datasources).toBe("degraded");
		expect(serialized).toContain("failed at /Users/me/Library/Containers/com.kakao");
		expect(serialized).not.toContain("Datasource operation failed; details suppressed");
		expect(JSON.stringify(refreshResult)).toContain("failed at /Users/me/Library/Containers/com.kakao");
	});

	it("dynamically loads a configured datasource skill's full instructions via tool calling", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
			datasourceSkills: [makeSkill([])],
		});
		const tool = createLoadDatasourceSkillTool(agent);

		const response = await tool.execute("call-load", { name: "datasource-kakao" });

		expect(response.details).toEqual({ skill: "datasource-kakao", loaded: true });
		const text = response.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text).toContain('<skill name="datasource-kakao"');
		expect(text).toContain("search_datasource_kakao");
	});

	it("does not load datasource skills for unknown names", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
			datasourceSkills: [makeSkill([])],
		});
		const unknownResponse = await createLoadDatasourceSkillTool(agent).execute("call-unknown", {
			name: "datasource-slack",
		});
		expect(unknownResponse.details).toEqual({ skill: "datasource-slack", loaded: false });
	});

	it("searchAllDocuments returns configured datasource results and narrows by scope", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
			datasourceSkills: [
				makeScopedSkill([
					result("allowed", "/slack/allowed/channel/message"),
					result("secret", "/slack/secret/channel/message"),
				]),
			],
		});
		const all = await agent.searchAllDocuments("message", { topK: 10 });
		expect(all.results.map((r) => r.source).sort()).toEqual([
			"/slack/allowed/channel/message",
			"/slack/secret/channel/message",
		]);
		expect(JSON.stringify(all)).not.toContain(tmpDir);

		const scoped = await agent.searchAllDocuments("message", { topK: 10, scope: "/slack/allowed/**" });
		expect(scoped.results.map((r) => r.source)).toEqual(["/slack/allowed/channel/message"]);
	});
});
