import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createLoadDatasourceSkillTool } from "../../src/agent/datasource-skill.ts";
import { createSearchDatasourceDocumentsTool } from "../../src/agent/search-datasource-tool.ts";
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
				description: "KakaoTalk chats exported through katok",
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
				content: "# KakaoTalk\nSearch with search_datasource_documents; scope /kakao/acct-1.",
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
					metadata: { description: "authorized KakaoTalk chat history" },
				},
				{
					source: "/kakao/acct-2",
					datasourceId: "kakao",
					skill: "kakao",
					instanceId: "acct-2",
					contentType: "chat",
					metadata: { description: "unauthorized KakaoTalk chat history" },
				},
			];
		},
	};
}

function result(id: string, source: string): RetrievalResult {
	return { id, source, content: `message ${id}`, score: 1, metadata: {} };
}

describe("AutoRAGAgent datasource integration", () => {
	it("filters datasource method results before merge using trusted tags and scopes", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [
				makeSkill([
					result("a", "/kakao/acct-1/chunks/a"),
					result("b", "/kakao/acct-2/chunks/b"),
					result("c", "/kakao/acct-1#fragment"),
				]),
			],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1/**"] },
		});

		const { results } = await agent.searchDatasourceDocuments("message");

		expect(results.map((r) => r.source)).toEqual(["/kakao/acct-1/chunks/a"]);
	});

	it("keeps datasource default-deny even when tool args try to grant tags or scopes", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [makeSkill([result("a", "/kakao/acct-1/chunks/a")])],
		});
		const tool = createSearchDatasourceDocumentsTool(agent);

		const response = await tool.execute("call-1", {
			query: "message",
			topK: 10,
			scope: "/kakao/acct-1/**",
			allowedTags: ["kakao"],
			allowedScopes: ["/kakao/**"],
		} as never);

		expect(response.details.resultCount).toBe(0);
		expect(response.details.sources).toEqual([]);
	});

	it("announces authorized datasource skills in the system prompt (progressive disclosure) without raw paths", () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [makeSkill([])],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1"] },
		});

		const prompt = agent.getSystemPrompt();

		expect(prompt).toContain("<available_skills>");
		expect(prompt).toContain("datasource-kakao");
		expect(prompt).toContain("Search indexed KakaoTalk chats.");
		expect(prompt).toContain("load_datasource_skill");
		expect(prompt).toContain("search_datasource_documents");
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
			datasourceSkills: [failingSkill],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1"] },
		});

		const refreshResult = await agent.refresh(true);
		const status = await agent.getRefreshStatus();
		const serialized = JSON.stringify(status);

		expect(status.components.datasources).toBe("degraded");
		expect(serialized).toContain("failed at /Users/me/Library/Containers/com.kakao");
		expect(serialized).not.toContain("Datasource operation failed; details suppressed");
		expect(JSON.stringify(refreshResult)).toContain("failed at /Users/me/Library/Containers/com.kakao");
	});

	it("dynamically loads an authorized datasource skill's full instructions via tool calling", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [makeSkill([])],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1"] },
		});
		const tool = createLoadDatasourceSkillTool(agent);

		const response = await tool.execute("call-load", { name: "datasource-kakao" });

		expect(response.details).toEqual({ skill: "datasource-kakao", loaded: true });
		const text = response.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		expect(text).toContain('<skill name="datasource-kakao"');
		expect(text).toContain("search_datasource_documents");
	});

	it("does not load datasource skills under default-deny or for unknown names", async () => {
		const denied = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [makeSkill([])],
		});
		const deniedTool = createLoadDatasourceSkillTool(denied);
		const deniedResponse = await deniedTool.execute("call-denied", { name: "datasource-kakao" });
		expect(deniedResponse.details).toEqual({ skill: "datasource-kakao", loaded: false });

		const authorized = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [makeSkill([])],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1"] },
		});
		const unknownResponse = await createLoadDatasourceSkillTool(authorized).execute("call-unknown", {
			name: "datasource-slack",
		});
		expect(unknownResponse.details).toEqual({ skill: "datasource-slack", loaded: false });
	});
	it("searchAllDocuments preserves datasource default-deny and authorized scope filtering", async () => {
		const rows = [result("a", "/kakao/acct-1/chunks/a"), result("b", "/kakao/acct-2/chunks/b")];
		const denied = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [makeSkill(rows)],
		});
		const deniedResult = await denied.searchAllDocuments("message", { topK: 10 });
		expect(deniedResult.results.map((r) => r.source)).not.toContain("/kakao/acct-1/chunks/a");
		expect(JSON.stringify(deniedResult)).not.toContain(tmpDir);

		const authorized = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [makeSkill(rows)],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1/**"] },
		});
		const authorizedResult = await authorized.searchAllDocuments("message", { topK: 10 });
		expect(authorizedResult.results.map((r) => r.source)).toContain("/kakao/acct-1/chunks/a");
		expect(authorizedResult.results.map((r) => r.source)).not.toContain("/kakao/acct-2/chunks/b");
		expect(JSON.stringify(authorizedResult)).not.toContain(tmpDir);
	});
});
