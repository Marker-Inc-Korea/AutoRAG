import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
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

	it("announces authorized datasource descriptions in the system prompt without raw paths", () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			datasourceSkills: [makeSkill([])],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1"] },
		});

		const prompt = agent.getSystemPrompt();

		expect(prompt).toContain("search_datasource_documents");
		expect(prompt).toContain("authorized KakaoTalk chat history");
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
		expect(serialized).toContain("Datasource operation failed; details suppressed");
		expect(JSON.stringify(refreshResult)).not.toContain("Library/Containers");
		expect(JSON.stringify(refreshResult)).not.toContain("com.kakao");
		expect(JSON.stringify(refreshResult)).not.toContain("/Users/me");
		expect(serialized).not.toContain("Library/Containers");
		expect(serialized).not.toContain("com.kakao");
		expect(serialized).not.toContain("/Users/me");
	});
});
