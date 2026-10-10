import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type {
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	PollingMetadata,
	SourceDescription,
} from "../../src/datasource/types.ts";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-datasource-catalog-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function result(id: string, source: string): RetrievalResult {
	return { id, source, content: `message ${id}`, score: 1, metadata: {} };
}

/** A datasource skill with an optional spied retrieval method. */
function makeCatalogSkill(options: {
	readonly datasourceId: string;
	readonly tags: readonly string[];
	readonly sources: readonly string[];
	readonly capabilities?: readonly string[];
	readonly withMethod?: boolean;
	readonly calls?: { count: number };
}): DatasourceSkill {
	const capabilities = options.capabilities ?? ["keyword"];
	const method: RetrievalMethod = {
		describe: () => ({
			name: `${options.datasourceId}.keyword`,
			type: "bm25" as const,
			description: `${options.datasourceId} keyword`,
			status: "active" as const,
			capabilities: ["keyword"],
			datasourceId: options.datasourceId,
			tags: options.tags,
		}),
		retrieve: async () => {
			if (options.calls !== undefined) options.calls.count += 1;
			return [result("a", `/${options.datasourceId}/default/chunks/a`)];
		},
	};
	return {
		describe: (): DatasourceSkillDescriptor => ({
			name: options.datasourceId,
			type: "chat",
			description: `${options.datasourceId} description`,
			capabilities,
			tags: options.tags,
			status: "active",
			datasourceId: options.datasourceId,
			instanceId: "default",
		}),
		polling: (): PollingMetadata => ({ mode: "none" }),
		skillManifest: () => ({
			name: `datasource-${options.datasourceId}`,
			description: `Search ${options.datasourceId}.`,
			content: `# ${options.datasourceId}`,
		}),
		index: async (): Promise<DatasourceIndexResult> => ({
			ok: true,
			instanceId: "default",
			skill: options.datasourceId,
			chunkCount: 1,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => (options.withMethod === false ? [] : [method]),
		describeSources: (): readonly SourceDescription[] =>
			options.sources.map((source) => ({
				source,
				datasourceId: options.datasourceId,
				skill: options.datasourceId,
				instanceId: "default",
				contentType: "chat",
				metadata: {},
			})),
	};
}

function makeAgent(skills: readonly DatasourceSkill[]): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: ["test/fixtures/sample-project"],
		workspacePath: tmpDir,
		jikji: false,
		minSync: { autoInstall: false },
		datasourceSkills: skills,
	});
}

describe("AutoRAGAgent.listDatasources", () => {
	it("lists configured descriptors, including a datasource with no retrieval methods", () => {
		const agent = makeAgent([
			makeCatalogSkill({ datasourceId: "kakao", tags: ["kakao"], sources: ["/kakao/default"] }),
			makeCatalogSkill({
				datasourceId: "empty",
				tags: ["kakao"],
				sources: ["/empty/default"],
				withMethod: false,
			}),
		]);

		const entries = agent.listDatasources();

		expect(entries.map((entry) => entry.datasourceId)).toEqual(["kakao", "empty"]);
		expect(entries[0]).toMatchObject({
			datasourceId: "kakao",
			name: "kakao",
			type: "chat",
			description: "kakao description",
			status: "active",
		});
		expect(entries[0]?.tags).toEqual(["kakao"]);
		expect(entries[0]?.sourceScopes).toEqual(["/kakao/default"]);
	});

	it("lists every source scope for a scoped datasource without permission filtering", () => {
		const agent = makeAgent([
			makeCatalogSkill({
				datasourceId: "slack",
				tags: ["slack"],
				sources: ["/slack/allowed/channel", "/slack/secret/channel"],
				capabilities: ["keyword", "scoped"],
			}),
		]);

		expect(agent.listDatasources()[0]?.sourceScopes).toEqual(["/slack/allowed/channel", "/slack/secret/channel"]);
	});

	it("collapses duplicate datasource ids to the first registration", () => {
		const agent = makeAgent([
			makeCatalogSkill({ datasourceId: "kakao", tags: ["kakao"], sources: ["/kakao/one"] }),
			makeCatalogSkill({ datasourceId: "kakao", tags: ["kakao"], sources: ["/kakao/two"], withMethod: false }),
		]);

		const entries = agent.listDatasources();
		expect(entries).toHaveLength(1);
		expect(entries[0]?.sourceScopes).toEqual(["/kakao/one"]);
	});
});

describe("AutoRAGAgent retrieval engine selection", () => {
	it("executes only the selected datasource, not the others", async () => {
		const kakaoCalls = { count: 0 };
		const slackCalls = { count: 0 };
		const agent = makeAgent([
			makeCatalogSkill({
				datasourceId: "kakao",
				tags: ["kakao"],
				sources: ["/kakao/default"],
				calls: kakaoCalls,
			}),
			makeCatalogSkill({
				datasourceId: "slack",
				tags: ["slack"],
				sources: ["/slack/default"],
				calls: slackCalls,
			}),
		]);

		const { results } = await agent.getRetrievalEngine().retrieveSelected("q", { datasourceIds: ["kakao"] });

		expect(results.map((r) => r.source)).toEqual(["/kakao/default/chunks/a"]);
		expect(kakaoCalls.count).toBe(1);
		expect(slackCalls.count).toBe(0);
	});

	it("selects a method-less datasource cleanly", async () => {
		const agent = makeAgent([
			makeCatalogSkill({ datasourceId: "empty", tags: ["kakao"], sources: ["/empty/default"], withMethod: false }),
		]);

		const { results } = await agent.getRetrievalEngine().retrieveSelected("q", { datasourceIds: ["empty"] });

		expect(results).toEqual([]);
	});
});
