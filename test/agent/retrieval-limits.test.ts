import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions, type AutoRAGRetrievalLimits } from "../../src/agent/agent.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";
import type { RetrievalMethod, RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

/** Bench-only view of the agent's private baseline builder, registry, and MinSync method. */
type AgentInternals = {
	minSyncMethod: unknown;
	jikjiClient: unknown;
	findJikji: (query: string, opts?: { readonly topK?: number }) => Promise<unknown>;
	getMethodRegistry: () => { register(method: RetrievalMethod): void };
	getRetrievalEngine: () => {
		retrieve(query: string, options?: RetrievalOptions): Promise<{ results: RetrievalResult[] }>;
	};
	retrieveWithDiagnostics: (query: string, options?: RetrievalOptions) => Promise<{ results: RetrievalResult[] }>;
	searchSingleDatasourceDocuments: (
		datasourceId: string,
		query: string,
		options?: { readonly topK?: number; readonly scope?: string },
	) => Promise<{ results: RetrievalResult[] }>;
	singleDatasourceToolSpecs: () => readonly { readonly instanceScopes: readonly string[] }[];
	prefetchInitialRetrievalContext: (queries: readonly string[], options: RetrievalOptions) => Promise<string>;
};

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-limits-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function rows(count: number): RetrievalResult[] {
	return Array.from({ length: count }, (_, index) => ({
		id: `row-${index}`,
		source: `/docs/doc-${index}.md`,
		content: `candidate ${index}`,
		score: 1 - index / 10,
		metadata: {},
	}));
}

function agentWith(limits?: AutoRAGRetrievalLimits, extra: Partial<AutoRAGAgentOptions> = {}): AgentInternals {
	const agent = new AutoRAGAgent({
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: false,
		jikji: false,
		...(limits === undefined ? {} : { limits }),
		...extra,
	});
	return agent as unknown as AgentInternals;
}

function injectMinSync(internals: AgentInternals, count: number, seenTopK: number[] = []): void {
	internals.minSyncMethod = {
		isReady: () => true,
		isBinaryMissing: () => false,
		retrieve: async (_query: string, options: RetrievalOptions) => {
			seenTopK.push(options.topK ?? 0);
			return rows(count);
		},
	};
}

function fakeMethod(name: string, count: number, datasourceId?: string): RetrievalMethod {
	return {
		describe: () => ({
			name,
			type: "posix",
			description: `${name} test method`,
			status: "active",
			capabilities: [],
			...(datasourceId === undefined ? {} : { datasourceId, tags: ["t"] }),
		}),
		retrieve: async () => rows(count),
	};
}

function fakeSkill(datasourceId: string, methodCount: number, sourceCount: number): DatasourceSkill {
	return {
		describe: () => ({
			name: datasourceId,
			type: "test",
			description: `${datasourceId} test connection`,
			capabilities: ["keyword"],
			tags: ["t"],
			status: "active",
			datasourceId,
		}),
		polling: () => ({ mode: "none" }),
		index: async () => ({
			ok: true,
			instanceId: datasourceId,
			skill: datasourceId,
			chunkCount: methodCount,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [fakeMethod(`${datasourceId}.keyword`, methodCount, datasourceId)],
		describeSources: () =>
			Array.from({ length: sourceCount }, (_, index) => ({
				source: `/${datasourceId}/inst-${index}`,
				datasourceId,
				skill: datasourceId,
				instanceId: `inst-${index}`,
				contentType: "test",
				metadata: {},
			})),
		skillManifest: () => ({
			name: datasourceId,
			description: `${datasourceId} test connection`,
			content: `# ${datasourceId}`,
		}),
	};
}

describe("AutoRAGAgent retrieval limits", () => {
	it("rejects a non-positive or non-integer limit", () => {
		expect(() => agentWith({ mergedEvidenceCeiling: 0 })).toThrow(/positive integer/u);
		expect(() => agentWith({ singleDatasourceTopK: 1.5 })).toThrow(/positive integer/u);
		expect(() => agentWith({ prefetch: { sectionLimit: -3 } })).toThrow(/positive integer/u);
	});

	it("truncates a baseline section to the configured limit", async () => {
		const internals = agentWith({ prefetch: { sectionLimit: 2 } });
		injectMinSync(internals, 5);

		const context = await internals.prefetchInitialRetrievalContext(["candidates"], {});

		expect((context.match(/^\[\d+\]/gmu) ?? []).length).toBe(2);
		expect(context).toContain("candidate 1");
		expect(context).not.toContain("candidate 2");
	});

	it("passes the configured MinSync topK to the prefetch retrieve call", async () => {
		const internals = agentWith({ prefetch: { minSyncTopK: 7 } });
		const seenTopK: number[] = [];
		injectMinSync(internals, 3, seenTopK);

		await internals.prefetchInitialRetrievalContext(["candidates"], {});

		expect(seenTopK).toEqual([7]);
	});

	it("keeps the shipped defaults when no limits are configured", async () => {
		const internals = agentWith();
		const seenTopK: number[] = [];
		injectMinSync(internals, 5, seenTopK);

		const context = await internals.prefetchInitialRetrievalContext(["candidates"], {});

		expect((context.match(/^\[\d+\]/gmu) ?? []).length).toBe(5);
		expect(seenTopK).toEqual([100]);
	});

	it("caps the search_all_documents merge at mergedEvidenceCeiling", async () => {
		const internals = agentWith({ mergedEvidenceCeiling: 2 });
		internals.getMethodRegistry().register(fakeMethod("plain", 5));

		const { results } = await internals.retrieveWithDiagnostics("query");

		expect(results).toHaveLength(2);
	});

	it("caps the single-datasource merge at singleDatasourceTopK", async () => {
		const internals = agentWith(
			{ singleDatasourceTopK: 2 },
			{ datasourceSkills: [fakeSkill("ds1", 5, 1)], datasourceAccess: { allowedTags: ["t"] } },
		);

		const { results } = await internals.searchSingleDatasourceDocuments("ds1", "query");

		expect(results).toHaveLength(2);
	});

	it("passes minSyncTopK and minSyncScopedQueryTopK to the MinSync methods", () => {
		const internals = agentWith(
			{ minSyncTopK: 7, minSyncScopedQueryTopK: 200 },
			{ minSync: { binaryPath: join(root, "missing-minsync"), autoInstall: false } },
		);

		const vector = internals.minSyncMethod as { readonly defaultTopK: number; readonly scopedTopK: number };
		expect(vector.defaultTopK).toBe(7);
		expect(vector.scopedTopK).toBe(200);
	});

	it("bounds the datasource tool description instance scopes", () => {
		const internals = agentWith(
			{ toolDescriptionInstanceScopes: 2 },
			{ datasourceSkills: [fakeSkill("ds1", 1, 5)], datasourceAccess: { allowedTags: ["t"] } },
		);

		const specs = internals.singleDatasourceToolSpecs();

		expect(specs).toHaveLength(1);
		expect(specs[0]?.instanceScopes).toHaveLength(2);
	});

	it("forwards prefetch.jikjiTopK and truncates prefetch.jikjiPathLimit", async () => {
		const internals = agentWith({ prefetch: { jikjiTopK: 12, jikjiPathLimit: 2 } });
		const seenTopK: number[] = [];
		internals.jikjiClient = {};
		internals.findJikji = async (_query, opts) => {
			seenTopK.push(opts?.topK ?? 0);
			return { answerPack: { answerPaths: ["/p/1", "/p/2", "/p/3", "/p/4"] } };
		};

		const context = await internals.prefetchInitialRetrievalContext(["query"], {});

		expect(seenTopK).toEqual([12]);
		expect((context.match(/^\[\d+\]/gmu) ?? []).length).toBe(2);
		expect(context).toContain("/p/2");
		expect(context).not.toContain("/p/3");
	});

	it("passes mergedEvidenceCeiling to the standalone retrieval engine", async () => {
		const internals = agentWith({ mergedEvidenceCeiling: 3 });
		internals.getMethodRegistry().register(fakeMethod("plain", 5));

		const { results } = await internals.getRetrievalEngine().retrieve("query");

		expect(results).toHaveLength(3);
	});
});
