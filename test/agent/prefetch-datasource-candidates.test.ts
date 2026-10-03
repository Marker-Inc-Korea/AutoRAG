import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { DatasourceIndexResult, DatasourceSkill, PollingMetadata } from "../../src/datasource/types.ts";
import type { RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-prefetch-datasource-"));
});

afterEach(() => {
	vi.useRealTimers();
	rmSync(tmpDir, { recursive: true, force: true });
});

function chunk(id: string, source: string, content: string): RetrievalResult {
	return { id, source, content, score: 1, metadata: {} };
}

/** A datasource connection whose single retrieval method is `retrieve`. */
function makeSkill(
	datasourceId: string,
	retrieve: (query: string, options: RetrievalOptions) => Promise<RetrievalResult[]>,
	calls: { count: number } = { count: 0 },
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
			chunkCount: 1,
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
				retrieve: async (query: string, options: RetrievalOptions) => {
					calls.count += 1;
					return retrieve(query, options);
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

type PrefetchInternals = {
	prefetchInitialRetrievalContext: (query: string, options: Record<string, unknown>) => Promise<string>;
};

function agentWith(skills: DatasourceSkill[], allowedTags: string[]): PrefetchInternals {
	const agent = new AutoRAGAgent({
		searchPaths: ["test/fixtures/sample-project"],
		workspacePath: tmpDir,
		memoryPath: join(tmpDir, "memory.json"),
		jikji: false,
		minSync: false,
		datasourceSkills: skills,
		datasourceAccess: { allowedTags },
	});
	return agent as unknown as PrefetchInternals;
}

describe("AutoRAGAgent baseline candidates from connected datasources", () => {
	it("puts connected-datasource hits into the baseline the fast answer reads", async () => {
		const agent = agentWith(
			[
				makeSkill("kakao", async () => [
					chunk("k1", "/kakao/default/chunks/k1", "Capacity confirmed at 100 seats on 9/16."),
				]),
			],
			["kakao"],
		);

		const context = await agent.prefetchInitialRetrievalContext("capacity", {});

		expect(context).toContain("Connected datasource initial candidates:");
		expect(context).toContain("/kakao/default/chunks/k1");
		expect(context).toContain("Capacity confirmed at 100 seats on 9/16.");
	});

	it("never runs or shows a datasource the trusted access context does not allow", async () => {
		const slackCalls = { count: 0 };
		const agent = agentWith(
			[
				makeSkill("kakao", async () => [chunk("k1", "/kakao/default/chunks/k1", "allowed chat")]),
				makeSkill("slack", async () => [chunk("s1", "/slack/default/chunks/s1", "private channel")], slackCalls),
			],
			["kakao"],
		);

		const context = await agent.prefetchInitialRetrievalContext("chat", {});

		expect(slackCalls.count).toBe(0);
		expect(context).toContain("allowed chat");
		expect(context).not.toContain("/slack/default");
		expect(context).not.toContain("private channel");
	});

	it("names a datasource that misses the deadline and keeps the ones that answered", async () => {
		vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
		let aborted = false;
		const agent = agentWith(
			[
				makeSkill("kakao", async () => [chunk("k1", "/kakao/default/chunks/k1", "fast chat")]),
				makeSkill(
					"mail",
					(_query, options) =>
						new Promise<RetrievalResult[]>(() => {
							options.signal?.addEventListener("abort", () => {
								aborted = true;
							});
						}),
				),
			],
			["kakao", "mail"],
		);

		const pending = agent.prefetchInitialRetrievalContext("chat", {});
		await vi.advanceTimersByTimeAsync(15_000);
		const context = await pending;

		expect(context).toContain("fast chat");
		expect(context).toContain("Connected datasources not searched within 15000ms: mail.");
		expect(aborted).toBe(true);
	});

	it("reports a failing datasource with its own error text", async () => {
		const agent = agentWith(
			[
				makeSkill("kakao", async () => {
					throw new Error("katok: archive locked by another process");
				}),
			],
			["kakao"],
		);

		const context = await agent.prefetchInitialRetrievalContext("chat", {});

		expect(context).toContain("Datasource diagnostic (kakao.keyword):");
		expect(context).toContain("katok: archive locked by another process");
	});

	it("leaves the baseline unchanged when no datasource is connected", async () => {
		const agent = agentWith([], []);

		const context = await agent.prefetchInitialRetrievalContext("anything", {});

		expect(context).not.toContain("Connected datasource");
	});
});
