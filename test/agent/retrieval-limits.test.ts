import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGRetrievalLimits } from "../../src/agent/agent.ts";
import type { RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

/** Bench-only view of the agent's private baseline builder and MinSync method. */
type PrefetchInternals = {
	minSyncMethod: unknown;
	prefetchInitialRetrievalContext: (query: string, options: Record<string, unknown>) => Promise<string>;
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

function agentWith(limits?: AutoRAGRetrievalLimits): PrefetchInternals {
	const agent = new AutoRAGAgent({
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: false,
		jikji: false,
		...(limits === undefined ? {} : { limits }),
	});
	const internals = agent as unknown as PrefetchInternals;
	return internals;
}

function injectMinSync(internals: PrefetchInternals, count: number, seenTopK: number[] = []): void {
	internals.minSyncMethod = {
		isReady: () => true,
		isBinaryMissing: () => false,
		retrieve: async (_query: string, options: RetrievalOptions) => {
			seenTopK.push(options.topK ?? 0);
			return rows(count);
		},
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

		const context = await internals.prefetchInitialRetrievalContext("candidates", {});

		expect((context.match(/^\[\d+\]/gmu) ?? []).length).toBe(2);
		expect(context).toContain("candidate 1");
		expect(context).not.toContain("candidate 2");
	});

	it("passes the configured MinSync topK to the prefetch retrieve call", async () => {
		const internals = agentWith({ prefetch: { minSyncTopK: 7 } });
		const seenTopK: number[] = [];
		injectMinSync(internals, 3, seenTopK);

		await internals.prefetchInitialRetrievalContext("candidates", {});

		expect(seenTopK).toEqual([7]);
	});

	it("keeps the shipped defaults when no limits are configured", async () => {
		const internals = agentWith();
		const seenTopK: number[] = [];
		injectMinSync(internals, 5, seenTopK);

		const context = await internals.prefetchInitialRetrievalContext("candidates", {});

		expect((context.match(/^\[\d+\]/gmu) ?? []).length).toBe(5);
		expect(seenTopK).toEqual([100]);
	});
});
