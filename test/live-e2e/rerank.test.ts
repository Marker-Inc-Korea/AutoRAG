/**
 * Live E2E: real OpenRouter rerank proof.
 *
 * Gated on AUTORAG_RERANK_LIVE=1 so the offline suite — and CI, which has no
 * OPENROUTER_API_KEY — never touches the network. Run explicitly:
 *
 *   AUTORAG_RERANK_LIVE=1 OPENROUTER_API_KEY=sk-or-... \
 *     bun run vitest run test/live-e2e/rerank.test.ts
 *
 * Covers the two production seams against the real API: the OpenRouterReranker
 * itself and the RetrievalEngine post-merge rerank stage.
 */
import { describe, expect, it } from "vitest";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import { DEFAULT_RERANK_MODEL, OpenRouterReranker } from "../../src/retrieval/rerank.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalResult } from "../../src/retrieval/types.ts";

const LIVE = process.env.AUTORAG_RERANK_LIVE === "1";

const QUERY = "How do interest rates affect bond prices?";

function doc(id: string, content: string): RetrievalResult {
	return { id, content, source: `/${id}.md`, score: 0.5, metadata: {} };
}

const DOCS: readonly RetrievalResult[] = [
	doc("cats", "Cats are small domesticated felines that purr when content."),
	doc("bread", "Sourdough bread rises through wild yeast fermentation and long proofing."),
	doc("bonds", "A bond's price falls when market interest rates rise, and rises when rates fall."),
];

const stubMethod = (results: readonly RetrievalResult[]): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name: "posix",
		type: "posix",
		description: "stub posix",
		status: "active",
		capabilities: [],
	}),
	retrieve: async () => [...results],
});

describe.skipIf(!LIVE)("OpenRouter rerank live e2e (real API)", () => {
	it("requires a real credential", () => {
		expect(process.env.OPENROUTER_API_KEY, "OPENROUTER_API_KEY must be set for the live rerank proof").toBeTruthy();
	});

	it("reorders real documents by relevance through the reranker seam", async () => {
		const reranker = new OpenRouterReranker();
		const descriptor = reranker.describe();
		expect(descriptor.available).toBe(true);
		expect(descriptor.model).toBe(DEFAULT_RERANK_MODEL);

		const out = await reranker.rerank(QUERY, DOCS);

		// Identity preserved, nothing dropped when topN is unset.
		expect(out).toHaveLength(DOCS.length);
		expect(new Set(out.map((entry) => entry.id))).toEqual(new Set(DOCS.map((entry) => entry.id)));
		// The rate-sensitive passage wins.
		expect(out[0]?.id).toBe("bonds");
		expect(out[0]?.source).toBe("/bonds.md");
		// Provider metadata + non-increasing real scores.
		expect(out[0]?.metadata.rerankProvider).toBe("openrouter");
		expect(typeof out[0]?.metadata.rerankModel).toBe("string");
		expect(out[0]?.metadata.rerankModel).not.toBe("");
		expect(typeof out[0]?.metadata.rerankScore).toBe("number");
		const scores = out.map((entry) => entry.score);
		expect([...scores].sort((a, b) => b - a)).toEqual(scores);
		expect(scores[0] as number).toBeGreaterThan(scores[scores.length - 1] as number);
	}, 30_000);

	it("honors topN against the real API", async () => {
		const out = await new OpenRouterReranker().rerank(QUERY, DOCS, { topN: 1 });
		expect(out).toHaveLength(1);
		expect(out[0]?.id).toBe("bonds");
	}, 30_000);

	it("reranks the engine's merged evidence against the real API", async () => {
		const engine = new RetrievalEngine({ reranker: new OpenRouterReranker() });
		engine.register(stubMethod(DOCS));
		const { results, diagnostics } = await engine.retrieve(QUERY, { topK: 10 });
		expect(diagnostics.some((entry) => entry.code === "rerank-failed")).toBe(false);
		expect(new Set(results.map((entry) => entry.id))).toEqual(new Set(DOCS.map((entry) => entry.id)));
		expect(results[0]?.id).toBe("bonds");
	}, 30_000);
});
