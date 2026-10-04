import { describe, expect, it } from "vitest";
import {
	createReranker,
	DEFAULT_RERANK_MODEL,
	OpenRouterReranker,
	type RerankClient,
} from "../../src/retrieval/rerank.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

function result(id: string, content: string): RetrievalResult {
	return { id, content, source: `/${id}.md`, score: 0.5, metadata: {} };
}

function clientReturning(entries: ReadonlyArray<{ index: number; relevanceScore: number }>): RerankClient {
	return {
		async rerank() {
			return { model: "voyageai/rerank-3-lite", provider: "OpenRouter", results: [...entries] };
		},
	};
}

describe("OpenRouterReranker", () => {
	it("reorders by relevance score and preserves source/id identity", async () => {
		const reranker = new OpenRouterReranker({
			client: clientReturning([
				{ index: 1, relevanceScore: 0.9 },
				{ index: 0, relevanceScore: 0.2 },
			]),
		});
		const out = await reranker.rerank("q", [result("a", "alpha"), result("b", "beta")]);
		expect(out.map((entry) => entry.id)).toEqual(["b", "a"]);
		expect(out.map((entry) => entry.source)).toEqual(["/b.md", "/a.md"]);
		expect(out[0]?.score).toBe(0.9);
		expect(out[0]?.metadata.rerankModel).toBe("voyageai/rerank-3-lite");
		expect(out[0]?.metadata.rerankScore).toBe(0.9);
	});

	it("sends the configured model and document contents", async () => {
		let seen: unknown;
		const reranker = new OpenRouterReranker({
			model: "voyageai/rerank-3-lite",
			client: {
				async rerank(request) {
					seen = request.requestBody;
					return { results: [{ index: 0, relevanceScore: 1 }] };
				},
			},
		});
		await reranker.rerank("query text", [result("a", "alpha"), result("b", "beta")]);
		expect(seen).toEqual({ model: "voyageai/rerank-3-lite", query: "query text", documents: ["alpha", "beta"] });
	});

	it("honors topN by dropping the tail", async () => {
		const reranker = new OpenRouterReranker({ client: clientReturning([{ index: 1, relevanceScore: 0.9 }]) });
		const out = await reranker.rerank("q", [result("a", "alpha"), result("b", "beta")], { topN: 1 });
		expect(out.map((entry) => entry.id)).toEqual(["b"]);
	});

	it("keeps provider-unscored results when topN is unset, so evidence is not dropped", async () => {
		const reranker = new OpenRouterReranker({ client: clientReturning([{ index: 1, relevanceScore: 0.9 }]) });
		const out = await reranker.rerank("q", [result("a", "alpha"), result("b", "beta")]);
		expect(out.map((entry) => entry.id)).toEqual(["b", "a"]);
	});

	it("reports unavailable without an API key and throws instead of silently ranking", async () => {
		const reranker = new OpenRouterReranker({ env: {} });
		expect(reranker.describe().available).toBe(false);
		expect(reranker.describe().reason).toContain("OPENROUTER_API_KEY");
		await expect(reranker.rerank("q", [result("a", "alpha")])).rejects.toThrow(/unavailable|API key/);
	});

	it("defaults the model to voyageai/rerank-3-lite", () => {
		expect(new OpenRouterReranker({ client: clientReturning([]) }).describe().model).toBe(DEFAULT_RERANK_MODEL);
	});
});

describe("createReranker", () => {
	it("returns undefined for absent config, explicit false, and unknown providers", () => {
		expect(createReranker(undefined)).toBeUndefined();
		expect(createReranker(false)).toBeUndefined();
		expect(createReranker({ provider: "not-a-provider", client: clientReturning([]) })).toBeUndefined();
	});

	it("builds an OpenRouter reranker for the default provider", () => {
		expect(createReranker({ client: clientReturning([]) })).toBeInstanceOf(OpenRouterReranker);
	});
});
