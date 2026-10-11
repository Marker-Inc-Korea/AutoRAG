import { describe, expect, it } from "vitest";
import {
	createReranker,
	DEFAULT_AUTORAG_RERANK_API_KEY_ENV,
	DEFAULT_AUTORAG_RERANK_MODEL,
	DEFAULT_RERANK_MODEL,
	DEFAULT_RERANK_PROVIDER,
	OpenRouterReranker,
} from "../../src/retrieval/rerank.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";
import { startFakeServer } from "../cloud/helpers/fake-server.ts";

function result(id: string, content: string): RetrievalResult {
	return { id, content, source: `/${id}.md`, score: 0.5, metadata: {} };
}

describe("autorag rerank provider", () => {
	it("posts top_n and reorders by relevance_score through <root>/v1/rerank", async () => {
		const server = await startFakeServer((request) => {
			const top =
				typeof request.json === "object" && request.json !== null && "top_n" in request.json
					? request.json.top_n
					: undefined;
			const all = [
				{ index: 1, relevance_score: 0.9, document: { text: "beta" } },
				{ index: 0, relevance_score: 0.2, document: { text: "alpha" } },
			];
			return {
				body: {
					id: "rerank-1",
					model: DEFAULT_AUTORAG_RERANK_MODEL,
					provider: "autorag",
					results: typeof top === "number" ? all.slice(0, top) : all,
					usage: { search_units: 1, cost: 0 },
				},
			};
		});
		try {
			const reranker = createReranker({
				provider: "autorag",
				env: { AUTORAG_BASE_URL: server.url, AUTORAG_API_KEY: "dz_test" },
			});
			expect(reranker).toBeDefined();
			const out = await reranker?.rerank("query text", [result("a", "alpha"), result("b", "beta")], { topN: 1 });

			const request = server.requests[0];
			expect(request?.url).toBe("/v1/rerank");
			expect(request?.headers.authorization).toBe("Bearer dz_test");
			expect(request?.json).toEqual({
				model: DEFAULT_AUTORAG_RERANK_MODEL,
				query: "query text",
				documents: ["alpha", "beta"],
				top_n: 1,
			});
			expect(out?.map((entry) => entry.id)).toEqual(["b"]);
			expect(out?.[0]?.metadata.rerankProvider).toBe("autorag");
			expect(out?.[0]?.score).toBe(0.9);

			// Without topN the whole ranked set comes back reordered.
			const full = await reranker?.rerank("query text", [result("a", "alpha"), result("b", "beta")]);
			expect(full?.map((entry) => entry.id)).toEqual(["b", "a"]);
			expect(server.requests[1]?.json).toEqual({
				model: DEFAULT_AUTORAG_RERANK_MODEL,
				query: "query text",
				documents: ["alpha", "beta"],
			});
		} finally {
			await server.close();
		}
	});

	it("reports provider autorag and the autorag model in describe()", () => {
		const reranker = createReranker({
			provider: "autorag",
			env: { AUTORAG_BASE_URL: "http://127.0.0.1:1", AUTORAG_API_KEY: "dz_test" },
		});
		expect(reranker?.describe()).toMatchObject({
			name: "autorag",
			provider: "autorag",
			model: DEFAULT_AUTORAG_RERANK_MODEL,
			available: true,
		});
	});

	it("reports an unavailable reason naming the account key env when the key is missing", async () => {
		const reranker = createReranker({ provider: "autorag", env: {} });
		expect(reranker?.describe().available).toBe(false);
		expect(reranker?.describe().reason).toContain(DEFAULT_AUTORAG_RERANK_API_KEY_ENV);
		await expect(reranker?.rerank("q", [result("a", "alpha")])).rejects.toThrow(/unavailable|API key/);
	});

	it("lets an explicit model override the autorag default", () => {
		const reranker = createReranker({
			provider: "autorag",
			model: "custom-rerank",
			env: { AUTORAG_API_KEY: "dz_test" },
		});
		expect(reranker?.describe().model).toBe("custom-rerank");
	});

	it("still defaults the openrouter provider to voyageai/rerank-3-lite", () => {
		const reranker = createReranker({
			client: {
				async rerank() {
					return { results: [] };
				},
			},
		});
		expect(reranker?.describe().provider).toBe(DEFAULT_RERANK_PROVIDER);
		expect(reranker?.describe().model).toBe(DEFAULT_RERANK_MODEL);
	});
});

describe("OpenRouterReranker provider-dependent defaults", () => {
	it("uses the autorag key env and <root>/v1 base directly from the constructor", async () => {
		const server = await startFakeServer(() => ({
			body: {
				model: DEFAULT_AUTORAG_RERANK_MODEL,
				results: [{ index: 0, relevance_score: 1, document: { text: "alpha" } }],
			},
		}));
		try {
			const reranker = new OpenRouterReranker({
				provider: "autorag",
				env: { AUTORAG_BASE_URL: server.url, AUTORAG_API_KEY: "dz_test" },
			});
			expect(reranker.describe()).toMatchObject({
				provider: "autorag",
				model: DEFAULT_AUTORAG_RERANK_MODEL,
				available: true,
			});
			const out = await reranker.rerank("q", [result("a", "alpha")]);
			expect(server.requests[0]?.url).toBe("/v1/rerank");
			expect(server.requests[0]?.headers.authorization).toBe("Bearer dz_test");
			expect(out.map((entry) => entry.id)).toEqual(["a"]);
		} finally {
			await server.close();
		}
	});

	it("does not double /v1 when AUTORAG_BASE_URL already carries it", async () => {
		const server = await startFakeServer(() => ({
			body: {
				model: DEFAULT_AUTORAG_RERANK_MODEL,
				results: [{ index: 0, relevance_score: 1, document: { text: "alpha" } }],
			},
		}));
		try {
			const reranker = new OpenRouterReranker({
				provider: "autorag",
				env: { AUTORAG_BASE_URL: `${server.url}/v1`, AUTORAG_API_KEY: "dz_test" },
			});
			await reranker.rerank("q", [result("a", "alpha")]);
			expect(server.requests[0]?.url).toBe("/v1/rerank");
		} finally {
			await server.close();
		}
	});

	it("reports the autorag key env when the key is missing on the direct constructor", () => {
		expect(new OpenRouterReranker({ provider: "autorag", env: {} }).describe().reason).toContain(
			DEFAULT_AUTORAG_RERANK_API_KEY_ENV,
		);
	});
});
