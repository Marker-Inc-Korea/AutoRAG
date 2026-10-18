import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DAZZI_LOGIN_REQUIRED_MESSAGE } from "../../src/cloud/credential.ts";
import {
	createReranker,
	DEFAULT_DAZZI_RERANK_MODEL,
	DEFAULT_RERANK_MODEL,
	DEFAULT_RERANK_PROVIDER,
	OpenRouterReranker,
} from "../../src/retrieval/rerank.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";
import { startFakeServer } from "../cloud/helpers/fake-server.ts";

let agentDir: string;

beforeEach(() => {
	agentDir = mkdtempSync(join(tmpdir(), "autorag-rerank-dazzi-"));
});

afterEach(() => {
	rmSync(agentDir, { recursive: true, force: true });
});

/** Store a `/login` credential the way Pi does. */
function signIn(token = "dz_login"): void {
	writeFileSync(
		join(agentDir, "auth.json"),
		JSON.stringify({ dazzi: { type: "oauth", access: token, refresh: "", expires: Number.MAX_SAFE_INTEGER } }),
	);
}

function result(id: string, content: string): RetrievalResult {
	return { id, content, source: `/${id}.md`, score: 0.5, metadata: {} };
}

describe("dazzi rerank provider", () => {
	it("posts top_n and reorders by relevance_score through <root>/v1/rerank with the login token", async () => {
		signIn("dz_login");
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
					model: DEFAULT_DAZZI_RERANK_MODEL,
					provider: "dazzi",
					results: typeof top === "number" ? all.slice(0, top) : all,
					usage: { search_units: 1, cost: 0 },
				},
			};
		});
		try {
			const reranker = createReranker({
				provider: "dazzi",
				agentDir,
				env: { AUTORAG_BASE_URL: server.url },
			});
			expect(reranker).toBeDefined();
			const out = await reranker?.rerank("query text", [result("a", "alpha"), result("b", "beta")], { topN: 1 });

			const request = server.requests[0];
			expect(request?.url).toBe("/v1/rerank");
			expect(request?.headers.authorization).toBe("Bearer dz_login");
			expect(request?.json).toEqual({
				model: DEFAULT_DAZZI_RERANK_MODEL,
				query: "query text",
				documents: ["alpha", "beta"],
				top_n: 1,
			});
			expect(out?.map((entry) => entry.id)).toEqual(["b"]);
			expect(out?.[0]?.metadata.rerankProvider).toBe("dazzi");
			expect(out?.[0]?.score).toBe(0.9);

			// Without topN the whole ranked set comes back reordered.
			const full = await reranker?.rerank("query text", [result("a", "alpha"), result("b", "beta")]);
			expect(full?.map((entry) => entry.id)).toEqual(["b", "a"]);
			expect(server.requests[1]?.json).toEqual({
				model: DEFAULT_DAZZI_RERANK_MODEL,
				query: "query text",
				documents: ["alpha", "beta"],
			});
		} finally {
			await server.close();
		}
	});

	it("reports provider dazzi and the dazzi model in describe() once signed in", () => {
		signIn();
		const reranker = createReranker({
			provider: "dazzi",
			agentDir,
			env: { AUTORAG_BASE_URL: "http://127.0.0.1:1" },
		});
		expect(reranker?.describe()).toMatchObject({
			name: "dazzi",
			provider: "dazzi",
			model: DEFAULT_DAZZI_RERANK_MODEL,
			available: true,
		});
	});

	it("is unavailable with a sign-in hint when no login is stored, even if AUTORAG_API_KEY is set", async () => {
		const reranker = createReranker({ provider: "dazzi", agentDir, env: { AUTORAG_API_KEY: "dz_env" } });
		expect(reranker?.describe().available).toBe(false);
		expect(reranker?.describe().reason).toBe(DAZZI_LOGIN_REQUIRED_MESSAGE);
		await expect(reranker?.rerank("q", [result("a", "alpha")])).rejects.toThrow(/sign-in/);
	});

	it("lets an explicit model override the dazzi default", () => {
		signIn();
		const reranker = createReranker({ provider: "dazzi", model: "custom-rerank", agentDir });
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
	it("does not double /v1 when AUTORAG_BASE_URL already carries it", async () => {
		signIn();
		const server = await startFakeServer(() => ({
			body: {
				model: DEFAULT_DAZZI_RERANK_MODEL,
				results: [{ index: 0, relevance_score: 1, document: { text: "alpha" } }],
			},
		}));
		try {
			const reranker = new OpenRouterReranker({
				provider: "dazzi",
				agentDir,
				env: { AUTORAG_BASE_URL: `${server.url}/v1` },
			});
			await reranker.rerank("q", [result("a", "alpha")]);
			expect(server.requests[0]?.url).toBe("/v1/rerank");
		} finally {
			await server.close();
		}
	});

	it("reports the sign-in hint when no login is stored on the direct constructor", () => {
		expect(new OpenRouterReranker({ provider: "dazzi", agentDir, env: {} }).describe().reason).toBe(
			DAZZI_LOGIN_REQUIRED_MESSAGE,
		);
	});
});
