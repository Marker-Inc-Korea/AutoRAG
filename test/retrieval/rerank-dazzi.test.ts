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
	RERANK_MAX_DOCUMENT_CHARS,
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

/** The `documents` array of a posted rerank body, narrowed at runtime. */
function postedDocuments(body: unknown): string[] {
	if (typeof body !== "object" || body === null || !("documents" in body) || !Array.isArray(body.documents)) {
		throw new Error("request body has no documents array");
	}
	return body.documents.map((entry: unknown) => String(entry));
}

describe("blank documents", () => {
	/** Echo every posted document back as a ranked result, last document best. */
	function rankingServer() {
		return startFakeServer((request) => {
			const documents = postedDocuments(request.json);
			return {
				body: {
					model: DEFAULT_DAZZI_RERANK_MODEL,
					results: documents.map((_, index) => ({ index, relevance_score: (index + 1) / 10 })).reverse(),
				},
			};
		});
	}

	it("never sends empty or whitespace-only documents and maps scores back to the original results", async () => {
		signIn();
		const server = await rankingServer();
		try {
			const reranker = new OpenRouterReranker({
				provider: "dazzi",
				agentDir,
				env: { AUTORAG_BASE_URL: server.url },
			});
			const out = await reranker.rerank("q", [
				result("a", "alpha"),
				result("blank", ""),
				result("b", "beta"),
				result("ws", " \n\t "),
				result("c", "gamma"),
			]);

			expect(postedDocuments(server.requests[0]?.json)).toEqual(["alpha", "beta", "gamma"]);
			// Server ranked the last sent document (gamma) best; ids must follow the original results.
			expect(out.map((entry) => entry.id)).toEqual(["c", "b", "a", "blank", "ws"]);
			expect(out[0]?.score).toBeCloseTo(0.3);
			expect(out[0]?.metadata.rerankProvider).toBe("dazzi");
			// Unscored blanks keep their original score and gain no rerank metadata.
			expect(out[3]?.metadata.rerankScore).toBeUndefined();
		} finally {
			await server.close();
		}
	});

	it("drops blank documents when topN is set", async () => {
		signIn();
		const server = await rankingServer();
		try {
			const reranker = new OpenRouterReranker({
				provider: "dazzi",
				agentDir,
				env: { AUTORAG_BASE_URL: server.url },
			});
			const out = await reranker.rerank("q", [result("blank", ""), result("a", "alpha"), result("b", "beta")], {
				topN: 5,
			});
			expect(out.map((entry) => entry.id)).toEqual(["b", "a"]);
		} finally {
			await server.close();
		}
	});

	it("makes no request and keeps the input order when every document is blank", async () => {
		signIn();
		const server = await rankingServer();
		try {
			const reranker = new OpenRouterReranker({
				provider: "dazzi",
				agentDir,
				env: { AUTORAG_BASE_URL: server.url },
			});
			const input = [result("x", ""), result("y", "   ")];
			expect((await reranker.rerank("q", input)).map((entry) => entry.id)).toEqual(["x", "y"]);
			expect((await reranker.rerank("q", input, { topN: 1 })).map((entry) => entry.id)).toEqual(["x"]);
			expect(server.requests).toHaveLength(0);
		} finally {
			await server.close();
		}
	});

	it("applies the same filtering for the openrouter provider", async () => {
		const sent: string[][] = [];
		const reranker = new OpenRouterReranker({
			client: {
				async rerank({ requestBody }) {
					sent.push(requestBody.documents);
					return { results: [{ index: 0, relevanceScore: 0.5 }] };
				},
			},
		});
		const out = await reranker.rerank("q", [result("blank", ""), result("a", "alpha")], { topN: 3 });
		expect(sent).toEqual([["alpha"]]);
		expect(out.map((entry) => entry.id)).toEqual(["a"]);
	});

	it("truncates documents to the provider's character limit without altering the returned results", async () => {
		signIn();
		const server = await rankingServer();
		try {
			const reranker = new OpenRouterReranker({
				provider: "dazzi",
				agentDir,
				env: { AUTORAG_BASE_URL: server.url },
			});
			const long = "가".repeat(RERANK_MAX_DOCUMENT_CHARS + 5_000);
			const exact = "b".repeat(RERANK_MAX_DOCUMENT_CHARS);
			const out = await reranker.rerank("q", [
				result("long", long),
				result("exact", exact),
				result("short", "short"),
			]);

			const sent = postedDocuments(server.requests[0]?.json);
			expect(sent.map((entry) => entry.length)).toEqual([RERANK_MAX_DOCUMENT_CHARS, RERANK_MAX_DOCUMENT_CHARS, 5]);
			expect(sent[0]).toBe(long.slice(0, RERANK_MAX_DOCUMENT_CHARS));
			expect(sent[1]).toBe(exact);
			// Evidence handed back to the caller is never truncated.
			expect(out.find((entry) => entry.id === "long")?.content).toBe(long);
		} finally {
			await server.close();
		}
	});

	it("does not split a surrogate pair at the truncation boundary", async () => {
		signIn();
		const server = await rankingServer();
		try {
			const reranker = new OpenRouterReranker({
				provider: "dazzi",
				agentDir,
				env: { AUTORAG_BASE_URL: server.url },
			});
			// The emoji's high surrogate lands on the last allowed UTF-16 unit.
			const content = `${"a".repeat(RERANK_MAX_DOCUMENT_CHARS - 1)}😀tail`;
			await reranker.rerank("q", [result("e", content)]);
			const [sent] = postedDocuments(server.requests[0]?.json);
			expect(sent).toBe("a".repeat(RERANK_MAX_DOCUMENT_CHARS - 1));
			// 0x61 is "a": the dangling high surrogate (0xd83d) was not kept.
			expect(sent?.charCodeAt((sent?.length ?? 1) - 1)).toBe(0x61);
		} finally {
			await server.close();
		}
	});
});
