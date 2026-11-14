import { once } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { MinSyncRequiredError } from "../../src/minsync/errors.ts";
import type { RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

/** The agent's private baseline builder and the two MinSync channels it queries. */
interface AgentInternals {
	minSyncMethod: unknown;
	minSyncSparseMethod: unknown;
	routingDiagnostics: { code: string; source?: string; message: string }[];
	prefetchInitialRetrievalContext: (
		query: string,
		searchQueries: readonly string[],
		options: RetrievalOptions,
	) => Promise<string>;
}

let root: string;
let server: Server | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-sparse-union-"));
});

afterEach(() => {
	server?.closeAllConnections();
	server?.close();
	server = undefined;
	rmSync(root, { recursive: true, force: true });
});

function chunk(id: string, content = `chunk ${id}`): RetrievalResult {
	return { id, source: `/docs/${id}.md`, content, score: 1, metadata: { method: "minsync" } };
}

interface Channel {
	readonly seenTopK: number[];
	readonly method: {
		isReady: () => boolean;
		isBinaryMissing: () => boolean;
		retrieve: (query: string, options: RetrievalOptions) => Promise<RetrievalResult[]>;
	};
}

function channel(results: RetrievalResult[] | Error): Channel {
	const seenTopK: number[] = [];
	return {
		seenTopK,
		method: {
			isReady: () => true,
			isBinaryMissing: () => false,
			retrieve: async (_query, options) => {
				seenTopK.push(options.topK ?? 0);
				if (results instanceof Error) throw results;
				return results;
			},
		},
	};
}

/** A rerank endpoint that records the documents it was sent and keeps their order. */
async function recordingRerankServer(): Promise<{ sent: string[][]; options: Partial<AutoRAGAgentOptions> }> {
	const sent: string[][] = [];
	server = createServer((request, response) => {
		let body = "";
		request.on("data", (part) => {
			body += part;
		});
		request.on("end", () => {
			const parsed = JSON.parse(body) as { documents: string[] };
			sent.push(parsed.documents);
			const results = parsed.documents.map((document, index) => ({
				document: { text: document },
				index,
				relevance_score: 1 - index / parsed.documents.length,
			}));
			response.writeHead(200, { "content-type": "application/json" });
			response.end(JSON.stringify({ model: "test-rerank", results }));
		});
	});
	server.listen(0, "127.0.0.1");
	await once(server, "listening");
	const address = server.address();
	const port = typeof address === "object" && address !== null ? address.port : 0;
	return {
		sent,
		options: {
			rerank: { provider: "openrouter", model: "test-rerank", apiKey: "k", baseUrl: `http://127.0.0.1:${port}/` },
		},
	};
}

function agentWith(extra: Partial<AutoRAGAgentOptions> = {}): AgentInternals {
	const agent = new AutoRAGAgent({
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		...extra,
	});
	return agent as unknown as AgentInternals;
}

describe("fast-answer baseline: SPLADE and dense candidates reach the reranker", () => {
	it("sends the union of both channels to the reranker, without the repeated chunk", async () => {
		const rerank = await recordingRerankServer();
		const internals = agentWith(rerank.options);
		internals.minSyncMethod = channel([chunk("dense-1"), chunk("both"), chunk("dense-2")]).method;
		internals.minSyncSparseMethod = channel([chunk("both"), chunk("sparse-only")]).method;

		await internals.prefetchInitialRetrievalContext("q", ["q"], {});

		expect(rerank.sent).toHaveLength(1);
		expect(rerank.sent[0]).toEqual(["chunk dense-1", "chunk both", "chunk sparse-only", "chunk dense-2"]);
	});

	it("shows the model a chunk only SPLADE found", async () => {
		const rerank = await recordingRerankServer();
		const internals = agentWith(rerank.options);
		internals.minSyncMethod = channel([chunk("dense-1")]).method;
		internals.minSyncSparseMethod = channel([chunk("sparse-only", "ERR_4711 exact error code")]).method;

		const context = await internals.prefetchInitialRetrievalContext("ERR_4711", ["ERR_4711"], {});

		expect(context).toContain("ERR_4711 exact error code");
		expect(context).toContain("/docs/sparse-only.md");
	});

	it("asks both channels for the same wide candidate count", async () => {
		const rerank = await recordingRerankServer();
		const internals = agentWith(rerank.options);
		const dense = channel([chunk("a")]);
		const sparse = channel([chunk("b")]);
		internals.minSyncMethod = dense.method;
		internals.minSyncSparseMethod = sparse.method;

		await internals.prefetchInitialRetrievalContext("q", ["q"], {});

		expect(dense.seenTopK).toEqual([100]);
		expect(sparse.seenTopK).toEqual([100]);
	});

	it("does not query SPLADE when no reranker decides the order", async () => {
		const internals = agentWith();
		const dense = channel([chunk("dense-1")]);
		const sparse = channel([chunk("sparse-only")]);
		internals.minSyncMethod = dense.method;
		internals.minSyncSparseMethod = sparse.method;

		const context = await internals.prefetchInitialRetrievalContext("q", ["q"], {});

		expect(sparse.seenTopK).toEqual([]);
		expect(dense.seenTopK).toEqual([10]);
		expect(context).not.toContain("sparse-only");
	});

	it("keeps the dense candidates and reports SPLADE when only the SPLADE channel fails", async () => {
		const rerank = await recordingRerankServer();
		const internals = agentWith(rerank.options);
		internals.minSyncMethod = channel([chunk("dense-1"), chunk("dense-2")]).method;
		internals.minSyncSparseMethod = channel(new Error("sparse index is not ready: run `minsync sync`")).method;

		const context = await internals.prefetchInitialRetrievalContext("q", ["q"], {});

		expect(rerank.sent[0]).toEqual(["chunk dense-1", "chunk dense-2"]);
		expect(context).toContain("chunk dense-1");
		const diagnostic = internals.routingDiagnostics.find((entry) => entry.code === "retrieval-method-failed");
		expect(diagnostic?.source).toBe("minsync-sparse");
		expect(diagnostic?.message).toContain("sparse index is not ready");
	});

	it("keeps the SPLADE candidates when only the dense channel fails", async () => {
		const rerank = await recordingRerankServer();
		const internals = agentWith(rerank.options);
		internals.minSyncMethod = channel(new Error("embedder unreachable")).method;
		internals.minSyncSparseMethod = channel([chunk("sparse-1")]).method;

		await internals.prefetchInitialRetrievalContext("q", ["q"], {});

		expect(rerank.sent[0]).toEqual(["chunk sparse-1"]);
		const diagnostic = internals.routingDiagnostics.find((entry) => entry.code === "retrieval-method-failed");
		expect(diagnostic?.source).toBe("minsync");
	});

	it("still fails the run when MinSync itself is missing", async () => {
		const rerank = await recordingRerankServer();
		const internals = agentWith(rerank.options);
		internals.minSyncMethod = channel([chunk("dense-1")]).method;
		internals.minSyncSparseMethod = channel(new MinSyncRequiredError("minsync binary is missing")).method;

		await expect(internals.prefetchInitialRetrievalContext("q", ["q"], {})).rejects.toBeInstanceOf(
			MinSyncRequiredError,
		);
	});

	it("has no second channel to query on a sparse-only index", () => {
		const internals = agentWith({
			minSync: { autoInstall: false, embedder: { id: "none" } },
		});

		expect(internals.minSyncSparseMethod).toBeUndefined();
	});
});
