import {
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runMethodQueries } from "../../benchmark/miracl/run.ts";
import { materializeBenchmarkWorkspace } from "../../benchmark/miracl/workspace.ts";
import { loadMirrorIndex } from "../../src/mirror/index-store.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";

function deterministicClock(values: readonly number[]): () => number {
	let index = 0;
	return () => {
		const value = values[index];
		if (value === undefined) throw new Error("deterministic clock exhausted");
		index += 1;
		return value;
	};
}

function retrievalMethod(retrieve: RetrievalMethod["retrieve"], name = "bm25"): RetrievalMethod {
	return {
		describe: () => ({
			name,
			type: "bm25",
			description: "benchmark test retrieval",
			status: "active",
			capabilities: [],
		}),
		retrieve,
	};
}

describe("MIRACL benchmark workspace and runner", () => {
	const temporaryRoots: string[] = [];
	const makeParent = () => {
		const parent = mkdtempSync(join(tmpdir(), "autorag-miracl-run-"));
		temporaryRoots.push(parent);
		return parent;
	};

	afterEach(() => {
		for (const root of temporaryRoots.splice(0)) {
			rmSync(root, { recursive: true, force: true });
		}
		vi.restoreAllMocks();
	});

	it("materializes document ids as reversible virtual paths using the production mirror index", () => {
		const parent = makeParent();
		const root = join(parent, "workspace");

		const result = materializeBenchmarkWorkspace(root, [
			{ documentId: "123#4", title: "제목", text: "검색 가능한 본문" },
			{ documentId: "123%234", title: "다른 제목", text: "다른 본문" },
		]);

		expect(result.root).toBe(join(realpathSync(parent), "workspace"));
		expect(result.documentBySource.get("/miracl/123%234.md")).toBe("123#4");
		expect(result.documentBySource.get("/miracl/123%25234.md")).toBe("123%234");
		expect(result.mirrorFiles).toHaveLength(2);
		expect(readFileSync(result.mirrorFiles[0] as string, "utf8")).toBe("# 제목\n\n검색 가능한 본문\n");
		expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual(["/miracl/123%234.md", "/miracl/123%25234.md"]);
	});

	it("rejects and preserves a pre-existing workspace root", () => {
		const parent = makeParent();
		const root = join(parent, "workspace");
		mkdirSync(root);
		const sentinel = join(root, "keep.txt");
		writeFileSync(sentinel, "user data\n");

		expect(() => materializeBenchmarkWorkspace(root, [{ documentId: "doc", title: "제목", text: "본문" }])).toThrow(
			"must not already exist",
		);
		expect(readFileSync(sentinel, "utf8")).toBe("user data\n");
		expect(existsSync(join(root, ".autorag"))).toBe(false);
	});

	it("rejects a root whose symlinked parent resolves inside an existing .autorag", () => {
		const parent = makeParent();
		const sourceCheckout = join(parent, "source-checkout");
		const userIndex = join(sourceCheckout, ".autorag");
		const alias = join(parent, "index-alias");
		mkdirSync(userIndex, { recursive: true });
		const sentinel = join(userIndex, "keep.txt");
		writeFileSync(sentinel, "user index\n");
		symlinkSync(userIndex, alias, "dir");

		expect(() =>
			materializeBenchmarkWorkspace(join(alias, "benchmark"), [{ documentId: "doc", title: "제목", text: "본문" }]),
		).toThrow("existing .autorag");
		expect(readFileSync(sentinel, "utf8")).toBe("user index\n");
		expect(existsSync(join(userIndex, "benchmark"))).toBe(false);
	});

	it("runs real BM25 and maps chunks back to MIRACL document ids", async () => {
		const parent = makeParent();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "123#4", title: "제목", text: "검색 가능한 본문" },
			{ documentId: "other", title: "기타", text: "무관한 내용" },
		]);
		const syncedBm25 = new BM25Method({
			root: workspace.root,
			forceEngine: "typescript-fallback",
		});
		const sync = await syncedBm25.sync();
		expect(sync.readiness).toBe("degraded_fallback");

		const records = await runMethodQueries({
			method: "bm25",
			retrieval: syncedBm25,
			queries: [{ queryId: "q1", text: "검색 가능한" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
			now: deterministicClock([0, 4]),
		});

		expect(records).toEqual([
			{
				schemaVersion: 1,
				method: "bm25",
				queryId: "q1",
				latencyMs: 4,
				hits: [
					expect.objectContaining({
						documentId: "123#4",
						rank: 1,
					}),
				],
			},
		]);
	});

	it("rejects a retrieval method that is not ready before running queries", async () => {
		const retrieve = vi.fn(async () => []);
		const retrieval: RetrievalMethod = {
			describe: () => ({
				name: "bm25",
				type: "bm25",
				description: "not synchronized",
				status: "stub",
				capabilities: ["readiness:index_missing"],
			}),
			retrieve,
		};

		await expect(
			runMethodQueries({
				method: "bm25",
				retrieval,
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource: new Map(),
				topK: 5,
			}),
		).rejects.toThrow("must be ready");
		expect(retrieve).not.toHaveBeenCalled();
	});

	it("requests 100 candidates and emits stable document-level rankings", async () => {
		const calls: Array<{ query: string; topK: number | undefined }> = [];
		const results: RetrievalResult[] = [
			{ id: "a-low", source: "/a", content: "a1", score: 1, metadata: {} },
			{ id: "b", source: "/b", content: "b", score: 3, metadata: {} },
			{ id: "a-high", source: "/a", content: "a2", score: 3, metadata: {} },
			{ id: "c", source: "/c", content: "c", score: 3, metadata: {} },
		];
		const retrieval = retrievalMethod(async (query, options) => {
			calls.push({ query, topK: options.topK });
			return results;
		});

		const records = await runMethodQueries({
			method: "bm25",
			retrieval,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: new Map([
				["/a", "a"],
				["/b", "b"],
				["/c", "c"],
			]),
			topK: 2,
			now: deterministicClock([10, 12]),
		});

		expect(calls).toEqual([{ query: "query", topK: 100 }]);
		expect(records[0]?.hits).toEqual([
			{ documentId: "a", score: 3, rank: 1 },
			{ documentId: "b", score: 3, rank: 2 },
		]);
	});

	it("records opaque query failures durably and continues sequentially", async () => {
		let inFlight = 0;
		let maximumInFlight = 0;
		const retrieval = retrievalMethod(async (query) => {
			inFlight += 1;
			maximumInFlight = Math.max(maximumInFlight, inFlight);
			try {
				await Promise.resolve();
				if (query === "first") throw new Error("secret endpoint and filesystem path");
				return [{ id: "ok", source: "/ok", content: "ok", score: 1, metadata: {} }];
			} finally {
				inFlight -= 1;
			}
		});

		const records = await runMethodQueries({
			method: "bm25",
			retrieval,
			queries: [
				{ queryId: "q1", text: "first" },
				{ queryId: "q2", text: "second" },
			],
			documentBySource: new Map([["/ok", "ok"]]),
			topK: 5,
			now: deterministicClock([0, 2, 3, 7]),
		});

		expect(records).toEqual([
			{
				schemaVersion: 1,
				method: "bm25",
				queryId: "q1",
				latencyMs: 2,
				hits: [],
				errorCode: "retrieval-failed",
			},
			{
				schemaVersion: 1,
				method: "bm25",
				queryId: "q2",
				latencyMs: 4,
				hits: [{ documentId: "ok", score: 1, rank: 1 }],
			},
		]);
		expect(maximumInFlight).toBe(1);
		expect(JSON.stringify(records)).not.toContain("secret endpoint");
		expect(JSON.stringify(records)).not.toContain("filesystem path");
	});
});
