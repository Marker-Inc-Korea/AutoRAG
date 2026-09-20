import { describe, expect, it, vi } from "vitest";
import { ParallelRetriever, ResultMerger } from "../../src/retrieval/merger.ts";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";

function makeResult(id: string, source: string, score: number): RetrievalResult {
	return { id, content: `content-${id}`, source, score, metadata: {} };
}

function makeMockMethod(name: string, results: RetrievalResult[]): RetrievalMethod {
	return {
		describe: () => ({ name, type: "posix" as const, description: "", status: "active" as const, capabilities: [] }),
		retrieve: vi.fn().mockResolvedValue(results),
	};
}

describe("ResultMerger", () => {
	it("merges results from two methods", () => {
		const merger = new ResultMerger();
		const results = new Map([
			["method1", [makeResult("a", "file1.ts", 0.9), makeResult("b", "file2.ts", 0.5)]],
			["method2", [makeResult("c", "file3.ts", 0.8)]],
		]);
		const merged = merger.merge(results, { topK: 10, dedup: false });
		expect(merged.length).toBe(3);
	});

	it("drops a pure duplicate evidence identity seen by two methods", () => {
		const merger = new ResultMerger();
		const results = new Map([
			["method1", [{ ...makeResult("chunk-1", "file1.ts", 0.9), id: "chunk-1" }]],
			["method2", [{ ...makeResult("chunk-1", "file1.ts", 0.5), id: "chunk-1" }]],
		]);
		const merged = merger.merge(results, { topK: 10, dedup: true });
		expect(merged).toHaveLength(1);
		expect(merged[0].id).toBe("chunk-1");
		expect(merged[0].metadata).toMatchObject({
			retrievalMethods: ["method1", "method2"],
			duplicateHitCount: 2,
		});
	});

	it("keeps every distinct chunk of one source instead of collapsing to its best", () => {
		const merger = new ResultMerger();
		const source = "opaque:document:multi-passage";
		const results = new Map([
			[
				"bm25",
				[
					makeResult("chunk-1", source, 0.95),
					makeResult("chunk-2", source, 0.94),
					makeResult("chunk-3", source, 0.93),
					makeResult("chunk-4", source, 0.92),
					makeResult("chunk-5", source, 0.91),
				],
			],
		]);

		const merged = merger.merge(results, { topK: 50, dedup: true });

		expect(merged).toHaveLength(5);
		expect(merged.map((result) => result.id).sort()).toEqual(["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]);
		for (const result of merged) expect(result.metadata.sourceChunkCount).toBe(5);
	});

	it("keeps distinct chunks from both methods that hit the same source", () => {
		const merger = new ResultMerger();
		const source = "opaque:document:shared";
		const results = new Map([
			["bm25", [makeResult("lexical-chunk", source, 0.9)]],
			["minsync", [makeResult("vector-chunk", source, 0.8)]],
		]);

		const merged = merger.merge(results, { topK: 50, dedup: true });

		expect(merged).toHaveLength(2);
		expect(merged.map((result) => result.id).sort()).toEqual(["lexical-chunk", "vector-chunk"]);
		for (const result of merged) {
			expect(result.metadata.retrievalMethods).toEqual(["bm25", "minsync"]);
		}
	});

	it("ranks a corroborated source above an equally scored one-off hit", () => {
		const merger = new ResultMerger();
		const corroborated = "opaque:document:repeat-7f9c";
		const singleton = "opaque:document:single-4a2d";
		const results = new Map([
			[
				"bm25",
				[
					makeResult("single", singleton, 0.95),
					makeResult("repeat-bm25", corroborated, 0.94),
					makeResult("bm25-floor", "opaque:floor:bm25", 0.1),
				],
			],
			[
				"minsync",
				[makeResult("repeat-vector", corroborated, 0.9), makeResult("vector-floor", "opaque:floor:vector", 0.1)],
			],
		]);

		const merged = merger.merge(results, { topK: 50, dedup: true });

		// Nothing is discarded: every distinct chunk is still present.
		expect(merged).toHaveLength(5);
		expect(merged[0].source).toBe(corroborated);
		expect(merged[0].score).toBeGreaterThan(merged[1].score);
		expect(merged[0].score).toBeLessThanOrEqual(1);
		expect(merged[0].metadata).toMatchObject({ retrievalMethods: ["bm25", "minsync"], sourceChunkCount: 2 });
	});

	it("treats distinct ids on one source as separate evidence even with dedup disabled", () => {
		const merger = new ResultMerger();
		const results = new Map([
			["method1", [makeResult("a", "opaque:same", 0.9), makeResult("b", "opaque:same", 0.8)]],
		]);

		const merged = merger.merge(results, { topK: 10, dedup: false });

		expect(merged).toHaveLength(2);
		expect(merged.map((result) => result.id)).toEqual(["a", "b"]);
	});

	it("enforces topK limit", () => {
		const merger = new ResultMerger();
		const results = new Map([
			["method1", [makeResult("a", "f1.ts", 1.0), makeResult("b", "f2.ts", 0.9), makeResult("c", "f3.ts", 0.8)]],
		]);
		const merged = merger.merge(results, { topK: 2, dedup: false });
		expect(merged.length).toBe(2);
	});

	it("returns empty array when no results", () => {
		const merger = new ResultMerger();
		const merged = merger.merge(new Map(), { topK: 10, dedup: true });
		expect(merged).toEqual([]);
	});

	it("handles single method passthrough", () => {
		const merger = new ResultMerger();
		const results = new Map([["method1", [makeResult("a", "f1.ts", 0.7)]]]);
		const merged = merger.merge(results, { topK: 10, dedup: false });
		expect(merged.length).toBe(1);
	});
});

describe("ParallelRetriever", () => {
	it("retrieves from multiple methods in parallel", async () => {
		const retriever = new ParallelRetriever();
		const method1 = makeMockMethod("m1", [makeResult("a", "f1.ts", 1.0)]);
		const method2 = makeMockMethod("m2", [makeResult("b", "f2.ts", 0.8)]);
		const results = await retriever.retrieve([method1, method2], "test", {});
		expect(results.size).toBe(2);
		expect(results.get("m1")).toHaveLength(1);
		expect(results.get("m2")).toHaveLength(1);
	});

	it("keeps method insertion order even when a later method finishes first", async () => {
		const retriever = new ParallelRetriever();
		let releaseSlow: (() => void) | undefined;
		const slowGate = new Promise<void>((resolve) => {
			releaseSlow = resolve;
		});
		const slow: RetrievalMethod = {
			describe: () => ({
				name: "slow",
				type: "vector",
				description: "",
				status: "active",
				capabilities: [],
			}),
			retrieve: async () => {
				await slowGate;
				return [makeResult("a", "shared.ts", 1)];
			},
		};
		const fast: RetrievalMethod = {
			describe: () => ({
				name: "fast",
				type: "hybrid",
				description: "",
				status: "active",
				capabilities: [],
			}),
			retrieve: async () => {
				releaseSlow?.();
				return [makeResult("b", "shared.ts", 1)];
			},
		};
		const results = await retriever.retrieveWithDiagnostics([slow, fast], "test", {});
		expect([...results.results.keys()]).toEqual(["slow", "fast"]);
		const merged = new ResultMerger().merge(results.results, { topK: 1, dedup: true });
		expect(merged[0]?.id).toBe("a");
	});

	it("isolates errors — one failure does not affect others", async () => {
		const retriever = new ParallelRetriever();
		const goodMethod = makeMockMethod("good", [makeResult("a", "f1.ts", 1.0)]);
		const badMethod: RetrievalMethod = {
			describe: () => ({
				name: "bad",
				type: "posix" as const,
				description: "",
				status: "active" as const,
				capabilities: [],
			}),
			retrieve: vi.fn().mockRejectedValue(new Error("backend down")),
		};
		const results = await retriever.retrieve([goodMethod, badMethod], "test", {});
		expect(results.get("good")).toHaveLength(1);
		expect(results.get("bad")).toEqual([]);
	});
	it("retrieveWithDiagnostics preserves partial results and records path-free method failures", async () => {
		const retriever = new ParallelRetriever();
		const goodMethod = makeMockMethod("good", [makeResult("a", "f1.ts", 1.0)]);
		const badMethod: RetrievalMethod = {
			describe: () => ({
				name: "bad",
				type: "posix" as const,
				description: "",
				status: "active" as const,
				capabilities: [],
			}),
			retrieve: vi.fn().mockRejectedValue(new Error("spawn /Users/x/bin/thing ENOENT")),
		};
		const { results, diagnostics } = await retriever.retrieveWithDiagnostics([goodMethod, badMethod], "test", {});

		expect(results.get("good")).toHaveLength(1);
		expect(results.get("bad")).toEqual([]);
		const diag = diagnostics.find((d) => d.source === "bad");
		expect(diag?.code).toBe("retrieval-method-failed");
		expect(diag?.severity).toBe("warning");
		// The failure reaches the operator as thrown, paths included.
		expect(diag?.message).toContain("spawn /Users/x/bin/thing ENOENT");
		expect(diag?.reason).toContain("spawn /Users/x/bin/thing ENOENT");
		expect(diagnostics.some((d) => d.source === "good")).toBe(false);
	});

	it("retrieveWithDiagnostics maps a failing minsync method to minsync-unavailable", async () => {
		const retriever = new ParallelRetriever();
		const minsync: RetrievalMethod = {
			describe: () => ({
				name: "minsync",
				type: "vector" as const,
				description: "",
				status: "active" as const,
				capabilities: [],
			}),
			retrieve: vi.fn().mockRejectedValue(new Error("spawn /opt/minsync ENOENT")),
		};
		const { diagnostics } = await retriever.retrieveWithDiagnostics([minsync], "test", {});
		expect(diagnostics[0]?.code).toBe("minsync-unavailable");
		expect(diagnostics[0]?.message).toContain("spawn /opt/minsync ENOENT");
	});

	it("retrieveWithDiagnostics reports no diagnostics when all methods succeed", async () => {
		const retriever = new ParallelRetriever();
		const { diagnostics } = await retriever.retrieveWithDiagnostics(
			[makeMockMethod("m1", [makeResult("a", "f1.ts", 1)])],
			"test",
			{},
		);
		expect(diagnostics).toEqual([]);
	});

	it("retrieveWithDiagnostics reports which surfaces were not searched, quoting the error", async () => {
		const retriever = new ParallelRetriever();
		const locked = (name: string): RetrievalMethod => ({
			describe: () => ({
				name,
				type: "hybrid" as const,
				description: "",
				status: "active" as const,
				capabilities: [],
			}),
			retrieve: vi.fn().mockRejectedValue(new Error("another sync is in progress for /Users/x/workspace")),
		});
		const discord: RetrievalMethod = {
			describe: () => ({
				name: "discord-hybrid",
				type: "hybrid" as const,
				description: "",
				status: "active" as const,
				capabilities: [],
				datasourceId: "discord",
			}),
			retrieve: vi.fn().mockResolvedValue([makeResult("d", "/discord/guild/chunks/1", 1)]),
		};

		const { results, diagnostics, unsearched } = await retriever.retrieveWithDiagnostics(
			[locked("minsync"), locked("hybrid"), discord],
			"test",
			{},
		);

		expect(results.get("discord-hybrid")).toHaveLength(1);
		expect(unsearched).toHaveLength(1);
		expect(unsearched[0]).toMatchObject({ surface: "minsync", methods: ["hybrid", "minsync"] });
		// The workspace path in the error is kept: that is what makes the lock debuggable.
		expect(unsearched[0]?.reason).toContain("another sync is in progress for /Users/x/workspace");
		expect(diagnostics.every((d) => d.reason === unsearched[0]?.reason)).toBe(true);
	});

	it("retrieveWithDiagnostics reports a failing datasource under its datasource id", async () => {
		const retriever = new ParallelRetriever();
		const datasource: RetrievalMethod = {
			describe: () => ({
				name: "discord-hybrid",
				type: "hybrid" as const,
				description: "",
				status: "active" as const,
				capabilities: [],
				datasourceId: "discord",
			}),
			retrieve: vi.fn().mockRejectedValue(new Error("spawn discrawl ENOENT")),
		};
		const { unsearched } = await retriever.retrieveWithDiagnostics([datasource], "test", {});
		expect(unsearched).toHaveLength(1);
		expect(unsearched[0]).toMatchObject({ surface: "discord", methods: ["discord-hybrid"] });
		expect(unsearched[0]?.reason).toContain("spawn discrawl ENOENT");
	});

	it("retrieveWithDiagnostics keeps the embedding dimension mismatch text intact", async () => {
		const retriever = new ParallelRetriever();
		const minsync: RetrievalMethod = {
			describe: () => ({
				name: "minsync",
				type: "vector" as const,
				description: "",
				status: "active" as const,
				capabilities: [],
			}),
			retrieve: vi
				.fn()
				.mockRejectedValue(
					new Error("configured embedder dimension 1024 does not match indexed dimension 768; reindex required."),
				),
		};
		const { unsearched } = await retriever.retrieveWithDiagnostics([minsync], "test", {});
		expect(unsearched[0]?.surface).toBe("minsync");
		expect(unsearched[0]?.reason).toContain(
			"configured embedder dimension 1024 does not match indexed dimension 768; reindex required.",
		);
	});

	it("retrieveWithDiagnostics reports one entry per surface even when methods fail differently", async () => {
		const retriever = new ParallelRetriever();
		const locked = (name: string, message: string): RetrievalMethod => ({
			describe: () => ({
				name,
				type: "hybrid" as const,
				description: "",
				status: "active" as const,
				capabilities: [],
			}),
			retrieve: vi.fn().mockRejectedValue(new Error(message)),
		});

		const { unsearched } = await retriever.retrieveWithDiagnostics(
			[
				locked("minsync", "another sync is in progress for /tmp/workspace"),
				locked("hybrid", "dimension mismatch at /tmp/index"),
			],
			"test",
			{},
		);

		expect(unsearched).toHaveLength(1);
		expect(unsearched[0]?.surface).toBe("minsync");
		expect(unsearched[0]?.methods).toEqual(["hybrid", "minsync"]);
		expect(unsearched[0]?.reason).toContain("another sync is in progress for /tmp/workspace");
		expect(unsearched[0]?.reason).toContain("dimension mismatch at /tmp/index");
	});

	it("retrieveWithDiagnostics reports no unsearched surfaces when all methods succeed", async () => {
		const retriever = new ParallelRetriever();
		const { unsearched } = await retriever.retrieveWithDiagnostics(
			[makeMockMethod("m1", [makeResult("a", "f1.ts", 1)])],
			"test",
			{},
		);
		expect(unsearched).toEqual([]);
	});
});
