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

	it("deduplicates by source keeping highest score", () => {
		const merger = new ResultMerger();
		const results = new Map([
			["method1", [makeResult("a", "file1.ts", 0.9)]],
			["method2", [makeResult("b", "file1.ts", 0.5)]],
		]);
		const merged = merger.merge(results, { topK: 10, dedup: true });
		expect(merged.length).toBe(1);
		expect(merged[0].id).toBe("a");
	});

	it("ranks repeated cross-method evidence above a comparable singleton", () => {
		const merger = new ResultMerger();
		const repeatedSource = "opaque:document:repeat-7f9c";
		const singletonSource = "opaque:document:single-4a2d";
		const results = new Map([
			[
				"bm25",
				[
					makeResult("single", singletonSource, 0.95),
					makeResult("repeat-bm25", repeatedSource, 0.94),
					makeResult("bm25-floor", "opaque:floor:bm25", 0.1),
				],
			],
			[
				"minsync",
				[makeResult("repeat-vector", repeatedSource, 0.9), makeResult("vector-floor", "opaque:floor:vector", 0.1)],
			],
		]);

		const merged = merger.merge(results, { topK: 5, dedup: true });

		expect(merged.map((result) => result.source).slice(0, 2)).toEqual([repeatedSource, singletonSource]);
		expect(merged[0].score).toBeGreaterThan(merged[1].score);
		expect(merged[0].score).toBeLessThanOrEqual(1);
		expect(merged[0].source).toBe(repeatedSource);
		expect(merged[0].metadata).toMatchObject({
			aggregateHitCount: 2,
			aggregateMethods: ["bm25", "minsync"],
		});
	});

	it("reinforces several same-method chunks with a bounded diminishing bonus", () => {
		const merger = new ResultMerger();
		const repeatedSource = "opaque:document:same-method";
		const results = new Map([
			[
				"bm25",
				[
					makeResult("single", "opaque:document:single", 0.96),
					makeResult("repeat-1", repeatedSource, 0.95),
					makeResult("repeat-2", repeatedSource, 0.94),
					makeResult("repeat-3", repeatedSource, 0.93),
					makeResult("repeat-4", repeatedSource, 0.92),
					makeResult("repeat-5", repeatedSource, 0.91),
					makeResult("floor", "opaque:floor", 0.1),
				],
			],
		]);

		const merged = merger.merge(results, { topK: 5, dedup: true });
		const repeated = merged.find((result) => result.source === repeatedSource);
		const singleton = merged.find((result) => result.source === "opaque:document:single");

		expect(merged[0].source).toBe(repeatedSource);
		expect(repeated?.score).toBeGreaterThan(singleton?.score ?? Number.POSITIVE_INFINITY);
		expect(repeated?.score).toBeLessThanOrEqual(1);
		expect(repeated?.metadata.aggregateHitCount).toBe(5);
		expect(repeated?.id).toBe("repeat-1");
	});

	it("does not reward exact duplicate evidence identities", () => {
		const merger = new ResultMerger();
		const repeatedSource = "opaque:document:duplicate";
		const duplicate = makeResult("same-chunk-id", repeatedSource, 0.9);
		const results = new Map([
			[
				"bm25",
				[
					makeResult("single", "opaque:document:single", 0.95),
					duplicate,
					{ ...duplicate },
					{ ...duplicate },
					makeResult("floor", "opaque:floor", 0.1),
				],
			],
		]);

		const merged = merger.merge(results, { topK: 5, dedup: true });
		const repeated = merged.find((result) => result.source === repeatedSource);

		expect(merged[0].source).toBe("opaque:document:single");
		expect(repeated?.metadata.aggregateHitCount).toBeUndefined();
		expect(repeated?.score).toBeLessThanOrEqual(1);
	});

	it("always merges exact sources when deduplication is disabled", () => {
		const merger = new ResultMerger();
		const results = new Map([
			["method1", [makeResult("a", "opaque:same", 0.9), makeResult("b", "opaque:same", 0.8)]],
		]);

		const merged = merger.merge(results, { topK: 10, dedup: false });

		expect(merged).toHaveLength(1);
		expect(merged[0]).toMatchObject({
			id: "a",
			source: "opaque:same",
			metadata: {
				aggregateHitCount: 2,
				aggregateMethods: ["method1"],
			},
		});
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
