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
		expect(diag?.message).not.toContain("/Users/");
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
		expect(diagnostics[0]?.message).not.toContain("/opt/");
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
});
