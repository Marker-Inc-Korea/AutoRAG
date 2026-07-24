import { describe, expect, it } from "vitest";
import { evaluateRun } from "../../benchmark/miracl/metrics.ts";
import type { Qrel, QueryRunRecord } from "../../benchmark/miracl/types.ts";

function record(queryId: string, overrides: Partial<QueryRunRecord> = {}): QueryRunRecord {
	return {
		schemaVersion: 1,
		method: "bm25",
		queryId,
		latencyMs: 10,
		hits: [],
		...overrides,
	};
}

describe("evaluateRun", () => {
	it("computes graded and binary retrieval metrics", () => {
		const qrels = [
			{ queryId: "q1", documentId: "a", relevance: 2 },
			{ queryId: "q1", documentId: "b", relevance: 1 },
			{ queryId: "q1", documentId: "x", relevance: 0 },
		];
		const records = [
			{
				schemaVersion: 1 as const,
				method: "bm25" as const,
				queryId: "q1",
				latencyMs: 10,
				hits: ["b", "x", "a"].map((documentId, index) => ({
					documentId,
					score: 3 - index,
					rank: index + 1,
				})),
			},
		];
		const metrics = evaluateRun(records, qrels)[0]!;

		expect(metrics.recallAt["5"]).toBe(1);
		expect(metrics.mrrAt10).toBe(1);
		expect(metrics.successAt["1"]).toBe(1);
		expect(metrics.ndcgAt10).toBeCloseTo(
			((2 ** 1 - 1) / Math.log2(2) + (2 ** 2 - 1) / Math.log2(4)) /
				((2 ** 2 - 1) / Math.log2(2) + (2 ** 1 - 1) / Math.log2(3)),
			10,
		);
	});

	it("counts failed queries and excludes them from latency percentiles", () => {
		const qrels = [
			{ queryId: "q1", documentId: "a", relevance: 1 },
			{ queryId: "q2", documentId: "b", relevance: 1 },
		];
		const metrics = evaluateRun(
			[
				record("q1", { hits: [{ documentId: "a", score: 1, rank: 1 }] }),
				record("q2", { latencyMs: 999, errorCode: "retrieval-failed" }),
			],
			qrels,
		)[0]!;

		expect(metrics.queryCount).toBe(2);
		expect(metrics.failureCount).toBe(1);
		expect(metrics.recallAt["5"]).toBe(0.5);
		expect(metrics.latencyMs.p95).toBe(10);
	});

	it("uses ranks for all cutoffs and discounts, not hit array positions", () => {
		const metrics = evaluateRun(
			[
				record("q1", {
					hits: [
						{ documentId: "low", score: 0.25, rank: 4 },
						{ documentId: "high", score: 0.5, rank: 2 },
					],
				}),
			],
			[
				{ queryId: "q1", documentId: "high", relevance: 2 },
				{ queryId: "q1", documentId: "low", relevance: 1 },
			],
		)[0]!;

		expect(metrics.recallAt["5"]).toBe(1);
		expect(metrics.mrrAt10).toBe(0.5);
		expect(metrics.successAt["1"]).toBe(0);
		expect(metrics.ndcgAt10).toBeCloseTo(
			((2 ** 2 - 1) / Math.log2(3) + (2 ** 1 - 1) / Math.log2(5)) /
				((2 ** 2 - 1) / Math.log2(2) + (2 ** 1 - 1) / Math.log2(3)),
			10,
		);
	});

	it("macro-averages zero-valued queries and uses nearest-rank percentiles", () => {
		const qrels: Qrel[] = [
			{ queryId: "q1", documentId: "a", relevance: 1 },
			{ queryId: "q2", documentId: "b", relevance: 1 },
			{ queryId: "q3", documentId: "c", relevance: 1 },
			{ queryId: "q4", documentId: "d", relevance: 1 },
		];
		const metrics = evaluateRun(
			[
				record("q1", { latencyMs: 4, hits: [{ documentId: "a", score: 1, rank: 1 }] }),
				record("q2", { latencyMs: 1 }),
				record("q3", { latencyMs: 3 }),
				record("q4", { latencyMs: 2 }),
			],
			qrels,
		)[0]!;

		expect(metrics.recallAt["5"]).toBe(0.25);
		expect(metrics.mrrAt10).toBe(0.25);
		expect(metrics.ndcgAt10).toBe(0.25);
		expect(metrics.latencyMs).toEqual({ p50: 2, p95: 4, p99: 4 });
	});

	it("returns method groups in deterministic order", () => {
		const qrels = [{ queryId: "q1", documentId: "a", relevance: 1 }];
		const metrics = evaluateRun([record("q1", { method: "minsync" }), record("q1", { method: "hybrid" })], qrels);

		expect(metrics.map((metric) => metric.method)).toEqual(["hybrid", "minsync"]);
	});

	it("rejects duplicate run records and qrels, record queries without qrels, and malformed numeric inputs", () => {
		const qrels = [{ queryId: "q1", documentId: "a", relevance: 1 }];

		expect(() => evaluateRun([record("q1"), record("q1")], qrels)).toThrow("duplicate query-method record");
		expect(() =>
			evaluateRun(
				[record("q1")],
				[
					{ queryId: "q1", documentId: "a", relevance: 1 },
					{ queryId: "q1", documentId: "a", relevance: 0 },
				],
			),
		).toThrow("duplicate qrel");
		expect(() => evaluateRun([record("missing")], qrels)).toThrow("no qrels");
		expect(() => evaluateRun([record("q1", { latencyMs: Number.NaN })], qrels)).toThrow("latency");
		expect(() =>
			evaluateRun([record("q1", { hits: [{ documentId: "a", score: Number.POSITIVE_INFINITY, rank: 1 }] })], qrels),
		).toThrow("score");
		expect(() => evaluateRun([record("q1", { hits: [{ documentId: "a", score: 1, rank: 0 }] })], qrels)).toThrow(
			"rank",
		);
	});

	it("rejects duplicate hit documents and ranks while treating unjudged documents as nonrelevant", () => {
		const qrels = [{ queryId: "q1", documentId: "a", relevance: 1 }];

		expect(() =>
			evaluateRun(
				[
					record("q1", {
						hits: [
							{ documentId: "a", score: 2, rank: 1 },
							{ documentId: "a", score: 1, rank: 2 },
						],
					}),
				],
				qrels,
			),
		).toThrow("duplicate hit document");
		expect(() =>
			evaluateRun(
				[
					record("q1", {
						hits: [
							{ documentId: "a", score: 2, rank: 1 },
							{ documentId: "other", score: 1, rank: 1 },
						],
					}),
				],
				qrels,
			),
		).toThrow("duplicate hit rank");
		expect(
			evaluateRun([record("q1", { hits: [{ documentId: "unjudged", score: 1, rank: 1 }] })], qrels)[0]?.recallAt[
				"5"
			],
		).toBe(0);
	});
});
