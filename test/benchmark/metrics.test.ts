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

	it("counts failed queries and excludes them from successful-query latency statistics", () => {
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
		expect(metrics.latencyMs.mean).toBe(10);
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

	it("macro-averages zero-valued queries and reports successful-query mean and nearest-rank percentiles", () => {
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
		expect(metrics.latencyMs).toEqual({ mean: 2.5, p50: 2, p95: 4 });
		expect(metrics.successAt).toEqual({ "1": 0.25, "5": 0.25 });
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

	it("applies rank 10 but not rank 11 to MRR and nDCG at 10", () => {
		const qrels = [
			{ queryId: "q10", documentId: "at-ten", relevance: 1 },
			{ queryId: "q11", documentId: "at-eleven", relevance: 1 },
		];
		const metrics = evaluateRun(
			[
				record("q10", { hits: [{ documentId: "at-ten", score: 1, rank: 10 }] }),
				record("q11", { hits: [{ documentId: "at-eleven", score: 1, rank: 11 }] }),
			],
			qrels,
		)[0]!;

		expect(metrics.mrrAt10).toBe(0.05);
		expect(metrics.ndcgAt10).toBeCloseTo(1 / (2 * Math.log2(11)), 10);
	});

	it("reports Recall@100 and Success@5 at their exact boundaries", () => {
		const qrels = [
			{ queryId: "q100", documentId: "at-one-hundred", relevance: 1 },
			{ queryId: "q5", documentId: "at-five", relevance: 1 },
			{ queryId: "q6", documentId: "at-six", relevance: 1 },
		];
		const metrics = evaluateRun(
			[
				record("q100", { hits: [{ documentId: "at-one-hundred", score: 1, rank: 100 }] }),
				record("q5", { hits: [{ documentId: "at-five", score: 1, rank: 5 }] }),
				record("q6", { hits: [{ documentId: "at-six", score: 1, rank: 6 }] }),
			],
			qrels,
		)[0]!;

		expect(metrics.recallAt).toEqual({ "5": 1 / 3, "10": 2 / 3, "100": 1 });
		expect(metrics.successAt).toEqual({ "1": 0, "5": 1 / 3 });
	});

	it("scores zero-positive qrels and all-failed methods as finite zeroes", () => {
		const zeroPositive = evaluateRun(
			[record("zero", { hits: [{ documentId: "judged-zero", score: 1, rank: 1 }] })],
			[{ queryId: "zero", documentId: "judged-zero", relevance: 0 }],
		)[0]!;
		const allFailed = evaluateRun(
			[
				record("failed-1", { latencyMs: 50, errorCode: "retrieval-failed" }),
				record("failed-2", { latencyMs: 100, errorCode: "retrieval-failed" }),
			],
			[
				{ queryId: "failed-1", documentId: "a", relevance: 1 },
				{ queryId: "failed-2", documentId: "b", relevance: 1 },
			],
		)[0]!;

		expect(zeroPositive.recallAt).toEqual({ "5": 0, "10": 0, "100": 0 });
		expect(zeroPositive.mrrAt10).toBe(0);
		expect(zeroPositive.successAt).toEqual({ "1": 0, "5": 0 });
		expect(zeroPositive.ndcgAt10).toBe(0);
		expect(allFailed.latencyMs).toEqual({ mean: 0, p50: 0, p95: 0 });
		expect(allFailed.recallAt).toEqual({ "5": 0, "10": 0, "100": 0 });
	});

	it("rejects relevance whose finite individual gains overflow nDCG accumulation", () => {
		const overflowingQrels = [
			{ queryId: "overflow", documentId: "a", relevance: 1023 },
			{ queryId: "overflow", documentId: "b", relevance: 1023 },
			{ queryId: "overflow", documentId: "c", relevance: 1023 },
		];

		expect(() => evaluateRun([record("overflow")], overflowingQrels)).toThrow("metric accumulation");
	});

	it("emits only finite metric values for accepted relevance and latency inputs", () => {
		const metrics = evaluateRun(
			[record("finite-1", { latencyMs: Number.MAX_VALUE }), record("finite-2", { latencyMs: Number.MAX_VALUE })],
			[
				{ queryId: "finite-1", documentId: "a", relevance: 1023 },
				{ queryId: "finite-2", documentId: "b", relevance: 0 },
			],
		)[0]!;

		for (const value of [
			metrics.mrrAt10,
			metrics.ndcgAt10,
			...Object.values(metrics.recallAt),
			...Object.values(metrics.successAt),
			...Object.values(metrics.latencyMs),
		]) {
			expect(Number.isFinite(value)).toBe(true);
		}
		expect(metrics.latencyMs.mean).toBe(Number.MAX_VALUE);
	});
});
