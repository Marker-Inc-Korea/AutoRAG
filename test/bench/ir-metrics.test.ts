import { describe, expect, it } from "vitest";
import { mrrAtK, ndcgAtK, pairedBootstrapDiff, recallAtK } from "./ir-metrics.ts";

const qrels = new Map([
	["d1", 1],
	["d3", 1],
]);

describe("ndcgAtK", () => {
	it("is 1 when every relevant document leads the ranking", () => {
		expect(ndcgAtK(["d1", "d3", "x"], qrels, 10)).toBeCloseTo(1, 10);
	});

	it("discounts a relevant document by log2(rank + 1)", () => {
		// d1 at rank 2, d3 at rank 4: DCG = 1/log2(3) + 1/log2(5); ideal = 1 + 1/log2(3).
		const dcg = 1 / Math.log2(3) + 1 / Math.log2(5);
		const ideal = 1 + 1 / Math.log2(3);
		expect(ndcgAtK(["x", "d1", "y", "d3"], qrels, 10)).toBeCloseTo(dcg / ideal, 10);
	});

	it("ignores relevant documents ranked below the cutoff", () => {
		expect(ndcgAtK(["x", "y", "d1"], qrels, 2)).toBe(0);
	});

	it("counts a document once even if the ranking repeats it", () => {
		expect(ndcgAtK(["d1", "d1", "d3"], qrels, 10)).toBeCloseTo((1 + 1 / Math.log2(3)) / (1 + 1 / Math.log2(3)), 10);
	});

	it("is 0 for a query without relevant documents", () => {
		expect(ndcgAtK(["a"], new Map(), 10)).toBe(0);
	});
});

describe("recallAtK and mrrAtK", () => {
	it("recall is the share of relevant documents inside the cutoff", () => {
		expect(recallAtK(["d1", "x", "d3"], qrels, 2)).toBe(0.5);
		expect(recallAtK(["d1", "x", "d3"], qrels, 3)).toBe(1);
	});

	it("mrr is the reciprocal of the first relevant rank within the cutoff", () => {
		expect(mrrAtK(["x", "y", "d3"], qrels, 10)).toBeCloseTo(1 / 3, 10);
		expect(mrrAtK(["x", "y", "d3"], qrels, 2)).toBe(0);
	});
});

describe("pairedBootstrapDiff", () => {
	it("reports a zero-width interval at zero when both systems are identical", () => {
		const scores = [0.1, 0.5, 0.9, 0.3];
		const diff = pairedBootstrapDiff(scores, scores, { resamples: 200, seed: 7 });
		expect(diff.mean).toBe(0);
		expect(diff.low).toBe(0);
		expect(diff.high).toBe(0);
	});

	it("puts a consistent improvement's interval above zero and is reproducible for a seed", () => {
		const base = Array.from({ length: 60 }, (_, index) => (index % 5) / 10);
		const better = base.map((score) => score + 0.1);
		const first = pairedBootstrapDiff(better, base, { resamples: 500, seed: 3 });
		const second = pairedBootstrapDiff(better, base, { resamples: 500, seed: 3 });
		expect(first.mean).toBeCloseTo(0.1, 10);
		expect(first.low).toBeGreaterThan(0.09);
		expect(first).toEqual(second);
	});

	it("rejects score lists of different lengths", () => {
		expect(() => pairedBootstrapDiff([1], [1, 2], { resamples: 10, seed: 1 })).toThrow(/same length/u);
	});
});
