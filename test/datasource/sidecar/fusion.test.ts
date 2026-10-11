import { describe, expect, it } from "vitest";
import {
	cosine,
	decodeVector,
	encodeVector,
	reciprocalRankFusion,
	topKByCosine,
} from "../../../src/datasource/sidecar/fusion.ts";

describe("vector codec", () => {
	it("round-trips float32 values and rejects a length that is not a whole number of floats", () => {
		const original = [0.25, -1.5, 3];
		expect(decodeVector(encodeVector(original))).toEqual(original);
		expect(decodeVector(new Uint8Array(7))).toBeUndefined();
	});
});

describe("cosine", () => {
	it("is 1 for parallel, 0 for orthogonal, and 0 (not NaN) for a zero or mismatched vector", () => {
		expect(cosine([1, 2], [2, 4])).toBeCloseTo(1);
		expect(cosine([1, 0], [0, 1])).toBe(0);
		expect(cosine([0, 0], [1, 1])).toBe(0);
		expect(cosine([1, 2, 3], [1, 2])).toBe(0);
	});
});

describe("topKByCosine", () => {
	const rows = [
		{ key: "a", vector: [1, 0] },
		{ key: "b", vector: [0.9, 0.1] },
		{ key: "c", vector: [0, 1] },
		{ key: "d", vector: [-1, 0] },
	];

	it("returns the k best keys by descending score and drops non-positive similarities", () => {
		expect(topKByCosine([1, 0], rows, 2).map((hit) => hit.key)).toEqual(["a", "b"]);
		expect(topKByCosine([1, 0], rows, 10).map((hit) => hit.key)).toEqual(["a", "b"]);
	});

	it("breaks score ties by key so ordering is deterministic", () => {
		const tied = [
			{ key: "z", vector: [1, 0] },
			{ key: "m", vector: [1, 0] },
		];
		expect(topKByCosine([1, 0], tied, 2).map((hit) => hit.key)).toEqual(["m", "z"]);
	});

	it("returns nothing for k <= 0", () => {
		expect(topKByCosine([1, 0], rows, 0)).toEqual([]);
	});
});

describe("reciprocalRankFusion", () => {
	it("ranks an item found by both lists above items found by only one, even when it is never first", () => {
		const fused = reciprocalRankFusion([
			["a", "both", "c"],
			["x", "both", "z"],
		]);
		expect(fused[0]?.key).toBe("both");
		expect(fused.map((entry) => entry.key).sort()).toEqual(["a", "both", "c", "x", "z"]);
	});

	it("scores 1/(k+rank) per list with the standard k=60 and sums across lists", () => {
		const [top] = reciprocalRankFusion([["a"], ["a"]]);
		expect(top?.score).toBeCloseTo(2 / 61);
	});

	it("is stable for ties (earlier list wins) and tolerates empty lists", () => {
		expect(reciprocalRankFusion([["a"], ["b"]]).map((entry) => entry.key)).toEqual(["a", "b"]);
		expect(reciprocalRankFusion([[], []])).toEqual([]);
	});

	it("does not double count a key repeated inside one list", () => {
		const [top] = reciprocalRankFusion([["a", "a"]]);
		expect(top?.score).toBeCloseTo(1 / 61);
	});
});
