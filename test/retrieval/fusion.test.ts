import { describe, expect, it } from "vitest";
import { reciprocalRankFusion } from "../../src/retrieval/fusion.ts";

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
		const fused = reciprocalRankFusion([
			["a", "b"],
			["a", "c", "b"],
		]);
		expect(fused.find((entry) => entry.key === "a")?.score).toBeCloseTo(2 / 61);
		expect(fused.find((entry) => entry.key === "b")?.score).toBeCloseTo(1 / 62 + 1 / 63);
	});

	it("keeps first-seen order for ties across lists", () => {
		expect(
			reciprocalRankFusion([
				["z", "b"],
				["a", "c"],
			]).map((entry) => entry.key),
		).toEqual(["z", "a", "b", "c"]);
	});

	it("tolerates missing and empty lists", () => {
		expect(reciprocalRankFusion([])).toEqual([]);
		expect(reciprocalRankFusion([[], []])).toEqual([]);
		expect(reciprocalRankFusion([[], ["a"], []])).toEqual([{ key: "a", score: 1 / 61 }]);
	});

	it("counts a repeated key once at its best rank within each list", () => {
		const fused = reciprocalRankFusion([
			["a", "a", "b"],
			["b", "a", "a"],
		]);
		expect(fused).toHaveLength(2);
		expect(fused[0]?.key).toBe("a");
		expect(fused[0]?.score).toBeCloseTo(1 / 61 + 1 / 62);
		expect(fused[1]?.score).toBeCloseTo(1 / 63 + 1 / 61);
	});
});
