import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { parseXgboostModel } from "../../src/ltr/xgboost-model.ts";

const FIXTURES = join(import.meta.dirname, "..", "fixtures", "ltr");

function load(tag: string): { model: string; rows: (number | null)[][]; margin: number[] } {
	const expected = JSON.parse(readFileSync(join(FIXTURES, `${tag}.expected.json`), "utf-8")) as {
		rows: (number | null)[][];
		margin: number[];
	};
	return { model: readFileSync(join(FIXTURES, `${tag}.model.json`), "utf-8"), ...expected };
}

describe("parseXgboostModel", () => {
	for (const tag of ["rank_pairwise", "rank_ndcg"]) {
		it(`reproduces the margins the xgboost library computes for ${tag}, including missing values`, () => {
			const { model, rows, margin } = load(tag);
			const booster = parseXgboostModel(model);
			expect(booster.featureNames).toEqual(["f0", "f1", "f2", "f3", "f4", "f5"]);
			expect(rows.some((row) => row.includes(null))).toBe(true);
			for (const [index, row] of rows.entries()) {
				const features = row.map((value) => (value === null ? Number.NaN : value));
				expect(booster.predict(features)).toBeCloseTo(margin[index] as number, 4);
			}
		});
	}

	it("rejects a model that is not an XGBoost gbtree JSON", () => {
		expect(() => parseXgboostModel("not json")).toThrow(/not valid JSON/u);
		expect(() => parseXgboostModel(JSON.stringify({ learner: { gradient_booster: { name: "gblinear" } } }))).toThrow(
			/gbtree/u,
		);
	});

	it("rejects categorical splits it cannot evaluate instead of mis-scoring", () => {
		const { model } = load("rank_pairwise");
		const parsed = JSON.parse(model);
		parsed.learner.gradient_booster.model.trees[0].split_type[0] = 1;
		expect(() => parseXgboostModel(JSON.stringify(parsed))).toThrow(/categorical/u);
	});

	it("rejects a feature vector of the wrong length", () => {
		const { model } = load("rank_pairwise");
		expect(() => parseXgboostModel(model).predict([1, 2])).toThrow(/6 features/u);
	});
});
