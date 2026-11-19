import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { averageRankers, type RankingMatrix, trainRanker } from "../../src/ltr/wasm-trainer.ts";
import { parseXgboostModel } from "../../src/ltr/xgboost-model.ts";

interface Packed {
	rows: (number | null)[][];
	labels: number[];
	groups: number[];
}
interface Parity {
	xgboost: string;
	train: Packed;
	valid: Packed;
	test: Packed;
	runs: Record<string, { rounds: number; bestScore: number; testMargins: number[] }>;
}

const parity = JSON.parse(
	readFileSync(join(import.meta.dirname, "..", "fixtures", "ltr", "train-parity.json"), "utf-8"),
) as Parity;
const names = Array.from({ length: 8 }, (_, i) => `f${i}`);

function matrix(set: Packed): RankingMatrix {
	return {
		rows: set.rows.map((row) => row.map((value) => (value === null ? Number.NaN : value))),
		labels: set.labels,
		groups: set.groups,
	};
}

describe("trainRanker (WASM XGBoost)", () => {
	// The reference numbers were produced by the Python xgboost library on the same data and parameters, so these
	// tests fail if the WASM build ever stops learning the same model as the library AutoRAG used to call.
	for (const seed of [0, 1, 2, 3]) {
		it(`learns what Python xgboost learns (seed ${seed}): same rounds, same score, same margins`, async () => {
			const expected = parity.runs[String(seed)];
			if (expected === undefined) throw new Error("fixture has no such seed");
			const trained = await trainRanker({
				train: matrix(parity.train),
				valid: matrix(parity.valid),
				featureNames: names,
				seed,
			});
			expect(trained.rounds).toBe(expected.rounds);
			expect(trained.validNdcg10).toBeCloseTo(expected.bestScore, 6);
			const model = parseXgboostModel(trained.modelJson);
			expect(model.treeCount).toBe(expected.rounds);
			expect(model.featureNames).toEqual(names);
			parity.test.rows.forEach((row, index) => {
				const features = row.map((value) => (value === null ? Number.NaN : value));
				expect(model.predict(features)).toBeCloseTo(expected.testMargins[index] as number, 4);
			});
		});
	}

	it("keeps only the trees up to the best round, not the rounds spent waiting for early stopping", async () => {
		const trained = await trainRanker({
			train: matrix(parity.train),
			valid: matrix(parity.valid),
			featureNames: names,
			seed: 1,
		});
		expect(trained.roundsRun).toBeGreaterThan(trained.rounds);
		expect(parseXgboostModel(trained.modelJson).treeCount).toBe(trained.rounds);
	});

	it("is deterministic for a seed and different seeds can pick different models", async () => {
		const run = (seed: number) =>
			trainRanker({ train: matrix(parity.train), valid: matrix(parity.valid), featureNames: names, seed });
		const [a, b, c] = await Promise.all([run(0), run(0), run(2)]);
		expect(a.modelJson).toBe(b.modelJson);
		expect(a.modelJson).not.toBe(c.modelJson);
	});

	it("rejects data it cannot train on with the reason", async () => {
		const good = matrix(parity.train);
		await expect(
			trainRanker({ train: { ...good, groups: [1, 2] }, valid: matrix(parity.valid), featureNames: names, seed: 0 }),
		).rejects.toThrow(/groups/u);
		await expect(
			trainRanker({ train: good, valid: matrix(parity.valid), featureNames: names.slice(1), seed: 0 }),
		).rejects.toThrow(/features/u);
		await expect(
			trainRanker({
				train: { rows: [], labels: [], groups: [] },
				valid: matrix(parity.valid),
				featureNames: names,
				seed: 0,
			}),
		).rejects.toThrow(/no training rows/u);
	});
});

describe("averageRankers", () => {
	async function trained(seed: number) {
		return trainRanker({ train: matrix(parity.train), valid: matrix(parity.valid), featureNames: names, seed });
	}

	it("builds one model whose margin is the mean of the seed models' margins", async () => {
		const models = await Promise.all([0, 1, 3].map(trained));
		const merged = parseXgboostModel(averageRankers(models.map((model) => model.modelJson)));
		const singles = models.map((model) => parseXgboostModel(model.modelJson));
		expect(merged.treeCount).toBe(singles.reduce((sum, model) => sum + model.treeCount, 0));
		expect(merged.featureNames).toEqual(names);
		for (const row of parity.test.rows.slice(0, 80)) {
			const features = row.map((value) => (value === null ? Number.NaN : value));
			const mean = singles.reduce((sum, model) => sum + model.predict(features), 0) / singles.length;
			expect(merged.predict(features)).toBeCloseTo(mean, 5);
		}
	});

	it("is the identity for one model, and refuses models of different features or nothing", async () => {
		const one = await trained(2);
		const alone = parseXgboostModel(averageRankers([one.modelJson]));
		const original = parseXgboostModel(one.modelJson);
		const features = (parity.test.rows[0] ?? []).map((value) => (value === null ? Number.NaN : value));
		expect(alone.predict(features)).toBeCloseTo(original.predict(features), 6);
		expect(() => averageRankers([])).toThrow(/no models/u);
		const other = await trainRanker({
			train: { ...matrix(parity.train), rows: matrix(parity.train).rows.map((row) => row.slice(1)) },
			valid: { ...matrix(parity.valid), rows: matrix(parity.valid).rows.map((row) => row.slice(1)) },
			featureNames: names.slice(1),
			seed: 0,
		});
		expect(() => averageRankers([one.modelJson, other.modelJson])).toThrow(/features/u);
	});
});
