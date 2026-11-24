import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { listLtrModels, ltrDirectory } from "../../src/ltr/model-store.ts";
import {
	evaluateRanking,
	holdoutPasses,
	MIN_TRAINING_RUNS,
	shouldRetrain,
	TRAINING_SEEDS,
	trainLtrModel,
} from "../../src/ltr/trainer.ts";
import { parseXgboostModel } from "../../src/ltr/xgboost-model.ts";
import { RankingLogStore, rankingLogPath } from "../../src/memory/ranking-log.ts";
import { recordedRun } from "./helpers.ts";

let root: string;
let memoryPath: string;
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ltr-train-"));
	memoryPath = join(root, "memory.json");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

function seedLog(count: number): void {
	const store = new RankingLogStore(rankingLogPath(memoryPath));
	for (let index = 0; index < count; index += 1) {
		// Gold chunk is pool index 0: dense rank 1, and last in the (reversed) reranker order.
		store.append(
			recordedRun(`run-${String(index).padStart(3, "0")}`, 100_000 + index * 1_000, { gold: 0 }).finish(true),
		);
	}
	store.close();
}

describe("evaluateRanking", () => {
	it("computes MRR and nDCG@10 per query", () => {
		const rows = [
			{ runId: "a", label: 0, score: 3 },
			{ runId: "a", label: 3, score: 2 },
			{ runId: "a", label: 0, score: 1 },
			{ runId: "b", label: 3, score: 9 },
			{ runId: "b", label: 0, score: 1 },
		];
		const result = evaluateRanking(rows);
		expect(result.queries).toBe(2);
		expect(result.mrr).toBeCloseTo((0.5 + 1) / 2, 6);
		expect(result.ndcg10).toBeCloseTo((1 / Math.log2(3) + 1) / 2, 6);
		expect(result.perQueryMrr).toEqual([0.5, 1]);
	});
});

describe("holdoutPasses", () => {
	const verdict = (base: number, model: number, ci: [number, number], ndcgBase = 0.8, ndcgModel = 0.8) => ({
		queries: 100,
		baseline: { ndcg10: ndcgBase, mrr: base },
		model: { ndcg10: ndcgModel, mrr: model },
		deltaMrrCi95: ci,
	});

	it("passes a significant MRR gain, even when nDCG@10 moved by float noise", () => {
		expect(holdoutPasses(verdict(0.765, 0.783, [0.004, 0.032], 0.79816, 0.79813))).toBe(true);
	});

	it("rejects a gain whose interval includes zero: a point estimate above baseline is not evidence", () => {
		expect(holdoutPasses(verdict(0.765, 0.783, [-0.01, 0.045]))).toBe(false);
	});

	it("rejects a significant MRR gain that costs real nDCG@10", () => {
		expect(holdoutPasses(verdict(0.7, 0.75, [0.01, 0.09], 0.8, 0.77))).toBe(false);
	});

	it("rejects an empty holdout and a model that is worse", () => {
		expect(holdoutPasses({ ...verdict(0.7, 0.8, [0.01, 0.2]), queries: 0 })).toBe(false);
		expect(holdoutPasses(verdict(0.8, 0.7, [-0.2, -0.01]))).toBe(false);
	});
});

describe("trainLtrModel (WASM XGBoost, no Python)", () => {
	it("refuses to train on too few labelled runs and says how many it needs", async () => {
		seedLog(MIN_TRAINING_RUNS - 5);
		const result = await trainLtrModel({ memoryPath });
		expect(result.status).toBe("insufficient-data");
		expect(result.reason).toContain(String(MIN_TRAINING_RUNS));
		expect(listLtrModels(ltrDirectory(memoryPath))).toHaveLength(0);
	});

	it("requires at least 300 usable runs: 299 is refused, 300 is trained on", async () => {
		expect(MIN_TRAINING_RUNS).toBe(300);
		seedLog(299);
		const refused = await trainLtrModel({ memoryPath });
		expect(refused.status).toBe("insufficient-data");
		expect(refused.eligibleRuns).toBe(299);
		expect(shouldRetrain({ loggedRuns: 299, lastAttemptRuns: undefined, minNewRuns: 1 })).toBe(false);
		expect(shouldRetrain({ loggedRuns: 300, lastAttemptRuns: undefined, minNewRuns: 1 })).toBe(true);
	});

	it("learns a pattern that beats the reranker and stores a model that passed the gate", async () => {
		// Gold is always the dense rank 1 chunk and the stub reranker always puts it last: a pattern worth learning.
		seedLog(MIN_TRAINING_RUNS + 60);
		const result = await trainLtrModel({ memoryPath, now: () => 9_000_000 });
		expect(result.status, result.reason).toBe("trained");
		const [stored] = listLtrModels(ltrDirectory(memoryPath));
		expect(stored?.passedGate).toBe(true);
		expect(stored?.holdout.model.mrr).toBeGreaterThan(stored?.holdout.baseline.mrr ?? 1);
		expect(stored?.holdout.deltaMrrCi95[0]).toBeGreaterThan(0);
		expect(stored?.trainer?.xgboost).toMatch(/^\d+\.\d+\.\d+$/u);
		expect(stored?.seeds).toEqual(TRAINING_SEEDS.length);
		expect(JSON.parse(readFileSync(join(ltrDirectory(memoryPath), "state.json"), "utf-8")).lastAttemptRuns).toBe(
			MIN_TRAINING_RUNS + 60,
		);
	});

	it("judges the exact model it stores: an ensemble of one model per seed", async () => {
		seedLog(MIN_TRAINING_RUNS + 60);
		const first = await trainLtrModel({ memoryPath, now: () => 1_000 });
		const second = await trainLtrModel({ memoryPath, now: () => 2_000 });
		// Deterministic: the same log trains to the same verdict, so a verdict never depends on which run you got.
		expect(first.model?.holdout).toEqual(second.model?.holdout);
		expect(first.model?.seedMrr).toHaveLength(TRAINING_SEEDS.length);
		const stored = readFileSync(
			join(ltrDirectory(memoryPath), "models", first.model?.id ?? "", "model.json"),
			"utf-8",
		);
		expect(parseXgboostModel(stored).treeCount).toBeGreaterThanOrEqual(TRAINING_SEEDS.length);
	});

	it("keeps a model that does not beat the reranker, but never lets `latest` pick it", async () => {
		// Gold is already the reranker's first pick (and the dense first): nothing to learn beyond the baseline.
		const store = new RankingLogStore(rankingLogPath(memoryPath));
		for (let index = 0; index < MIN_TRAINING_RUNS + 60; index += 1) {
			store.append(
				recordedRun(`run-${String(index).padStart(3, "0")}`, 100_000 + index * 1_000, {
					gold: 0,
					rerankMode: "same",
				}).finish(true),
			);
		}
		store.close();
		const result = await trainLtrModel({ memoryPath });
		expect(result.status).toBe("rejected");
		const [stored] = listLtrModels(ltrDirectory(memoryPath));
		expect(stored?.passedGate).toBe(false);
	});

	it("remembers an attempt that found too little data, so the next search does not rescan the log", async () => {
		seedLog(MIN_TRAINING_RUNS - 5);
		await trainLtrModel({ memoryPath });
		expect(JSON.parse(readFileSync(join(ltrDirectory(memoryPath), "state.json"), "utf-8")).lastAttemptRuns).toBe(
			MIN_TRAINING_RUNS - 5,
		);
	});

	it("does not create a ranking log or a model directory when nothing was ever logged", async () => {
		const result = await trainLtrModel({ memoryPath });
		expect(result.status).toBe("insufficient-data");
		expect(existsSync(rankingLogPath(memoryPath))).toBe(false);
		expect(existsSync(ltrDirectory(memoryPath))).toBe(false);
	});

	it("stops when the caller aborts", async () => {
		seedLog(MIN_TRAINING_RUNS + 60);
		const controller = new AbortController();
		controller.abort(new Error("stop"));
		await expect(trainLtrModel({ memoryPath, signal: controller.signal })).rejects.toThrow(/stop/u);
		expect(listLtrModels(ltrDirectory(memoryPath))).toHaveLength(0);
	});
});
