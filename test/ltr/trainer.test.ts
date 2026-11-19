import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { listLtrModels, ltrDirectory } from "../../src/ltr/model-store.ts";
import { evaluateRanking, holdoutPasses, MIN_TRAINING_RUNS, trainLtrModel } from "../../src/ltr/trainer.ts";
import { LTR_SERVING_FEATURE_NAMES } from "../../src/memory/ltr-export.ts";
import { RankingLogStore, rankingLogPath } from "../../src/memory/ranking-log.ts";
import { recordedRun, stumpModelJson } from "./helpers.ts";

let root: string;
let memoryPath: string;
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ltr-train-"));
	memoryPath = join(root, "memory.json");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

/**
 * Fake trainer: a script that honours the real contract (argv: script train.tsv valid.tsv out.json; prints a JSON line)
 * but writes a stump model. `good` puts the dense-rank-1 chunk first, which is the cited one in these logs;
 * `bad` leaves the cited chunk last, exactly where the reranker put it.
 */
function fakePython(kind: "good" | "bad", fail = false): string {
	const path = join(root, `fake-python-${kind}${fail ? "-fail" : ""}.sh`);
	// Gold is always dense rank 1 and the reranker reverses the pool, so "dense rank low => high score" is the truth.
	const model =
		kind === "good"
			? stumpModelJson([...LTR_SERVING_FEATURE_NAMES], "dense_best_rank", 1.5, 5, 0)
			: stumpModelJson([...LTR_SERVING_FEATURE_NAMES], "rerank_rank", 7.5, 5, 0);
	writeFileSync(
		path,
		fail
			? "#!/bin/sh\necho 'xgboost exploded' >&2\nexit 3\n"
			: `#!/bin/sh\ncat > "$4" <<'JSON'\n${model}\nJSON\necho '{"xgboost":"fake-1.0","bestIteration":1}'\n`,
	);
	chmodSync(path, 0o755);
	return path;
}

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

describe("trainLtrModel", () => {
	it("refuses to train on too few labelled runs and says how many it needs", async () => {
		seedLog(MIN_TRAINING_RUNS - 5);
		const result = await trainLtrModel({ memoryPath, python: [fakePython("good")] });
		expect(result.status).toBe("insufficient-data");
		expect(result.reason).toContain(String(MIN_TRAINING_RUNS));
		expect(listLtrModels(ltrDirectory(memoryPath))).toHaveLength(0);
	});

	it("stores a model that beats the baseline on the holdout and marks it as passing the gate", async () => {
		seedLog(MIN_TRAINING_RUNS + 20);
		const result = await trainLtrModel({ memoryPath, python: [fakePython("good")], now: () => 9_000_000 });
		expect(result.status).toBe("trained");
		const [stored] = listLtrModels(ltrDirectory(memoryPath));
		expect(stored?.passedGate).toBe(true);
		expect(stored?.holdout.model.mrr).toBeGreaterThan(stored?.holdout.baseline.mrr ?? 1);
		expect(stored?.holdout.queries).toBeGreaterThan(0);
		expect(stored?.trainer?.xgboost).toBe("fake-1.0");
		expect(JSON.parse(readFileSync(join(ltrDirectory(memoryPath), "state.json"), "utf-8")).lastAttemptRuns).toBe(
			MIN_TRAINING_RUNS + 20,
		);
	});

	it("keeps a model that does not beat the baseline but never lets `latest` pick it", async () => {
		seedLog(MIN_TRAINING_RUNS + 20);
		const result = await trainLtrModel({ memoryPath, python: [fakePython("bad")] });
		expect(result.status).toBe("rejected");
		const [stored] = listLtrModels(ltrDirectory(memoryPath));
		expect(stored?.passedGate).toBe(false);
	});

	it("reports the trainer's own error and stores nothing when python fails", async () => {
		seedLog(MIN_TRAINING_RUNS + 20);
		const result = await trainLtrModel({ memoryPath, python: [fakePython("good", true)] });
		expect(result.status).toBe("failed");
		expect(result.reason).toContain("xgboost exploded");
		expect(listLtrModels(ltrDirectory(memoryPath))).toHaveLength(0);
	});

	it("reports a missing python as an actionable failure", async () => {
		seedLog(MIN_TRAINING_RUNS + 20);
		const result = await trainLtrModel({ memoryPath, python: [join(root, "no-such-python")] });
		expect(result.status).toBe("failed");
		expect(result.reason).toMatch(/python/iu);
	});

	it("remembers an attempt that found too little data, so the next search does not rescan the log", async () => {
		seedLog(MIN_TRAINING_RUNS - 5);
		await trainLtrModel({ memoryPath, python: [fakePython("good")] });
		expect(JSON.parse(readFileSync(join(ltrDirectory(memoryPath), "state.json"), "utf-8")).lastAttemptRuns).toBe(
			MIN_TRAINING_RUNS - 5,
		);
	});

	it("does not create a ranking log when none exists", async () => {
		const result = await trainLtrModel({ memoryPath, python: [fakePython("good")] });
		expect(result.status).toBe("insufficient-data");
		expect(existsSync(rankingLogPath(memoryPath))).toBe(false);
		expect(existsSync(ltrDirectory(memoryPath))).toBe(false);
	});
});
