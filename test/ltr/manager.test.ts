import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { acquireFileLock } from "../../src/filesystem/file-lock.ts";
import {
	AUTO_TRAIN_CLAIM_MS,
	applyLtrPatch,
	DEFAULT_AUTO_TRAIN_MIN_NEW_RUNS,
	getLtrStatus,
	runAutoTrain,
} from "../../src/ltr/manager.ts";
import {
	type LtrModelMeta,
	listLtrModels,
	ltrDirectory,
	readLtrState,
	saveLtrModel,
	writeLtrState,
} from "../../src/ltr/model-store.ts";
import { MIN_TRAINING_RUNS } from "../../src/ltr/trainer.ts";
import { LTR_SERVING_FEATURE_NAMES } from "../../src/memory/ltr-export.ts";
import { RankingLogStore, rankingLogPath } from "../../src/memory/ranking-log.ts";
import { recordedRun, stumpModelJson } from "./helpers.ts";

let root: string;
let memoryPath: string;
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ltr-manager-"));
	memoryPath = join(root, "memory.json");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

function meta(id: string, createdAt: number, passedGate = true): LtrModelMeta {
	return {
		id,
		createdAt,
		featureNames: [...LTR_SERVING_FEATURE_NAMES],
		objective: "rank:pairwise",
		trainedOnRuns: 100,
		loggedRunsAtTraining: 100,
		holdout: {
			queries: 20,
			baseline: { ndcg10: 0.5, mrr: 0.5 },
			model: { ndcg10: 0.6, mrr: 0.6 },
			deltaMrrCi95: [0, 0.2],
		},
		passedGate,
	};
}
const MODEL = stumpModelJson([...LTR_SERVING_FEATURE_NAMES], "rerank_score", 0.5, 0, 1);

function logRuns(count: number): void {
	const store = new RankingLogStore(rankingLogPath(memoryPath));
	for (let i = 0; i < count; i += 1) store.append(recordedRun(`r-${i}`, 1_000 + i, { gold: 0 }).finish(true));
	store.close();
}

describe("getLtrStatus", () => {
	it("reports an empty, unconfigured install without creating any file", () => {
		const status = getLtrStatus({ memoryPath, config: undefined });
		expect(status).toMatchObject({
			enabled: false,
			selection: "latest",
			loggedRuns: 0,
			models: [],
			serving: { status: "disabled" },
		});
		expect(status.autoTrain).toEqual({ enabled: false, minNewRuns: DEFAULT_AUTO_TRAIN_MIN_NEW_RUNS, due: false });
	});

	it("says which model would serve and which models exist", () => {
		saveLtrModel(ltrDirectory(memoryPath), meta("m-1", 1_000), MODEL);
		saveLtrModel(ltrDirectory(memoryPath), meta("m-2", 2_000, false), MODEL);
		const status = getLtrStatus({ memoryPath, config: { enabled: true } });
		expect(status.serving).toEqual({ status: "active", modelId: "m-1" });
		expect(status.models.map((entry) => entry.id)).toEqual(["m-2", "m-1"]);
	});

	it("explains why nothing serves when enabled with no passing model", () => {
		saveLtrModel(ltrDirectory(memoryPath), meta("m-2", 2_000, false), MODEL);
		const status = getLtrStatus({ memoryPath, config: { enabled: true } });
		expect(status.serving.status).toBe("no-model");
	});

	it("marks retraining due only when auto-train is on, the log is big enough and has grown since the last attempt", () => {
		logRuns(MIN_TRAINING_RUNS + 10);
		const base = { memoryPath, config: { autoTrain: { enabled: true, minNewRuns: 50 } } };
		expect(getLtrStatus(base).autoTrain.due).toBe(true);
		writeLtrState(ltrDirectory(memoryPath), { lastAttemptRuns: MIN_TRAINING_RUNS + 10 - 20 });
		expect(getLtrStatus(base).autoTrain.due).toBe(false);
		expect(getLtrStatus({ memoryPath, config: { autoTrain: { enabled: false } } }).autoTrain.due).toBe(false);
	});

	it("never counts a missing log as runs", () => {
		expect(getLtrStatus({ memoryPath, config: { autoTrain: { enabled: true } } }).loggedRuns).toBe(0);
	});
});

describe("applyLtrPatch", () => {
	it("turns serving on and off without touching the other settings", () => {
		const start = { model: "xgb-1", autoTrain: { enabled: true, minNewRuns: 300 } };
		expect(applyLtrPatch(start, { enabled: true })).toEqual({ ...start, enabled: true });
		expect(applyLtrPatch({ ...start, enabled: true }, { enabled: false })).toEqual({ ...start, enabled: false });
	});

	it("selects a model, and `latest` clears the pin", () => {
		expect(applyLtrPatch({ enabled: true }, { model: "xgb-7" })).toEqual({ enabled: true, model: "xgb-7" });
		expect(applyLtrPatch({ enabled: true, model: "xgb-7" }, { model: "latest" })).toEqual({ enabled: true });
	});

	it("merges auto-train fields", () => {
		expect(applyLtrPatch({}, { autoTrain: { enabled: true } })).toEqual({ autoTrain: { enabled: true } });
		expect(applyLtrPatch({ autoTrain: { enabled: true } }, { autoTrain: { minNewRuns: 50 } })).toEqual({
			autoTrain: { enabled: true, minNewRuns: 50 },
		});
	});

	it("rejects ids that are not model ids and non-positive run counts", () => {
		expect(() => applyLtrPatch({}, { model: "../x" })).toThrow(/model/u);
		expect(() => applyLtrPatch({}, { autoTrain: { minNewRuns: 0 } })).toThrow(/minNewRuns/u);
		expect(() => applyLtrPatch({}, { autoTrain: { minNewRuns: 1.5 } })).toThrow(/minNewRuns/u);
	});
});

describe("runAutoTrain", () => {
	function fakePython(): string {
		const path = join(root, "fake-python.sh");
		const model = stumpModelJson([...LTR_SERVING_FEATURE_NAMES], "dense_best_rank", 1.5, 5, 0);
		writeFileSync(path, `#!/bin/sh\ncat > "$4" <<'JSON'\n${model}\nJSON\necho '{"xgboost":"fake"}'\n`);
		chmodSync(path, 0o755);
		return path;
	}
	const settings = { autoTrain: { enabled: true, minNewRuns: 20 } };

	it("does nothing when auto-train is off, the log is small, or it has not grown enough", async () => {
		logRuns(MIN_TRAINING_RUNS + 25);
		expect(await runAutoTrain({ memoryPath, settings: {}, python: [fakePython()] })).toBeUndefined();
		writeLtrState(ltrDirectory(memoryPath), { lastAttemptRuns: MIN_TRAINING_RUNS + 25 - 5 });
		expect(await runAutoTrain({ memoryPath, settings, python: [fakePython()] })).toBeUndefined();
		expect(listLtrModels(ltrDirectory(memoryPath))).toHaveLength(0);
	});

	it("trains once enough new runs were logged and records the attempt", async () => {
		logRuns(MIN_TRAINING_RUNS + 25);
		const result = await runAutoTrain({ memoryPath, settings, python: [fakePython()] });
		expect(result?.status).toBe("trained");
		expect(listLtrModels(ltrDirectory(memoryPath))).toHaveLength(1);
		expect(readLtrState(ltrDirectory(memoryPath)).lastAttemptRuns).toBe(MIN_TRAINING_RUNS + 25);
		// Not due again until the log grows by minNewRuns.
		expect(await runAutoTrain({ memoryPath, settings, python: [fakePython()] })).toBeUndefined();
	});

	it("leaves a live claim alone so two processes never train at once, and releases its own", async () => {
		logRuns(MIN_TRAINING_RUNS + 25);
		const dir = ltrDirectory(memoryPath);
		mkdirSync(dir, { recursive: true });
		const held = acquireFileLock(join(dir, "train.lock"), {
			timeoutMs: 0,
			staleMs: AUTO_TRAIN_CLAIM_MS,
			timeoutError: () => new Error("held"),
		});
		try {
			expect(await runAutoTrain({ memoryPath, settings, python: [fakePython()] })).toBeUndefined();
			expect(existsSync(join(dir, "models"))).toBe(false);
		} finally {
			held.release();
		}
		expect((await runAutoTrain({ memoryPath, settings, python: [fakePython()] }))?.status).toBe("trained");
		expect(existsSync(join(dir, "train.lock"))).toBe(false);
	});
});
