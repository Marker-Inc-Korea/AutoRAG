import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import {
	buildServingDataset,
	LTR_SERVING_FEATURE_NAMES,
	type ServingRow,
	type ServingSplit,
} from "../memory/ltr-export.ts";
import { RankingLogStore, rankingLogPath } from "../memory/ranking-log.ts";
import {
	type LtrHoldoutMetrics,
	type LtrModelMeta,
	ltrDirectory,
	readLtrState,
	saveLtrModel,
	writeLtrState,
} from "./model-store.ts";
import { averageRankers, type RankingMatrix, type TrainedRanker, trainRanker } from "./wasm-trainer.ts";
import { parseXgboostModel } from "./xgboost-model.ts";

/** The companion metric may fall this far below the baseline and still pass when the other metric's 95% CI clears zero. */
const COMPANION_TOLERANCE = 0.005;

/**
 * The training gate. A model passes when either:
 * - its MRR gain over the reranker's own order on the holdout is statistically distinguishable from zero
 *   (95% bootstrap CI lower bound > 0) and nDCG@10 does not drop beyond tolerance, OR
 * - its nDCG@10 gain on the holdout is statistically distinguishable from zero (95% bootstrap CI lower bound > 0)
 *   and MRR does not drop beyond tolerance.
 *
 * A point estimate above baseline is not enough: on a few hundred queries it is as often luck as signal.
 */
export function holdoutPasses(holdout: LtrHoldoutMetrics): boolean {
	if (holdout.queries === 0) return false;
	const mrrPasses =
		holdout.deltaMrrCi95[0] > 0 && holdout.model.ndcg10 >= holdout.baseline.ndcg10 - COMPANION_TOLERANCE;
	const ndcgPasses =
		holdout.deltaNdcg10Ci95 !== undefined &&
		holdout.deltaNdcg10Ci95[0] > 0 &&
		holdout.model.mrr >= holdout.baseline.mrr - COMPANION_TOLERANCE;
	return mrrPasses || ndcgPasses;
}

/** Labelled search runs (after dropping runs that teach nothing) below which AutoRAG does not train. */
export const MIN_TRAINING_RUNS = 300;
/** Share of the newest eligible runs used for early stopping and for judging the finished model. */
const VALID_FRACTION = 0.15;
const HOLDOUT_FRACTION = 0.2;
const BOOTSTRAP_SAMPLES = 2_000;
/**
 * One XGBoost model is fitted per seed and the stored model is their average. XGBoost's row and column subsampling
 * makes a single fit depend on its seed (on one real log, seeds moved the holdout MRR by about 0.007 either way), so
 * the gate judges the ensemble that is served, not whichever seed happened to be tried first.
 */
export const TRAINING_SEEDS: readonly number[] = [0, 1, 2, 3, 4];

export interface RankedRow {
	readonly runId: string;
	readonly label: number;
	readonly score: number;
}

export interface RankingScores {
	readonly queries: number;
	readonly ndcg10: number;
	readonly mrr: number;
	/** Reciprocal rank of the first relevant row per query, in query order. */
	readonly perQueryMrr: readonly number[];
	readonly perQueryNdcg10: readonly number[];
}

function dcg(labels: readonly number[]): number {
	let total = 0;
	for (const [index, label] of labels.slice(0, 10).entries()) total += (2 ** label - 1) / Math.log2(index + 2);
	return total;
}

/** Mean MRR and nDCG@10 of `rows` ordered by descending score within each run; ties keep input order. Rows of one run are contiguous. */
export function evaluateRanking(rows: readonly RankedRow[]): RankingScores {
	const perQueryMrr: number[] = [];
	const perQueryNdcg10: number[] = [];
	let start = 0;
	while (start < rows.length) {
		let end = start;
		while (end < rows.length && rows[end]?.runId === rows[start]?.runId) end += 1;
		const group = rows.slice(start, end);
		const ranked = group
			.map((row, index) => ({ row, index }))
			.sort((a, b) => b.row.score - a.row.score || a.index - b.index);
		const labels = ranked.map((entry) => entry.row.label);
		const firstRelevant = labels.findIndex((label) => label > 0);
		perQueryMrr.push(firstRelevant < 0 ? 0 : 1 / (firstRelevant + 1));
		const ideal = dcg([...labels].sort((a, b) => b - a));
		perQueryNdcg10.push(ideal === 0 ? 0 : dcg(labels) / ideal);
		start = end;
	}
	const mean = (values: readonly number[]): number =>
		values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
	return {
		queries: perQueryMrr.length,
		ndcg10: mean(perQueryNdcg10),
		mrr: mean(perQueryMrr),
		perQueryMrr,
		perQueryNdcg10,
	};
}

/** Small deterministic PRNG so the same log trains to the same verdict. */
function mulberry32(seed: number): () => number {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function bootstrapCi95(deltas: readonly number[]): [number, number] {
	if (deltas.length === 0) return [0, 0];
	const random = mulberry32(7);
	const means: number[] = [];
	for (let sample = 0; sample < BOOTSTRAP_SAMPLES; sample += 1) {
		let sum = 0;
		for (let draw = 0; draw < deltas.length; draw += 1) sum += deltas[Math.floor(random() * deltas.length)] as number;
		means.push(sum / deltas.length);
	}
	means.sort((a, b) => a - b);
	return [
		means[Math.floor(0.025 * BOOTSTRAP_SAMPLES)] as number,
		means[Math.floor(0.975 * BOOTSTRAP_SAMPLES)] as number,
	];
}

export type TrainStatus = "trained" | "rejected" | "insufficient-data" | "failed";

export interface TrainResult {
	readonly status: TrainStatus;
	readonly reason?: string;
	/** Present for `trained` and `rejected`: the model that was stored. */
	readonly model?: LtrModelMeta;
	/** Labelled runs available when the attempt started. */
	readonly eligibleRuns: number;
	readonly loggedRuns: number;
}

export interface TrainOptions {
	readonly memoryPath: string;
	readonly now?: () => number;
	readonly signal?: AbortSignal;
}

function toMatrix(split: ServingSplit): RankingMatrix {
	return {
		rows: split.rows.map((row) => row.features),
		labels: split.rows.map((row) => row.label),
		groups: split.groups,
	};
}

function score(rows: readonly ServingRow[], predict: (row: ServingRow) => number): RankedRow[] {
	return rows.map((row) => ({ runId: row.runId, label: row.label, score: predict(row) }));
}

const RERANK_RANK_INDEX = LTR_SERVING_FEATURE_NAMES.indexOf("rerank_rank");

/**
 * Train an XGBoost ranker on the ranking log and store it with its holdout verdict.
 *
 * Only runs where the baseline reranker returned candidates are used, and only the features the pipeline has at that
 * moment ({@link LTR_SERVING_FEATURE_NAMES}). One model is fitted per seed in {@link TRAINING_SEEDS} (XGBoost compiled
 * to WebAssembly, in process) and averaged into the model that is stored. The newest runs are held out: that averaged
 * model is scored there against the reranker's own order, and it carries `passedGate` only by {@link holdoutPasses}.
 * A rejected model is still stored so the operator can inspect it, but `latest` never selects it.
 */
export async function trainLtrModel(options: TrainOptions): Promise<TrainResult> {
	const now = options.now ?? Date.now;
	const directory = ltrDirectory(options.memoryPath);
	const logPath = rankingLogPath(options.memoryPath);
	const runs = existsSync(logPath)
		? (() => {
				const store = new RankingLogStore(logPath);
				try {
					return store.list().runs;
				} finally {
					store.close();
				}
			})()
		: [];
	const dataset = buildServingDataset(runs, { validFraction: VALID_FRACTION, holdoutFraction: HOLDOUT_FRACTION });
	const base = { eligibleRuns: dataset.runs, loggedRuns: runs.length };
	const remember = (result: string): void =>
		writeLtrState(directory, { lastAttemptRuns: runs.length, lastAttemptAt: now(), lastAttemptResult: result });

	if (dataset.runs < MIN_TRAINING_RUNS) {
		// A fresh install has no log: nothing to remember, and no directory to create.
		if (runs.length > 0) remember("insufficient-data");
		return {
			...base,
			status: "insufficient-data",
			reason: `${dataset.runs} usable search runs in the log; at least ${MIN_TRAINING_RUNS} are needed (runs that taught nothing are skipped: ${dataset.skipped.noRerank} without a rerank, ${dataset.skipped.noLabelVariance} with nothing cited)`,
		};
	}

	options.signal?.throwIfAborted();
	const fitted: TrainedRanker[] = [];
	try {
		for (const seed of TRAINING_SEEDS) {
			fitted.push(
				await trainRanker({
					train: toMatrix(dataset.train),
					valid: toMatrix(dataset.valid),
					featureNames: dataset.featureNames,
					seed,
				}),
			);
			options.signal?.throwIfAborted();
		}
	} catch (error) {
		if (options.signal?.aborted) throw error;
		const detail = error instanceof Error ? error.message : String(error);
		remember(`failed: ${detail}`);
		return { ...base, status: "failed", reason: `XGBoost training failed: ${detail}` };
	}

	const modelJson = averageRankers(fitted.map((entry) => entry.modelJson));
	const model = parseXgboostModel(modelJson);
	const baseline = evaluateRanking(
		score(dataset.holdout.rows, (row) => -(row.features[RERANK_RANK_INDEX] ?? Number.POSITIVE_INFINITY)),
	);
	const served = evaluateRanking(score(dataset.holdout.rows, (row) => model.predict(row.features)));
	const seedMrr = fitted.map((entry) => {
		const single = parseXgboostModel(entry.modelJson);
		return evaluateRanking(score(dataset.holdout.rows, (row) => single.predict(row.features))).mrr;
	});
	const mrrDeltas = served.perQueryMrr.map((value, index) => value - (baseline.perQueryMrr[index] as number));
	const ndcgDeltas = served.perQueryNdcg10.map((value, index) => value - (baseline.perQueryNdcg10[index] as number));
	const holdout: LtrHoldoutMetrics = {
		queries: served.queries,
		baseline: { ndcg10: baseline.ndcg10, mrr: baseline.mrr },
		model: { ndcg10: served.ndcg10, mrr: served.mrr },
		deltaMrrCi95: bootstrapCi95(mrrDeltas),
		deltaNdcg10Ci95: bootstrapCi95(ndcgDeltas),
	};
	const passedGate = holdoutPasses(holdout);
	const meta: LtrModelMeta = {
		id: `xgb-${new Date(now()).toISOString().replace(/[-:T]/gu, "").slice(0, 14)}-${randomUUID().slice(0, 6)}`,
		createdAt: now(),
		featureNames: [...LTR_SERVING_FEATURE_NAMES],
		objective: "rank:pairwise",
		trainedOnRuns: dataset.train.groups.length + dataset.valid.groups.length,
		loggedRunsAtTraining: runs.length,
		holdout,
		passedGate,
		trainer: { xgboost: fitted[0]?.xgboostVersion ?? "unknown" },
		seeds: TRAINING_SEEDS.length,
		seedMrr,
	};
	mkdirSync(directory, { recursive: true });
	saveLtrModel(directory, meta, modelJson);
	remember(passedGate ? `trained ${meta.id}` : `rejected ${meta.id}`);
	const summary = `holdout ${holdout.queries} queries: MRR ${baseline.mrr.toFixed(3)} -> ${served.mrr.toFixed(3)}, nDCG@10 ${baseline.ndcg10.toFixed(3)} -> ${served.ndcg10.toFixed(3)}`;
	return passedGate
		? { ...base, status: "trained", model: meta, reason: summary }
		: { ...base, status: "rejected", model: meta, reason: `did not beat the reranker's order: ${summary}` };
}

/** True when the log has grown enough since the last attempt that training again is worth a run. */
export function shouldRetrain(input: {
	readonly loggedRuns: number;
	readonly lastAttemptRuns: number | undefined;
	readonly minNewRuns: number;
}): boolean {
	if (input.loggedRuns < MIN_TRAINING_RUNS) return false;
	return input.loggedRuns - (input.lastAttemptRuns ?? 0) >= input.minNewRuns;
}

export { readLtrState };
