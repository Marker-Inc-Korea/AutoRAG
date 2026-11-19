import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import {
	buildServingDataset,
	LTR_SERVING_FEATURE_NAMES,
	type ServingRow,
	type ServingSplit,
} from "../memory/ltr-export.ts";
import { RankingLogStore, rankingLogPath } from "../memory/ranking-log.ts";
import { spawnProcess } from "../minsync/process.ts";
import {
	type LtrHoldoutMetrics,
	type LtrModelMeta,
	ltrDirectory,
	readLtrState,
	saveLtrModel,
	writeLtrState,
} from "./model-store.ts";
import { XGBOOST_TRAIN_SCRIPT } from "./train-script.ts";
import { parseXgboostModel } from "./xgboost-model.ts";

/** nDCG@10 may fall this far below the baseline and still pass: it is a noisy metric, and the MRR interval decides. */
const NDCG_TOLERANCE = 0.005;

/**
 * The training gate. A model passes only when its MRR gain over the reranker's own order on the holdout is
 * statistically distinguishable from zero (the 95% bootstrap interval of the per-query difference lies above 0) and
 * it does not give up real nDCG@10. A point estimate above baseline is not enough: on a few hundred queries it is
 * as often luck as signal.
 */
export function holdoutPasses(holdout: LtrHoldoutMetrics): boolean {
	if (holdout.queries === 0) return false;
	if (!(holdout.deltaMrrCi95[0] > 0)) return false;
	return holdout.model.ndcg10 >= holdout.baseline.ndcg10 - NDCG_TOLERANCE;
}

/** Labelled search runs (after dropping runs that teach nothing) below which AutoRAG does not train. */
export const MIN_TRAINING_RUNS = 100;
/** Share of the newest eligible runs used for early stopping and for judging the finished model. */
const VALID_FRACTION = 0.15;
const HOLDOUT_FRACTION = 0.2;
const TRAIN_TIMEOUT_MS = 10 * 60 * 1000;
const BOOTSTRAP_SAMPLES = 2_000;

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
	/** Python command prefix, e.g. `["uv","run","--with","xgboost","--with","numpy","python"]`. Default: see {@link defaultPythonCommand}. */
	readonly python?: readonly string[];
	readonly now?: () => number;
	readonly signal?: AbortSignal;
}

function onPath(executable: string): boolean {
	return (process.env.PATH ?? "")
		.split(delimiter)
		.some((directory) => directory.length > 0 && existsSync(join(directory, executable)));
}

/** `uv` runs the trainer with xgboost resolved on demand; otherwise a `python3` that already has xgboost and numpy. */
export function defaultPythonCommand(): readonly string[] {
	return onPath("uv")
		? ["uv", "run", "--quiet", "--with", "xgboost", "--with", "numpy", "python"]
		: [process.platform === "win32" ? "python" : "python3"];
}

function tsv(split: ServingSplit, path: string): void {
	const header = ["label", ...LTR_SERVING_FEATURE_NAMES].join("\t");
	const body = split.rows.map((row: ServingRow) =>
		[String(row.label), ...row.features.map((value) => (Number.isFinite(value) ? String(value) : "nan"))].join("\t"),
	);
	writeFileSync(path, `${[header, ...body].join("\n")}\n`, "utf-8");
	writeFileSync(`${path}.query`, `${split.groups.join("\n")}\n`, "utf-8");
}

function score(rows: readonly ServingRow[], predict: (row: ServingRow) => number): RankedRow[] {
	return rows.map((row) => ({ runId: row.runId, label: row.label, score: predict(row) }));
}

const RERANK_RANK_INDEX = LTR_SERVING_FEATURE_NAMES.indexOf("rerank_rank");
/** The xgboost version the trainer script reported on its last stdout line; undefined when it printed none. */
function readTrainerVersion(stdout: string): string | undefined {
	const line = stdout.trim().split("\n").at(-1);
	if (line === undefined) return undefined;
	try {
		const parsed: unknown = JSON.parse(line);
		if (typeof parsed === "object" && parsed !== null && "xgboost" in parsed && typeof parsed.xgboost === "string") {
			return parsed.xgboost;
		}
	} catch {
		// A trainer that printed something else still produced the model; the version is only informational.
	}
	return undefined;
}

/**
 * Train an XGBoost ranker on the ranking log and store it with its holdout verdict.
 *
 * Only runs where the baseline reranker returned candidates are used, and only the features the pipeline has at that
 * moment ({@link LTR_SERVING_FEATURE_NAMES}). The newest runs are held out: the finished model is scored there against
 * the reranker's own order, and it carries `passedGate` only when it is strictly better on MRR and no worse on
 * nDCG@10. A rejected model is still stored so the operator can inspect it, but `latest` never selects it.
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

	const work = mkdtempSync(join(tmpdir(), "autorag-ltr-train-"));
	try {
		const script = join(work, "train_xgboost.py");
		const modelPath = join(work, "model.json");
		writeFileSync(script, XGBOOST_TRAIN_SCRIPT, "utf-8");
		tsv(dataset.train, join(work, "train.tsv"));
		tsv(dataset.valid, join(work, "valid.tsv"));
		const python = options.python ?? defaultPythonCommand();
		const [command, ...prefix] = python;
		if (command === undefined) throw new Error("ltr python command is empty");
		const result = await spawnProcess(
			command,
			[...prefix, script, join(work, "train.tsv"), join(work, "valid.tsv"), modelPath],
			work,
			{ timeoutMs: TRAIN_TIMEOUT_MS, ...(options.signal === undefined ? {} : { signal: options.signal }) },
		);
		if (!result.ok || !existsSync(modelPath)) {
			const detail = result.timedOut
				? "the trainer timed out"
				: result.stderr.trim().split("\n").slice(-6).join("\n") || `exit code ${String(result.code)}`;
			const reason = `XGBoost training failed (python command: ${python.join(" ")}). Training needs Python with the xgboost and numpy packages, or uv on PATH. ${detail}`;
			remember(`failed: ${detail}`);
			return { ...base, status: "failed", reason };
		}

		const modelJson = readFileSync(modelPath, "utf-8");
		const model = parseXgboostModel(modelJson);
		const baseline = evaluateRanking(
			score(dataset.holdout.rows, (row) => -(row.features[RERANK_RANK_INDEX] ?? Number.POSITIVE_INFINITY)),
		);
		const served = evaluateRanking(score(dataset.holdout.rows, (row) => model.predict(row.features)));
		const deltas = served.perQueryMrr.map((value, index) => value - (baseline.perQueryMrr[index] as number));
		const holdout: LtrHoldoutMetrics = {
			queries: served.queries,
			baseline: { ndcg10: baseline.ndcg10, mrr: baseline.mrr },
			model: { ndcg10: served.ndcg10, mrr: served.mrr },
			deltaMrrCi95: bootstrapCi95(deltas),
		};
		const passedGate = holdoutPasses(holdout);
		const trainerInfo = readTrainerVersion(result.stdout);
		const meta: LtrModelMeta = {
			id: `xgb-${new Date(now()).toISOString().replace(/[-:T]/gu, "").slice(0, 14)}-${randomUUID().slice(0, 6)}`,
			createdAt: now(),
			featureNames: [...LTR_SERVING_FEATURE_NAMES],
			objective: "rank:pairwise",
			trainedOnRuns: dataset.train.groups.length + dataset.valid.groups.length,
			loggedRunsAtTraining: runs.length,
			holdout,
			passedGate,
			...(trainerInfo !== undefined ? { trainer: { xgboost: trainerInfo } } : {}),
		};
		mkdirSync(directory, { recursive: true });
		saveLtrModel(directory, meta, modelJson);
		remember(passedGate ? `trained ${meta.id}` : `rejected ${meta.id}`);
		const summary = `holdout ${holdout.queries} queries: MRR ${baseline.mrr.toFixed(3)} -> ${served.mrr.toFixed(3)}, nDCG@10 ${baseline.ndcg10.toFixed(3)} -> ${served.ndcg10.toFixed(3)}`;
		return passedGate
			? { ...base, status: "trained", model: meta, reason: summary }
			: { ...base, status: "rejected", model: meta, reason: `did not beat the reranker's order: ${summary}` };
	} finally {
		rmSync(work, { recursive: true, force: true });
	}
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
