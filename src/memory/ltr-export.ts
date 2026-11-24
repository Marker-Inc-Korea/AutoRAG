import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { CITATION_RELEVANCE_FLOOR } from "../agent/citation-relevance.ts";
import { EVIDENCE_SUPPORT_THRESHOLD } from "./judged-evidence.ts";
import {
	RANKING_LOG_SCHEMA_VERSION,
	type RankingCandidate,
	type RankingLabels,
	type RankingObservation,
	type RankingRunRecord,
} from "./ranking-log.ts";

/**
 * Turns the ranking log into a LightGBM LambdaRank dataset.
 *
 * - One search run is one query group; its rows are contiguous and `.query` lists the group sizes in row order.
 * - Hybrid channel representation:
 *   1. Categorical `origin_id` and `method_id` (vocabularies fit on `train` only; unseen future datasources become
 *      `nan`) paired with `primary_rank`, `primary_raw_score`, and within-method `primary_norm_score` so tree splits
 *      learn per-datasource score calibration without adding new columns when datasources change.
 *   2. Dedicated `dense_*`, `sparse_*`, and `hybrid_*` columns for local files, the only surface where two retrieval
 *      modes (dense vectors + SPLADE sparse terms) index the same chunk and can intersect on one candidate.
 *   3. Point-in-time `origin_prior_win_rate` / `origin_prior_candidates` across strictly finished earlier runs so a
 *      newly connected datasource gets historical calibration as soon as runs finish, even before retraining.
 * - Exposure is position bias, not relevance, so it goes to `.position` and never becomes a feature.
 * - A channel that did not find the chunk leaves its rank/score missing (`nan`), not zero.
 * - The split is by whole run in time order, newest to validation.
 */

const LOCAL_CHANNELS = ["dense", "sparse", "hybrid"] as const;
type LocalChannel = (typeof LOCAL_CHANNELS)[number];

const LOCAL_CHANNEL_FEATURES = LOCAL_CHANNELS.flatMap((channel) => [
	`${channel}_best_rank`,
	`${channel}_max_score`,
	`${channel}_reciprocal_rank_sum`,
	`${channel}_hits`,
]);

export const LTR_CATEGORICAL_FEATURE_NAMES = ["origin_id", "method_id"] as const;

/** Column order of every exported feature row. */
export const LTR_FEATURE_NAMES = [
	...LTR_CATEGORICAL_FEATURE_NAMES,
	"primary_rank",
	"primary_raw_score",
	"primary_norm_score",
	"primary_reciprocal_rank_sum",
	...LOCAL_CHANNEL_FEATURES,
	"channel_count",
	"query_hit_count",
	"observation_count",
	"rerank_rank",
	"rerank_score",
	"tool_rerank_score_max",
	"merger_retrieval_method_count",
	"merger_source_chunk_count",
	"merger_duplicate_hit_count",
	"text_bm25",
	"text_question_coverage",
	"content_chars_log",
	"memory_prior_support",
	"memory_similar_question_support",
	"origin_prior_win_rate",
	"origin_prior_candidates",
	"origin_local",
	"origin_datasource",
	"origin_web",
	"candidate_pool_size",
	"search_query_count",
] as const;

export const LTR_CATEGORICAL_FEATURE_INDICES: readonly number[] = LTR_CATEGORICAL_FEATURE_NAMES.map((name) =>
	LTR_FEATURE_NAMES.indexOf(name),
);

/** One exported row; `features` follows {@link LTR_FEATURE_NAMES}, NaN = missing. */
export interface LtrRow {
	readonly runId: string;
	readonly chunkId: string;
	readonly source: string;
	readonly origin?: string;
	readonly label: number;
	/** Best position the model was shown the chunk at (1 = first); 0 = never shown. */
	readonly position: number;
	readonly features: readonly number[];
}

export interface LtrSplit {
	readonly rows: readonly LtrRow[];
	/** Rows per query group, in row order (LightGBM `.query`). */
	readonly groups: readonly number[];
}

export interface LtrVocabularies {
	readonly origin: Readonly<Record<string, number>>;
	readonly method: Readonly<Record<string, number>>;
}

export interface LtrManifest {
	readonly schemaVersion: number;
	readonly featureNames: readonly string[];
	readonly categoricalFeatures: {
		readonly names: readonly string[];
		readonly indices: readonly number[];
	};
	readonly vocabularies: LtrVocabularies;
	readonly labels: Readonly<Record<string, string>>;
	readonly train: { readonly groups: number; readonly rows: number };
	readonly valid: { readonly groups: number; readonly rows: number };
	readonly labelCounts: Readonly<Record<string, number>>;
	readonly runs: number;
	readonly skippedGroups: { readonly singleCandidate: number; readonly noLabelVariance: number };
	readonly firstRunAt?: number;
	readonly lastRunAt?: number;
	readonly validFraction: number;
	readonly lightgbm: Readonly<Record<string, string | number | boolean>>;
}

export interface LtrDataset {
	readonly train: LtrSplit;
	readonly valid: LtrSplit;
	readonly manifest: LtrManifest;
}

export interface BuildLtrDatasetOptions {
	/** Share of the newest eligible runs held out for validation, in [0, 1). */
	readonly validFraction: number;
}

const LABEL_DESCRIPTIONS = {
	"3": `cited by the final answer and Jev P(supports) >= ${EVIDENCE_SUPPORT_THRESHOLD} (stored in retrieval memory)`,
	"2": "cited by the final answer",
	"1": `cited only by a draft answer and not found unrelated (Jev P(relevant) >= ${CITATION_RELEVANCE_FLOOR} or unjudged)`,
	"0": "retrieved but not cited, or cited and found unrelated",
} as const;

/** Graded relevance of one candidate; see {@link LABEL_DESCRIPTIONS}. */
export function ltrLabel(labels: RankingLabels): number {
	if (labels.citedFinal) return labels.support !== undefined && labels.support >= EVIDENCE_SUPPORT_THRESHOLD ? 3 : 2;
	if (labels.modelCited && (labels.relevance === undefined || labels.relevance >= CITATION_RELEVANCE_FLOOR)) return 1;
	return 0;
}

function localChannelOf(observation: RankingObservation): LocalChannel | undefined {
	switch (observation.method) {
		case "minsync":
			return "dense";
		case "minsync-sparse":
			return "sparse";
		case "minsync-hybrid":
			return "hybrid";
		default:
			return undefined;
	}
}

function primaryObservation(observations: readonly RankingObservation[]): RankingObservation | undefined {
	let best: RankingObservation | undefined;
	for (const observation of observations) {
		if (
			best === undefined ||
			observation.rank < best.rank ||
			(observation.rank === best.rank && observation.score > best.score)
		) {
			best = observation;
		}
	}
	return best;
}

function computeRunListBounds(run: RankingRunRecord): Map<string, { min: number; max: number }> {
	const bounds = new Map<string, { min: number; max: number }>();
	for (const candidate of run.candidates) {
		for (const observation of candidate.observations) {
			const key = `${observation.stage}\0${observation.surface}\0${observation.method}\0${observation.query}`;
			const current = bounds.get(key);
			if (current === undefined) {
				bounds.set(key, { min: observation.score, max: observation.score });
			} else {
				if (observation.score < current.min) current.min = observation.score;
				if (observation.score > current.max) current.max = observation.score;
			}
		}
	}
	return bounds;
}

interface OriginPriorStats {
	candidates: number;
	wins: number;
}

function buildVocabularies(trainRuns: readonly RankingRunRecord[]): LtrVocabularies {
	const origins = new Set<string>();
	const methods = new Set<string>();
	for (const run of trainRuns) {
		for (const candidate of run.candidates) {
			if (candidate.origin !== undefined && candidate.origin.length > 0) origins.add(candidate.origin);
			for (const observation of candidate.observations) {
				if (observation.method.length > 0) methods.add(observation.method);
			}
		}
	}
	const origin: Record<string, number> = {};
	for (const [index, key] of [...origins].sort().entries()) origin[key] = index;
	const method: Record<string, number> = {};
	for (const [index, key] of [...methods].sort().entries()) method[key] = index;
	return { origin, method };
}

function maxOrNaN(values: readonly number[]): number {
	return values.length === 0 ? Number.NaN : Math.max(...values);
}

function candidateFeatures(
	candidate: RankingCandidate,
	run: RankingRunRecord,
	listBounds: ReadonlyMap<string, { min: number; max: number }>,
	vocabularies: LtrVocabularies,
	originPriors: ReadonlyMap<string, Readonly<OriginPriorStats>>,
): number[] {
	const observations = candidate.observations;
	const primary = primaryObservation(observations);
	const origin = candidate.origin;
	const originId = origin !== undefined && origin in vocabularies.origin ? vocabularies.origin[origin] : Number.NaN;
	const methodId =
		primary !== undefined && primary.method in vocabularies.method ? vocabularies.method[primary.method] : Number.NaN;

	const normScores = observations.map((observation) => {
		const bounds = listBounds.get(
			`${observation.stage}\0${observation.surface}\0${observation.method}\0${observation.query}`,
		);
		if (bounds === undefined) return 1;
		const range = bounds.max - bounds.min;
		return range > 0 ? (observation.score - bounds.min) / range : 1;
	});

	const byLocal: Record<LocalChannel, RankingObservation[]> = { dense: [], sparse: [], hybrid: [] };
	for (const observation of observations) {
		const channel = localChannelOf(observation);
		if (channel !== undefined) byLocal[channel].push(observation);
	}
	const localChannelValues = LOCAL_CHANNELS.flatMap((channel) => {
		const hits = byLocal[channel];
		return [
			hits.length === 0 ? Number.NaN : Math.min(...hits.map((hit) => hit.rank)),
			maxOrNaN(hits.map((hit) => hit.score)),
			hits.reduce((sum, hit) => sum + 1 / hit.rank, 0),
			hits.length,
		];
	});

	const prior = origin === undefined ? undefined : originPriors.get(origin);
	const originPriorCandidates = origin === undefined ? Number.NaN : (prior?.candidates ?? 0);
	const originPriorWinRate = prior !== undefined && prior.candidates > 0 ? prior.wins / prior.candidates : Number.NaN;

	return [
		originId,
		methodId,
		primary?.rank ?? Number.NaN,
		primary?.score ?? Number.NaN,
		maxOrNaN(normScores),
		observations.reduce((sum, hit) => sum + 1 / hit.rank, 0),
		...localChannelValues,
		new Set(observations.map((observation) => observation.method)).size,
		new Set(observations.map((observation) => observation.query)).size,
		observations.length,
		candidate.rerank?.rank ?? Number.NaN,
		candidate.rerank?.score ?? Number.NaN,
		maxOrNaN(observations.flatMap((observation) => observation.rerankScore ?? [])),
		maxOrNaN(observations.flatMap((observation) => observation.retrievalMethodCount ?? [])),
		maxOrNaN(observations.flatMap((observation) => observation.sourceChunkCount ?? [])),
		maxOrNaN(observations.flatMap((observation) => observation.duplicateHitCount ?? [])),
		candidate.text.bm25,
		candidate.text.questionCoverage,
		Math.log1p(candidate.contentChars),
		candidate.memory.priorSupport,
		candidate.memory.similarQuestionSupport ?? Number.NaN,
		originPriorWinRate,
		originPriorCandidates,
		origin === "local files" ? 1 : 0,
		origin !== undefined && origin !== "local files" && origin !== "web" ? 1 : 0,
		origin === "web" ? 1 : 0,
		run.candidates.length,
		run.searchQueries.length,
	];
}

function toRows(
	run: RankingRunRecord,
	vocabularies: LtrVocabularies,
	originPriors: ReadonlyMap<string, Readonly<OriginPriorStats>>,
): LtrRow[] {
	const listBounds = computeRunListBounds(run);
	return run.candidates.map((candidate) => ({
		runId: run.runId,
		chunkId: candidate.chunkId,
		source: candidate.source,
		...(candidate.origin !== undefined ? { origin: candidate.origin } : {}),
		label: ltrLabel(candidate.labels),
		position: candidate.exposures.length === 0 ? 0 : Math.min(...candidate.exposures.map((entry) => entry.position)),
		features: candidateFeatures(candidate, run, listBounds, vocabularies, originPriors),
	}));
}

export function buildLtrDataset(runs: readonly RankingRunRecord[], options: BuildLtrDatasetOptions): LtrDataset {
	const { validFraction } = options;
	if (!(validFraction >= 0 && validFraction < 1)) {
		throw new Error(`validFraction must be in [0, 1), got ${validFraction}`);
	}
	const ordered = [...runs].sort((a, b) => a.startedAt - b.startedAt || a.runId.localeCompare(b.runId));
	const byFinished = [...runs].sort((a, b) => {
		const endA = Math.max(a.startedAt, a.finishedAt);
		const endB = Math.max(b.startedAt, b.finishedAt);
		return endA - endB || a.startedAt - b.startedAt || a.runId.localeCompare(b.runId);
	});

	// Sweep finished runs in causal time order so every run snapshots only strictly completed earlier runs.
	const runningOriginTotals = new Map<string, OriginPriorStats>();
	const priorsByRunId = new Map<string, Map<string, Readonly<OriginPriorStats>>>();
	let finishedCursor = 0;
	for (const run of ordered) {
		while (finishedCursor < byFinished.length) {
			const finishedRun = byFinished[finishedCursor] as RankingRunRecord;
			const effectiveEnd = Math.max(finishedRun.startedAt, finishedRun.finishedAt);
			if (!(effectiveEnd <= run.startedAt && finishedRun.startedAt < run.startedAt)) break;
			for (const candidate of finishedRun.candidates) {
				if (candidate.origin === undefined) continue;
				const stats = runningOriginTotals.get(candidate.origin) ?? { candidates: 0, wins: 0 };
				stats.candidates += 1;
				if (candidate.labels.citedFinal) stats.wins += 1;
				runningOriginTotals.set(candidate.origin, stats);
			}
			finishedCursor += 1;
		}
		const snapshot = new Map<string, Readonly<OriginPriorStats>>();
		for (const [origin, stats] of runningOriginTotals) snapshot.set(origin, { ...stats });
		priorsByRunId.set(run.runId, snapshot);
	}

	let singleCandidate = 0;
	let noLabelVariance = 0;
	const eligibleRuns: RankingRunRecord[] = [];
	for (const run of ordered) {
		if (run.candidates.length < 2) {
			singleCandidate += 1;
			continue;
		}
		if (new Set(run.candidates.map((candidate) => ltrLabel(candidate.labels))).size < 2) {
			noLabelVariance += 1;
			continue;
		}
		eligibleRuns.push(run);
	}

	const validCount =
		eligibleRuns.length < 2 ? 0 : Math.min(eligibleRuns.length - 1, Math.round(eligibleRuns.length * validFraction));
	const trainRuns = eligibleRuns.slice(0, eligibleRuns.length - validCount);
	const validRuns = eligibleRuns.slice(eligibleRuns.length - validCount);
	const vocabularies = buildVocabularies(trainRuns);

	const emptyPriors = new Map<string, Readonly<OriginPriorStats>>();
	const trainRows = trainRuns.map((run) => toRows(run, vocabularies, priorsByRunId.get(run.runId) ?? emptyPriors));
	const validRows = validRuns.map((run) => toRows(run, vocabularies, priorsByRunId.get(run.runId) ?? emptyPriors));
	const train: LtrSplit = { rows: trainRows.flat(), groups: trainRows.map((group) => group.length) };
	const valid: LtrSplit = { rows: validRows.flat(), groups: validRows.map((group) => group.length) };

	const labelCounts: Record<string, number> = {};
	for (const row of [...train.rows, ...valid.rows]) {
		labelCounts[String(row.label)] = (labelCounts[String(row.label)] ?? 0) + 1;
	}
	const first = ordered[0];
	const last = ordered.at(-1);
	return {
		train,
		valid,
		manifest: {
			schemaVersion: RANKING_LOG_SCHEMA_VERSION,
			featureNames: [...LTR_FEATURE_NAMES],
			categoricalFeatures: {
				names: [...LTR_CATEGORICAL_FEATURE_NAMES],
				indices: [...LTR_CATEGORICAL_FEATURE_INDICES],
			},
			vocabularies,
			labels: LABEL_DESCRIPTIONS,
			train: { groups: train.groups.length, rows: train.rows.length },
			valid: { groups: valid.groups.length, rows: valid.rows.length },
			labelCounts,
			runs: runs.length,
			skippedGroups: { singleCandidate, noLabelVariance },
			...(first !== undefined ? { firstRunAt: first.startedAt } : {}),
			...(last !== undefined ? { lastRunAt: last.startedAt } : {}),
			validFraction,
			lightgbm: {
				objective: "lambdarank",
				header: true,
				label_column: 0,
				categorical_feature: LTR_CATEGORICAL_FEATURE_INDICES.join(","),
				metric: "ndcg",
				eval_at: "1,3,5,10",
				lambdarank_position_bias_regularization: 0,
			},
		},
	};
}

/** Write one split; returns the files it wrote (none for an empty split). */
function writeSplit(directory: string, name: "train" | "valid", data: LtrSplit): string[] {
	if (data.rows.length === 0) return [];
	const table = join(directory, `${name}.tsv`);
	const header = ["label", ...LTR_FEATURE_NAMES].join("\t");
	const body = data.rows.map((row) =>
		[String(row.label), ...row.features.map((value) => (Number.isFinite(value) ? String(value) : "nan"))].join("\t"),
	);
	writeFileSync(table, `${[header, ...body].join("\n")}\n`, "utf-8");
	writeFileSync(`${table}.query`, `${data.groups.join("\n")}\n`, "utf-8");
	writeFileSync(`${table}.position`, `${data.rows.map((row) => row.position).join("\n")}\n`, "utf-8");
	const rows = join(directory, `${name}.rows.jsonl`);
	writeFileSync(
		rows,
		`${data.rows
			.map(({ runId, chunkId, source, origin, label, position }) =>
				JSON.stringify({ runId, chunkId, source, ...(origin !== undefined ? { origin } : {}), label, position }),
			)
			.join("\n")}\n`,
		"utf-8",
	);
	return [table, `${table}.query`, `${table}.position`, rows];
}

/**
 * Write the dataset as LightGBM text files: `<split>.tsv` (header row, label first, `nan` = missing),
 * `<split>.tsv.query` and `<split>.tsv.position` (loaded automatically beside the data file), `<split>.rows.jsonl`
 * (which run and chunk each row is, for error analysis), and `manifest.json`.
 */
export function writeLtrDataset(dataset: LtrDataset, directory: string): { readonly files: readonly string[] } {
	mkdirSync(directory, { recursive: true });
	const manifest = join(directory, "manifest.json");
	writeFileSync(manifest, `${JSON.stringify(dataset.manifest, null, 2)}\n`, "utf-8");
	return {
		files: [
			...writeSplit(directory, "train", dataset.train),
			...writeSplit(directory, "valid", dataset.valid),
			manifest,
		],
	};
}

/**
 * Serving view: the features a learning-to-rank model can have at the moment the reranker returns, and the dataset
 * built from them.
 *
 * The full export above describes a finished run. Serving happens earlier, right after the baseline rerank, so a
 * serving model may only use what exists then: the prefetch observations (not the tool calls that come later), the
 * reranker's rank and score, and per-chunk text signals. Features that depend on the rest of the run (BM25 over the
 * run's final pool, pool size, memory and origin priors, merger counts) are left out, and a candidate the reranker
 * did not return is never re-scored. Training and serving both go through {@link servingRows}, so a column cannot
 * mean one thing offline and another in production.
 */
export const LTR_SERVING_FEATURE_NAMES = [
	"primary_rank",
	"primary_raw_score",
	"primary_norm_score",
	"primary_reciprocal_rank_sum",
	"dense_best_rank",
	"dense_max_score",
	"dense_reciprocal_rank_sum",
	"dense_hits",
	"sparse_best_rank",
	"sparse_max_score",
	"sparse_reciprocal_rank_sum",
	"sparse_hits",
	"channel_count",
	"query_hit_count",
	"observation_count",
	"rerank_rank",
	"rerank_score",
	"text_question_coverage",
	"content_chars_log",
] as const;
const SERVING_FEATURE_INDICES = LTR_SERVING_FEATURE_NAMES.map((name) => LTR_FEATURE_NAMES.indexOf(name));

export interface ServingRow {
	readonly runId: string;
	readonly chunkId: string;
	readonly source: string;
	readonly label: number;
	/** Follows {@link LTR_SERVING_FEATURE_NAMES}; NaN = missing. */
	readonly features: readonly number[];
}

const NO_VOCABULARY: LtrVocabularies = { origin: {}, method: {} };

/**
 * One row per candidate the baseline reranker returned, in the log's candidate order. Features use prefetch
 * observations only, so the rows of a run snapshotted at rerank time equal the rows of its finished record.
 */
export function servingRows(run: RankingRunRecord): ServingRow[] {
	const candidates = run.candidates
		.filter((candidate) => candidate.rerank !== undefined)
		.map((candidate) => ({
			...candidate,
			observations: candidate.observations.filter((observation) => observation.stage === "prefetch"),
		}));
	const view: RankingRunRecord = { ...run, candidates };
	const bounds = computeRunListBounds(view);
	return candidates.map((candidate) => {
		const full = candidateFeatures(candidate, view, bounds, NO_VOCABULARY, new Map());
		return {
			runId: run.runId,
			chunkId: candidate.chunkId,
			source: candidate.source,
			label: ltrLabel(candidate.labels),
			features: SERVING_FEATURE_INDICES.map((index) => full[index] as number),
		};
	});
}

export interface ServingSplit {
	readonly rows: readonly ServingRow[];
	/** Rows per query group, in row order. */
	readonly groups: readonly number[];
}

export interface ServingDataset {
	readonly featureNames: readonly string[];
	readonly train: ServingSplit;
	readonly valid: ServingSplit;
	readonly holdout: ServingSplit;
	readonly skipped: {
		readonly noRerank: number;
		readonly singleCandidate: number;
		readonly noLabelVariance: number;
	};
	/** Labelled runs that went into the dataset (before the split). */
	readonly runs: number;
	readonly firstRunAt?: number;
	readonly lastRunAt?: number;
}

export interface BuildServingDatasetOptions {
	/** Share of the newest eligible runs used for early stopping, in [0, 1). */
	readonly validFraction: number;
	/** Share of the newest eligible runs kept back to judge the finished model, in [0, 1). */
	readonly holdoutFraction: number;
}

function toSplit(groups: readonly (readonly ServingRow[])[]): ServingSplit {
	return { rows: groups.flat(), groups: groups.map((group) => group.length) };
}

/**
 * Build the train/validation/holdout sets for a serving model. Whole runs are split in time order (oldest to train,
 * then validation, newest to holdout) so no query has rows on both sides and the holdout looks like the traffic the
 * model would serve next. Runs that cannot teach a ranker are counted and dropped.
 */
export function buildServingDataset(
	runs: readonly RankingRunRecord[],
	options: BuildServingDatasetOptions,
): ServingDataset {
	const { validFraction, holdoutFraction } = options;
	if (!(validFraction >= 0 && validFraction < 1) || !(holdoutFraction >= 0 && holdoutFraction < 1)) {
		throw new Error("validFraction and holdoutFraction must be in [0, 1)");
	}
	if (validFraction + holdoutFraction >= 1) throw new Error("validFraction + holdoutFraction must be below 1");
	const ordered = [...runs].sort((a, b) => a.startedAt - b.startedAt || a.runId.localeCompare(b.runId));
	let noRerank = 0;
	let singleCandidate = 0;
	let noLabelVariance = 0;
	const eligible: ServingRow[][] = [];
	for (const run of ordered) {
		const rows = servingRows(run);
		if (rows.length === 0) noRerank += 1;
		else if (rows.length < 2) singleCandidate += 1;
		else if (new Set(rows.map((row) => row.label)).size < 2) noLabelVariance += 1;
		else eligible.push(rows);
	}
	const holdoutCount = eligible.length < 3 ? 0 : Math.round(eligible.length * holdoutFraction);
	const validCount = eligible.length - holdoutCount < 2 ? 0 : Math.round(eligible.length * validFraction);
	const trainEnd = eligible.length - holdoutCount - validCount;
	const first = ordered[0];
	const last = ordered.at(-1);
	return {
		featureNames: [...LTR_SERVING_FEATURE_NAMES],
		train: toSplit(eligible.slice(0, trainEnd)),
		valid: toSplit(eligible.slice(trainEnd, trainEnd + validCount)),
		holdout: toSplit(eligible.slice(trainEnd + validCount)),
		skipped: { noRerank, singleCandidate, noLabelVariance },
		runs: eligible.length,
		...(first !== undefined ? { firstRunAt: first.startedAt } : {}),
		...(last !== undefined ? { lastRunAt: last.startedAt } : {}),
	};
}
