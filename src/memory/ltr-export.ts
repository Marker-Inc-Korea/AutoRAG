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
 * - Labels are graded 0–3 from what the run did with the chunk ({@link ltrLabel}); LightGBM's default `label_gain`
 *   (2^label − 1) covers them.
 * - Features describe the chunk as the retrieval channels saw it before the model chose anything. How the model was
 *   shown the chunk is position bias, not relevance, so it is written to the `.position` sidecar for LightGBM's
 *   unbiased LambdaRank and never becomes a feature.
 * - A channel that did not find the chunk leaves its rank/score missing (`nan`), not zero: rank 0 would read as
 *   "better than first".
 * - The split is by whole run in time order, newest to validation, so no query has rows on both sides and the
 *   validation set looks like the future traffic a model would serve.
 */

/** Channels a retrieval observation is attributed to; each gets the same four features. */
const CHANNELS = ["dense", "sparse", "hybrid", "datasource", "web", "other"] as const;
type Channel = (typeof CHANNELS)[number];

const CHANNEL_FEATURES = CHANNELS.flatMap((channel) => [
	`${channel}_best_rank`,
	`${channel}_max_score`,
	`${channel}_reciprocal_rank_sum`,
	`${channel}_hits`,
]);

/** Column order of every exported feature row. Changing it changes {@link RANKING_LOG_SCHEMA_VERSION}'s meaning only for exports. */
export const LTR_FEATURE_NAMES = [
	...CHANNEL_FEATURES,
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
	"origin_local",
	"origin_datasource",
	"origin_web",
	"candidate_pool_size",
	"search_query_count",
] as const;

/** One exported row; `features` follows {@link LTR_FEATURE_NAMES}, NaN = missing. */
export interface LtrRow {
	readonly runId: string;
	readonly chunkId: string;
	readonly source: string;
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

export interface LtrManifest {
	readonly schemaVersion: number;
	readonly featureNames: readonly string[];
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

function channelOf(observation: RankingObservation, candidate: RankingCandidate): Channel {
	switch (observation.method) {
		case "minsync":
			return "dense";
		case "minsync-sparse":
			return "sparse";
		case "minsync-hybrid":
			return "hybrid";
		case "web_search":
			return "web";
	}
	if (observation.surface === "web") return "web";
	if (observation.surface.startsWith("datasource:") || observation.surface.startsWith("search_datasource_")) {
		return "datasource";
	}
	const origin = candidate.origin;
	return origin !== undefined && origin !== "local files" && origin !== "web" ? "datasource" : "other";
}

function maxOrNaN(values: readonly number[]): number {
	return values.length === 0 ? Number.NaN : Math.max(...values);
}

function candidateFeatures(candidate: RankingCandidate, run: RankingRunRecord): number[] {
	const byChannel: Record<Channel, RankingObservation[]> = {
		dense: [],
		sparse: [],
		hybrid: [],
		datasource: [],
		web: [],
		other: [],
	};
	for (const observation of candidate.observations) byChannel[channelOf(observation, candidate)].push(observation);
	const channelValues = CHANNELS.flatMap((channel) => {
		const hits = byChannel[channel];
		return [
			hits.length === 0 ? Number.NaN : Math.min(...hits.map((hit) => hit.rank)),
			maxOrNaN(hits.map((hit) => hit.score)),
			hits.reduce((sum, hit) => sum + 1 / hit.rank, 0),
			hits.length,
		];
	});
	const observations = candidate.observations;
	const origin = candidate.origin;
	return [
		...channelValues,
		CHANNELS.filter((channel) => byChannel[channel].length > 0).length,
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
		origin === "local files" ? 1 : 0,
		origin !== undefined && origin !== "local files" && origin !== "web" ? 1 : 0,
		origin === "web" ? 1 : 0,
		run.candidates.length,
		run.searchQueries.length,
	];
}

function toRows(run: RankingRunRecord): LtrRow[] {
	return run.candidates.map((candidate) => ({
		runId: run.runId,
		chunkId: candidate.chunkId,
		source: candidate.source,
		label: ltrLabel(candidate.labels),
		position: candidate.exposures.length === 0 ? 0 : Math.min(...candidate.exposures.map((entry) => entry.position)),
		features: candidateFeatures(candidate, run),
	}));
}

function split(groups: readonly LtrRow[][]): LtrSplit {
	return { rows: groups.flat(), groups: groups.map((group) => group.length) };
}

export function buildLtrDataset(runs: readonly RankingRunRecord[], options: BuildLtrDatasetOptions): LtrDataset {
	const { validFraction } = options;
	if (!(validFraction >= 0 && validFraction < 1)) {
		throw new Error(`validFraction must be in [0, 1), got ${validFraction}`);
	}
	const ordered = [...runs].sort((a, b) => a.startedAt - b.startedAt || a.runId.localeCompare(b.runId));
	let singleCandidate = 0;
	let noLabelVariance = 0;
	const groups: LtrRow[][] = [];
	for (const run of ordered) {
		if (run.candidates.length < 2) {
			singleCandidate += 1;
			continue;
		}
		const rows = toRows(run);
		if (new Set(rows.map((row) => row.label)).size < 2) {
			noLabelVariance += 1;
			continue;
		}
		groups.push(rows);
	}
	const validCount = groups.length < 2 ? 0 : Math.min(groups.length - 1, Math.round(groups.length * validFraction));
	const train = split(groups.slice(0, groups.length - validCount));
	const valid = split(groups.slice(groups.length - validCount));
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
				metric: "ndcg",
				eval_at: "1,3,5,10",
				lambdarank_position_bias_regularization: 0,
			},
		},
	};
}

function formatValue(value: number): string {
	return Number.isFinite(value) ? String(value) : "nan";
}

/** Write one split; returns the files it wrote (none for an empty split). */
function writeSplit(directory: string, name: "train" | "valid", data: LtrSplit): string[] {
	if (data.rows.length === 0) return [];
	const table = join(directory, `${name}.tsv`);
	const header = ["label", ...LTR_FEATURE_NAMES].join("\t");
	const body = data.rows.map((row) => [String(row.label), ...row.features.map(formatValue)].join("\t"));
	writeFileSync(table, `${[header, ...body].join("\n")}\n`, "utf-8");
	writeFileSync(`${table}.query`, `${data.groups.join("\n")}\n`, "utf-8");
	writeFileSync(`${table}.position`, `${data.rows.map((row) => row.position).join("\n")}\n`, "utf-8");
	const rows = join(directory, `${name}.rows.jsonl`);
	writeFileSync(
		rows,
		`${data.rows.map(({ runId, chunkId, source, label, position }) => JSON.stringify({ runId, chunkId, source, label, position })).join("\n")}\n`,
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
