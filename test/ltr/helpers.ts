import { type RankingRunRecord, RankingRunRecorder } from "../../src/memory/ranking-log.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

export function hit(source: string, content: string, score: number, method = "minsync"): RetrievalResult {
	return { id: `${source}#${content}`, source, content, score, metadata: { method } };
}

/** One query's pool: `poolSize` chunks, the reranker reverses the dense order, the `gold` chunk is the final citation. */
export function recordedRun(
	runId: string,
	startedAt: number,
	options: { poolSize?: number; gold?: number; question?: string; rerankMode?: "reverse" | "same" } = {},
): { recorder: RankingRunRecorder; pool: RetrievalResult[]; finish: (cited: boolean) => RankingRunRecord } {
	const poolSize = options.poolSize ?? 8;
	const question = options.question ?? `question ${runId} budget approval`;
	const recorder = new RankingRunRecorder({ runId, conversationId: runId, question, startedAt });
	recorder.plan({ searchQueries: [question], datasources: [], web: false });
	const pool = Array.from({ length: poolSize }, (_, index) =>
		hit(
			`/docs/${runId}-${index}.md`,
			`chunk ${index} of ${runId} about budget approval number ${index}`,
			1 - index * 0.05,
		),
	);
	recorder.retrieved({ surface: "minsync", query: question, results: pool });
	const reranked = options.rerankMode === "same" ? pool : [...pool].reverse();
	recorder.reranked({
		provider: "openrouter",
		model: "test-rerank",
		results: reranked.map((result, index) => ({ ...result, score: 0.9 - index * 0.05 })),
	});
	recorder.exposed({ surface: "baseline", query: question, results: reranked });
	const goldIndex = options.gold ?? 0;
	const goldChunk = pool[goldIndex] as RetrievalResult;
	return {
		recorder,
		pool,
		finish: (cited) => {
			const key = `${goldChunk.source}\0${goldChunk.content}`;
			if (cited) {
				recorder.cited([key], new Map([[key, 0.9]]));
				recorder.supported(new Map([[key, 0.9]]));
			}
			return recorder.finish({
				finalCitationKeys: cited ? [key] : [],
				finishedAt: startedAt + 10,
				origin: () => "local files",
			});
		},
	};
}

/**
 * A valid XGBoost `rank:pairwise` JSON model with `trees` single-split trees on one feature: margin =
 * `lowValue` below `threshold` (missing goes left too), `highValue` at or above it. A test double for the real
 * Python trainer that the TypeScript predictor can evaluate exactly.
 */
export function stumpModelJson(
	featureNames: readonly string[],
	feature: string,
	threshold: number,
	lowValue: number,
	highValue: number,
	trees = 1,
): string {
	const index = featureNames.indexOf(feature);
	if (index < 0) throw new Error(`unknown feature ${feature}`);
	const tree = (id: number) => ({
		base_weights: [0, lowValue, highValue],
		categories: [],
		categories_nodes: [],
		categories_segments: [],
		categories_sizes: [],
		default_left: [1, 0, 0],
		id,
		left_children: [1, -1, -1],
		loss_changes: [1, 0, 0],
		parents: [2147483647, 0, 0],
		right_children: [2, -1, -1],
		split_conditions: [threshold, lowValue, highValue],
		split_indices: [index, 0, 0],
		split_type: [0, 0, 0],
		sum_hessian: [1, 1, 1],
		tree_param: { num_deleted: "0", num_feature: String(featureNames.length), num_nodes: "3", size_leaf_vector: "1" },
	});
	return JSON.stringify({
		version: [3, 4, 1],
		learner: {
			attributes: {},
			feature_names: [...featureNames],
			feature_types: [],
			gradient_booster: {
				name: "gbtree",
				model: {
					cats: {},
					gbtree_model_param: { num_parallel_tree: "1", num_trees: String(trees) },
					iteration_indptr: Array.from({ length: trees + 1 }, (_, i) => i),
					tree_info: Array.from({ length: trees }, () => 0),
					trees: Array.from({ length: trees }, (_, id) => tree(id)),
				},
			},
			learner_model_param: {
				base_score: "[0E0]",
				boost_from_average: "1",
				num_class: "0",
				num_feature: String(featureNames.length),
				num_target: "1",
			},
			objective: { name: "rank:pairwise", lambdarank_param: {} },
		},
	});
}
