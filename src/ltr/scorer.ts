import { servingRows } from "../memory/ltr-export.ts";
import { chunkIdOf, type RankingRunRecord } from "../memory/ranking-log.ts";
import type { RetrievalResult } from "../retrieval/types.ts";
import { loadLtrModel, ltrDirectory, resolveLtrModel } from "./model-store.ts";
import type { XgboostModel } from "./xgboost-model.ts";

export interface LtrScorerConfig {
	/** Master switch; nothing is loaded or scored while false. */
	readonly enabled: boolean;
	/** A model id, or `latest` / undefined for the newest model that passed the training gate. */
	readonly model?: string;
	readonly memoryPath: string;
}

export interface LtrRankOutcome {
	/** The pool in the order the model chose; unscored candidates keep their reranker order after the scored ones. */
	readonly results: RetrievalResult[];
	/** Candidates the model scored. */
	readonly scored: number;
	/** Set when scoring failed: `results` is then the reranker's order, untouched. */
	readonly error?: string;
}

export type LtrScorer =
	| { readonly status: "disabled" }
	/** Enabled, but nothing can be served: no model passed the gate, or the selected one does not exist. */
	| { readonly status: "no-model"; readonly reason: string }
	/** The selected model exists but cannot be loaded or no longer matches the pipeline's features. */
	| { readonly status: "unusable"; readonly modelId: string; readonly reason: string }
	| {
			readonly status: "active";
			readonly modelId: string;
			/** Reorder a reranked pool. Never throws and never drops a candidate. */
			rank(snapshot: RankingRunRecord, reranked: readonly RetrievalResult[]): LtrRankOutcome;
	  };

/**
 * Build the serving scorer from the config. Opt-in twice over: `enabled` must be true and a model must exist.
 * The scorer applies after the baseline reranker and only reorders the candidates that reranker returned, with
 * the features training used ({@link servingRows}); it can never add or remove evidence.
 */
export function createLtrScorer(config: LtrScorerConfig): LtrScorer {
	if (!config.enabled) return { status: "disabled" };
	const directory = ltrDirectory(config.memoryPath);
	const meta = resolveLtrModel(directory, config.model);
	if (meta === undefined) {
		return {
			status: "no-model",
			reason:
				config.model === undefined || config.model === "latest"
					? "no trained model has passed the holdout gate yet; run `autorag ltr train`"
					: `model "${config.model}" does not exist; run \`autorag ltr list\``,
		};
	}
	let model: XgboostModel;
	try {
		model = loadLtrModel(directory, meta.id).model;
	} catch (error) {
		return { status: "unusable", modelId: meta.id, reason: error instanceof Error ? error.message : String(error) };
	}
	return {
		status: "active",
		modelId: meta.id,
		rank(snapshot, reranked) {
			try {
				const scores = new Map<string, number>();
				for (const row of servingRows(snapshot)) scores.set(row.chunkId, model.predict(row.features));
				const scored: { result: RetrievalResult; score: number }[] = [];
				const unscored: RetrievalResult[] = [];
				for (const result of reranked) {
					const score = scores.get(chunkIdOf(result.source, result.content));
					if (score === undefined) unscored.push(result);
					else scored.push({ result, score });
				}
				// Stable: equal scores keep the reranker's order.
				const ordered = scored
					.map((entry, index) => ({ ...entry, index }))
					.sort((a, b) => b.score - a.score || a.index - b.index)
					.map((entry) => entry.result);
				return { results: [...ordered, ...unscored], scored: scored.length };
			} catch (error) {
				return { results: [...reranked], scored: 0, error: error instanceof Error ? error.message : String(error) };
			}
		},
	};
}
