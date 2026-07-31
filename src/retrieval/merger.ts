import type {
	RetrievalDiagnostic,
	RetrievalDiagnosticCode,
	RetrievalMethod,
	RetrievalOptions,
	RetrievalResult,
	RetrievalWithDiagnostics,
} from "./types.ts";

export interface MergeOptions {
	topK: number;
	dedup: boolean;
}

const MAX_AGGREGATION_BONUS = 0.3;
const MAX_SECONDARY_HITS = 3;
const SECONDARY_HIT_WEIGHT = 0.15;

interface MethodResult {
	readonly method: string;
	readonly result: RetrievalResult;
}

export class ResultMerger {
	merge(results: Map<string, RetrievalResult[]>, options: MergeOptions): RetrievalResult[] {
		const { topK, dedup } = options;
		if (results.size === 0) return [];

		// Normalize scores per method using min-max normalization
		const normalized = new Map<string, RetrievalResult[]>();
		for (const [method, methodResults] of results) {
			if (methodResults.length === 0) {
				normalized.set(method, []);
				continue;
			}
			const scores = methodResults.map((r) => r.score);
			const min = Math.min(...scores);
			const max = Math.max(...scores);
			const range = max - min;
			normalized.set(
				method,
				methodResults.map((r) => ({
					...r,
					score: range > 0 ? (r.score - min) / range : 1.0,
				})),
			);
		}

		// Merge all results
		const allResults: MethodResult[] = [];
		for (const [method, methodResults] of normalized) {
			allResults.push(...methodResults.map((result) => ({ method, result })));
		}

		if (dedup) {
			return aggregateBySource(allResults).slice(0, topK);
		}

		return allResults
			.map(({ result }) => result)
			.sort((a, b) => b.score - a.score)
			.slice(0, topK);
	}
}

function aggregateBySource(results: readonly MethodResult[]): RetrievalResult[] {
	const bySource = new Map<string, { readonly order: number; readonly hits: [MethodResult, ...MethodResult[]] }>();
	for (const hit of results) {
		const group = bySource.get(hit.result.source);
		if (group) group.hits.push(hit);
		else bySource.set(hit.result.source, { order: bySource.size, hits: [hit] });
	}
	return Array.from(bySource.values())
		.map(({ order, hits }) => {
			const ranked = hits
				.map((hit, index) => ({ ...hit, index }))
				.sort((a, b) => {
					return b.result.score - a.result.score || a.index - b.index;
				});
			const representative = ranked[0];
			if (!representative || ranked.length === 1) {
				return { order, result: representative?.result ?? hits[0].result };
			}
			const bonus = ranked
				.slice(1, MAX_SECONDARY_HITS + 1)
				.reduce((sum, hit, index) => sum + hit.result.score / (index + 1), 0);
			return {
				order,
				result: {
					...representative.result,
					score: representative.result.score + Math.min(MAX_AGGREGATION_BONUS, bonus * SECONDARY_HIT_WEIGHT),
					metadata: {
						...representative.result.metadata,
						aggregateHitCount: ranked.length,
						aggregateMethods: Array.from(new Set(ranked.map((hit) => hit.method))).sort(),
					},
				},
			};
		})
		.sort((a, b) => b.result.score - a.result.score || a.order - b.order)
		.map(({ result }) => result);
}

export class ParallelRetriever {
	async retrieve(
		methods: RetrievalMethod[],
		query: string,
		options: RetrievalOptions,
	): Promise<Map<string, RetrievalResult[]>> {
		const results = new Map<string, RetrievalResult[]>();
		await Promise.all(
			methods.map(async (method) => {
				const name = method.describe().name;
				try {
					const methodResults = await method.retrieve(query, options);
					results.set(name, methodResults);
				} catch {
					results.set(name, []);
				}
			}),
		);
		return results;
	}

	/**
	 * Like {@link retrieve} but also returns path-opaque diagnostics for methods
	 * that failed. Partial results from healthy methods are preserved; failed
	 * methods yield an empty result set plus a diagnostic. The legacy
	 * {@link retrieve} return shape is intentionally unchanged for compatibility.
	 */
	async retrieveWithDiagnostics(
		methods: RetrievalMethod[],
		query: string,
		options: RetrievalOptions,
	): Promise<RetrievalWithDiagnostics> {
		const results = new Map<string, RetrievalResult[]>();
		const diagnostics: RetrievalDiagnostic[] = [];
		await Promise.all(
			methods.map(async (method) => {
				const name = method.describe().name;
				try {
					results.set(name, await method.retrieve(query, options));
				} catch {
					results.set(name, []);
					diagnostics.push({
						code: methodFailureCode(name),
						severity: "warning",
						message: `Retrieval method "${name}" failed and was skipped; partial results from other methods were used.`,
						source: name,
					});
				}
			}),
		);
		return { results, diagnostics };
	}
}

function methodFailureCode(name: string): RetrievalDiagnosticCode {
	if (name === "minsync") return "minsync-unavailable";
	if (name === "bm25") return "bm25-unavailable";
	return "retrieval-method-failed";
}
