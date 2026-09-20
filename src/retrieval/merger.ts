import { describeRetrievalError, groupUnsearchedSurfaces, type RetrievalSkip, retrievalSurfaceFor } from "./skip.ts";
import type {
	RetrievalDiagnostic,
	RetrievalDiagnosticCode,
	RetrievalMethod,
	RetrievalOptions,
	RetrievalResult,
	RetrievalWithDiagnostics,
} from "./types.ts";

export interface MergeOptions {
	/**
	 * Upper bound on returned evidence. This is a safety ceiling, not a
	 * relevance filter: the merger is expected to return every distinct chunk
	 * it was given. Callers that want the full pool pass a large value.
	 */
	topK: number;
	/** Retained for API compatibility; pure-duplicate removal is always enabled. */
	dedup: boolean;
}

const SAME_SOURCE_CORROBORATION_BONUS = 0.15;
const MAX_CORROBORATION_BONUS = 0.3;

interface MethodResult {
	readonly method: string;
	readonly result: RetrievalResult;
}

/**
 * Cross-method evidence collator.
 *
 * `search_all_documents` is the *exhaustive* retrieval surface: it fans out to
 * every configured method and the librarian is expected to judge the evidence
 * itself. Collapsing that fan-out to one chunk per source threw away most of
 * what was retrieved — a document whose several passages all matched surfaced
 * only its single best chunk, and a datasource that returned twenty relevant
 * messages could be reduced to one. Modern context windows hold the full pool
 * (a live 16-method run over this repository's own corpus measured ~26k tokens
 * on average, ~53k at its worst), so withholding it bought nothing.
 *
 * This merger therefore preserves every distinct chunk and removes only *pure*
 * duplicates: the same evidence identity (`source` + `id`) seen more than once.
 * Ranking still normalizes each method's scores into a comparable 0..1 band and
 * gives a bounded bonus to sources corroborated by several chunks or methods,
 * so the strongest evidence stays at the top of a complete list.
 */
export class ResultMerger {
	merge(results: Map<string, RetrievalResult[]>, options: MergeOptions): RetrievalResult[] {
		const { topK } = options;
		if (results.size === 0) return [];

		// Normalize scores per method using min-max normalization so a BM25 score
		// and a cosine score become comparable ranks rather than raw magnitudes.
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

		const allResults: MethodResult[] = [];
		for (const [method, methodResults] of normalized) {
			allResults.push(...methodResults.map((result) => ({ method, result })));
		}

		return collateDistinctEvidence(allResults).slice(0, topK);
	}
}

/**
 * Drop pure duplicates, keep every other chunk, and rank the survivors.
 *
 * A pure duplicate is the same `source` + `id` pair: the identical chunk
 * reached us twice (the same method repeating itself, or two methods over a
 * shared index). The first occurrence wins and later ones only record that the
 * evidence was corroborated. Two different chunks of one document are *not*
 * duplicates and both survive.
 */
function collateDistinctEvidence(results: readonly MethodResult[]): RetrievalResult[] {
	interface Entry {
		readonly order: number;
		readonly method: string;
		readonly result: RetrievalResult;
		methods: Set<string>;
		duplicateHits: number;
	}

	const byEvidence = new Map<string, Entry>();
	for (const hit of results) {
		const evidenceKey = `${hit.result.source}\0${hit.result.id}`;
		const existing = byEvidence.get(evidenceKey);
		if (existing === undefined) {
			byEvidence.set(evidenceKey, {
				order: byEvidence.size,
				method: hit.method,
				result: hit.result,
				methods: new Set([hit.method]),
				duplicateHits: 1,
			});
			continue;
		}
		// Same chunk again: keep the first, remember it was corroborated.
		existing.methods.add(hit.method);
		existing.duplicateHits += 1;
	}

	const entries = [...byEvidence.values()];

	// Count distinct chunks per source so a document several of whose passages
	// matched ranks above an equally-scored one-off hit — without discarding any
	// of those passages.
	const chunksPerSource = new Map<string, number>();
	const methodsPerSource = new Map<string, Set<string>>();
	for (const entry of entries) {
		const source = entry.result.source;
		chunksPerSource.set(source, (chunksPerSource.get(source) ?? 0) + 1);
		const methods = methodsPerSource.get(source);
		if (methods === undefined) methodsPerSource.set(source, new Set(entry.methods));
		else for (const method of entry.methods) methods.add(method);
	}

	const ranked = entries
		.map((entry) => {
			const source = entry.result.source;
			const sourceChunks = chunksPerSource.get(source) ?? 1;
			const sourceMethods = methodsPerSource.get(source) ?? entry.methods;
			// Corroboration: how many *other* distinct chunks and methods back this
			// source. Bounded so it reorders ties without overpowering relevance.
			const corroboration = sourceChunks - 1 + (sourceMethods.size - 1);
			const bonus =
				corroboration > 0
					? Math.min(MAX_CORROBORATION_BONUS, Math.log1p(corroboration) * SAME_SOURCE_CORROBORATION_BONUS)
					: 0;
			const metadata: Record<string, unknown> = {
				...entry.result.metadata,
				retrievalMethods: [...sourceMethods].sort(),
				sourceChunkCount: sourceChunks,
			};
			if (entry.duplicateHits > 1) metadata.duplicateHitCount = entry.duplicateHits;
			return {
				order: entry.order,
				result: { ...entry.result, score: entry.result.score + bonus, metadata },
			};
		})
		.sort((a, b) => b.result.score - a.result.score || a.order - b.order);

	const scores = ranked.map(({ result }) => result.score);
	const min = Math.min(...scores);
	const max = Math.max(...scores);
	const range = max - min;
	return ranked.map(({ result }) => ({
		...result,
		score: range > 0 ? (result.score - min) / range : 1,
	}));
}

export class ParallelRetriever {
	async retrieve(
		methods: RetrievalMethod[],
		query: string,
		options: RetrievalOptions,
	): Promise<Map<string, RetrievalResult[]>> {
		const results = new Map<string, RetrievalResult[]>();
		for (const method of methods) results.set(method.describe().name, []);
		await Promise.all(
			methods.map(async (method) => {
				const name = method.describe().name;
				try {
					results.set(name, await method.retrieve(query, options));
				} catch {
					results.set(name, []);
				}
			}),
		);
		return results;
	}

	/**
	 * Like {@link retrieve} but also returns diagnostics for methods that failed,
	 * plus the retrieval surfaces those methods belong to. Partial results from
	 * healthy methods are preserved; a failed method yields an empty result set,
	 * a diagnostic quoting the underlying error, and an entry in `unsearched`
	 * carrying that error verbatim. The legacy {@link retrieve} return shape is
	 * intentionally unchanged for compatibility.
	 */
	async retrieveWithDiagnostics(
		methods: RetrievalMethod[],
		query: string,
		options: RetrievalOptions,
	): Promise<RetrievalWithDiagnostics> {
		const results = new Map<string, RetrievalResult[]>();
		for (const method of methods) results.set(method.describe().name, []);
		const diagnostics: RetrievalDiagnostic[] = [];
		const skips: RetrievalSkip[] = [];
		await Promise.all(
			methods.map(async (method) => {
				const descriptor = method.describe();
				const name = descriptor.name;
				try {
					results.set(name, await method.retrieve(query, options));
				} catch (error) {
					results.set(name, []);
					const reason = describeRetrievalError(error);
					skips.push({ method: name, surface: retrievalSurfaceFor(descriptor), reason });
					diagnostics.push({
						code: methodFailureCode(name),
						severity: "warning",
						message: `Retrieval method "${name}" failed and was skipped; partial results from other methods were used: ${reason}`,
						source: name,
						reason,
					});
				}
			}),
		);
		return { results, diagnostics, unsearched: groupUnsearchedSurfaces(skips) };
	}
}

function methodFailureCode(name: string): RetrievalDiagnosticCode {
	if (name === "minsync") return "minsync-unavailable";
	return "retrieval-method-failed";
}
