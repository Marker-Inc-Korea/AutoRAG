/**
 * RetrievalEngine — standalone typed seam for model-free retrieval.
 *
 * Wraps the existing retrieval pipeline (registry, parallel retriever,
 * datasource scope filter, and merger) into a single public entry point.
 * Exposes a deterministic, self-contained retrieval API with diagnostics.
 *
 * Every configured/connected datasource is searchable without permission
 * setup. The ordinary query `scope` narrows scope-capable datasource results
 * after retrieval; local (non-datasource) methods and datasources without
 * source-scope support pass through untouched.
 */

import { filterDatasourceScope } from "../datasource/scope.ts";
import { ParallelRetriever, ResultMerger } from "./merger.ts";
import { RetrievalMethodRegistry } from "./registry.ts";
import { DEFAULT_RERANK_TOP_N, type Reranker } from "./rerank.ts";
import { derivedDatasourceIds, type RetrievalSelection, resolveSelectedMethods } from "./selection.ts";
import type {
	RetrievalDiagnostic,
	RetrievalMethod,
	RetrievalOptions,
	RetrievalResult,
	RetrievalUnsearchedSurface,
} from "./types.ts";

/**
 * Safety ceiling on merged evidence, not a relevance filter. The merger keeps
 * every distinct chunk, so a run's real size is whatever the registered methods
 * returned; this only guards against an unbounded registry.
 */
const DEFAULT_MERGED_EVIDENCE_CEILING = 500;

/** Options for constructing a standalone {@link RetrievalEngine}. */
export interface RetrievalEngineOptions {
	/**
	 * Default `topK` when the caller omits it. This is a safety ceiling on
	 * merged evidence, not a relevance cut: the merger returns every distinct
	 * chunk, so the effective size is whatever the registered methods returned.
	 * @default 500
	 */
	readonly defaultTopK?: number;
	/** Default deduplication flag. @default true */
	readonly defaultDedup?: boolean;
	/**
	 * Configured datasource ids from the skill catalog, including datasources
	 * that expose no retrieval methods. {@link retrieveSelected} uses this to
	 * tell an unknown datasource id from a known one. When omitted, the engine
	 * derives ids from its registered methods; method-less datasources are then
	 * treated as unknown.
	 */
	readonly datasourceIds?: () => readonly string[];
	/**
	 * Optional post-merge reranker. When set, merged results are reordered by
	 * the reranker after dedup and score normalization. A reranker failure is
	 * reported as a `rerank-failed` diagnostic and the unranked order is kept —
	 * the engine never silently drops evidence because a reranker was down.
	 */
	readonly reranker?: Reranker;
	/**
	 * Number of merged results kept after reranking when the caller omits `topK`.
	 * Only applies when {@link reranker} is set. @default DEFAULT_RERANK_TOP_N
	 */
	readonly rerankTopN?: number;
}

/**
 * Model-free retrieval pipeline.
 *
 * The engine composes the standard AutoRAG retrieval chain:
 * ```
 * methods = registry.list()
 * byMethod = retriever.retrieveWithDiagnostics(methods, query, options)
 * filtered = filterDatasourceScope(byMethod, methods, options.scope)
 * merged = merger.merge(filtered, { topK, dedup: true })  // distinct chunks, pure duplicates dropped
 * → { results: merged, diagnostics }
 * ```
 *
 * Exposes two paths:
 *  1. `retrieve(query, options?)` — merged results + diagnostics (the standard
 *     pipeline).
 *  2. `retrieveByMethod(query, options?)` — per-method results (unmerged) with
 *     diagnostics, for callers that need the raw per-method view.
 *
 * The engine does NOT handle refresh lifecycle, MinSync sync, Jikji, or
 * agent-level concerns. It is purely the retrieval composition seam.
 */
export class RetrievalEngine {
	private readonly registry: RetrievalMethodRegistry;
	private readonly retriever: ParallelRetriever;
	private readonly merger: ResultMerger;
	private readonly defaultTopK: number;
	private readonly defaultDedup: boolean;
	private readonly datasourceIdsProvider: (() => readonly string[]) | undefined;
	private readonly reranker: Reranker | undefined;
	private readonly rerankTopN: number;

	constructor(options: RetrievalEngineOptions = {}) {
		this.registry = new RetrievalMethodRegistry();
		this.retriever = new ParallelRetriever();
		this.merger = new ResultMerger();
		this.defaultTopK = options.defaultTopK ?? DEFAULT_MERGED_EVIDENCE_CEILING;
		this.defaultDedup = options.defaultDedup ?? true;
		this.datasourceIdsProvider = options.datasourceIds;
		this.reranker = options.reranker;
		this.rerankTopN = options.rerankTopN ?? DEFAULT_RERANK_TOP_N;
	}

	/** The underlying method registry. Intentionally public for tooling. */
	getMethodRegistry(): RetrievalMethodRegistry {
		return this.registry;
	}

	/**
	 * Register a retrieval method. Overrides the registered-once check from
	 * {@link RetrievalMethodRegistry} per caller requirement.
	 *
	 * @throws {Error} if a method with the same name is already registered.
	 */
	register(method: RetrievalMethod): void {
		this.registry.register(method);
	}

	/**
	 * Register multiple methods in one call. Stop-on-first-error semantics:
	 * registration fails atomically if any method conflicts with an already-
	 * registered name; earlier methods in the batch are not retained.
	 *
	 * @throws {Error} if any method name duplicates an existing or in-batch registration.
	 */
	registerMany(methods: readonly RetrievalMethod[]): void {
		// Check all conflicts before mutating.
		const names = new Set<string>();
		for (const method of methods) {
			const name = method.describe().name;
			if (this.registry.get(name) !== undefined || names.has(name)) {
				throw new Error(`Retrieval method "${name}" is already registered`);
			}
			names.add(name);
		}
		for (const method of methods) {
			this.registry.register(method);
		}
	}

	/**
	 * Run the full retrieval pipeline — parallel retrieve, datasource filter,
	 * merge — and return merged results with diagnostics.
	 */
	async retrieve(
		query: string,
		options: RetrievalOptions = {},
	): Promise<{
		results: RetrievalResult[];
		diagnostics: RetrievalDiagnostic[];
		unsearched: RetrievalUnsearchedSurface[];
	}> {
		const topK = options.topK ?? this.defaultTopK;
		const methods = this.registry.list();
		const {
			results: byMethod,
			diagnostics,
			unsearched,
		} = await this.retriever.retrieveWithDiagnostics(methods, query, options);
		const filtered = filterDatasourceScope(byMethod, methods, options.scope);
		const merged = this.merger.merge(filtered, { topK, dedup: this.defaultDedup });
		const results = await this.applyRerank(query, merged, diagnostics, options);
		return { results, diagnostics, unsearched };
	}

	/**
	 * Reorder merged results with the configured reranker. On failure the
	 * unranked order is returned and a `rerank-failed` diagnostic is appended.
	 */
	private async applyRerank(
		query: string,
		results: RetrievalResult[],
		diagnostics: RetrievalDiagnostic[],
		options: RetrievalOptions,
	): Promise<RetrievalResult[]> {
		if (this.reranker === undefined || results.length === 0) return results;
		try {
			const reranked = await this.reranker.rerank(query, results, {
				topN: options.topK ?? this.rerankTopN,
				signal: options.signal,
			});
			return reranked;
		} catch (error) {
			const reason = error instanceof Error ? error.message : String(error);
			const descriptor = this.reranker.describe();
			diagnostics.push({
				code: "rerank-failed",
				severity: "warning",
				message: `Reranker "${descriptor.name}" failed and was skipped; merged order preserved: ${reason}`,
				source: descriptor.name,
				reason,
			});
			return results;
		}
	}

	/**
	 * Run retrieval and return results keyed per method (unmerged) with
	 * diagnostics. Unlike {@link retrieve}, this does NOT merge or sort —
	 * callers get the raw per-method view after datasource filtering.
	 */
	async retrieveByMethod(
		query: string,
		options: RetrievalOptions = {},
	): Promise<{
		byMethod: Map<string, RetrievalResult[]>;
		diagnostics: RetrievalDiagnostic[];
		unsearched: RetrievalUnsearchedSurface[];
	}> {
		const methods = this.registry.list();
		const {
			results: byMethod,
			diagnostics,
			unsearched,
		} = await this.retriever.retrieveWithDiagnostics(methods, query, options);
		const filtered = filterDatasourceScope(byMethod, methods, options.scope);
		return { byMethod: filtered, diagnostics, unsearched };
	}

	/**
	 * Run a selected subset of the registered methods — parallel retrieve,
	 * scope filter, merge — and return merged results with diagnostics.
	 *
	 * Unlike {@link retrieve}, selection happens BEFORE any backend is invoked:
	 * the eligible method list is reduced first, so an unselected datasource
	 * backend is never spawned. Unknown method or datasource selections throw
	 * {@link RetrievalSelectionError}. The ordinary query `scope` still narrows
	 * results after retrieval.
	 */
	async retrieveSelected(
		query: string,
		selection: RetrievalSelection = {},
		options: RetrievalOptions = {},
	): Promise<{
		results: RetrievalResult[];
		diagnostics: RetrievalDiagnostic[];
		unsearched: RetrievalUnsearchedSurface[];
	}> {
		const topK = options.topK ?? this.defaultTopK;
		const datasourceIds = this.datasourceIdsProvider?.() ?? derivedDatasourceIds(this.registry.list());
		const methods = resolveSelectedMethods(this.registry.list(), selection, datasourceIds);
		const {
			results: byMethod,
			diagnostics,
			unsearched,
		} = await this.retriever.retrieveWithDiagnostics(methods, query, options);
		const filtered = filterDatasourceScope(byMethod, methods, options.scope);
		return {
			results: this.merger.merge(filtered, { topK, dedup: this.defaultDedup }),
			diagnostics,
			unsearched,
		};
	}
}
