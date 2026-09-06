/**
 * RetrievalEngine — standalone typed seam for model-free retrieval.
 *
 * Wraps the existing retrieval pipeline (registry, parallel retriever,
 * datasource result filter, and merger) into a single public entry point.
 * Exposes a deterministic, self-contained retrieval API with diagnostics.
 *
 * Security invariants (delegated to {@link DatasourceAccessContext} and
 * {@link DatasourceResultFilter}):
 *  - Access is default-deny: when no trusted allow-tags are configured,
 *    every datasource method result is dropped.
 *  - Deny is always an explicit empty result array, never `undefined`.
 *  - Model/tool arguments (`allowedTags`, `allowedScopes`) cannot widen the
 *    trusted access context constructed during engine creation.
 */

import { DatasourceAccessContext, type DatasourceAccessContextOptions } from "../datasource/access-context.ts";
import { DatasourceResultFilter } from "../datasource/result-filter.ts";
import { ParallelRetriever, ResultMerger } from "./merger.ts";
import { RetrievalMethodRegistry } from "./registry.ts";
import type { RetrievalDiagnostic, RetrievalMethod, RetrievalOptions, RetrievalResult } from "./types.ts";

/** Options for constructing a standalone {@link RetrievalEngine}. */
export interface RetrievalEngineOptions {
	/**
	 * Datasource access configuration. Default-deny when omitted.
	 * @default { allowedTags: undefined, allowedScopes: undefined }
	 */
	readonly datasourceAccess?: DatasourceAccessContextOptions;
	/** Default `topK` when the caller omits it. @default 20 */
	readonly defaultTopK?: number;
	/** Default deduplication flag. @default true */
	readonly defaultDedup?: boolean;
	/**
	 * Optional predicate that returns `true` when the MinSync binary is missing.
	 * When set, the engine checks every method named `"minsync"`: if the binary
	 * is unavailable the method returns empty results without throwing, the engine
	 * emits a `minsync-unavailable` diagnostic with full method-agnostic path
	 * preservation — matching the existing {@link AutoRAGAgent} behavior.
	 */
	readonly isMinSyncBinaryMissing?: () => boolean;
}

/**
 * Model-free retrieval pipeline.
 *
 * The engine composes the standard AutoRAG retrieval chain:
 * ```
 * methods = registry.list()
 * byMethod = retriever.retrieveWithDiagnostics(methods, query, options)
 * filtered = filter.filter(byMethod, methods, ctx, options.scope)
 * merged = merger.merge(filtered, { topK, dedup: true })
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
	private readonly filter: DatasourceResultFilter;
	private readonly merger: ResultMerger;
	private readonly accessContext: DatasourceAccessContext;
	private readonly defaultTopK: number;
	private readonly defaultDedup: boolean;
	private readonly isMinSyncBinaryMissing: (() => boolean) | undefined;

	constructor(options: RetrievalEngineOptions = {}) {
		this.registry = new RetrievalMethodRegistry();
		this.retriever = new ParallelRetriever();
		this.filter = new DatasourceResultFilter();
		this.merger = new ResultMerger();
		this.accessContext = new DatasourceAccessContext(options.datasourceAccess);
		this.defaultTopK = options.defaultTopK ?? 20;
		this.defaultDedup = options.defaultDedup ?? true;
		this.isMinSyncBinaryMissing = options.isMinSyncBinaryMissing;
	}

	/** The underlying method registry. Intentionally public for tooling. */
	getMethodRegistry(): RetrievalMethodRegistry {
		return this.registry;
	}

	/** The underlying datasource access context. Read-only for inspection. */
	getAccessContext(): DatasourceAccessContext {
		return this.accessContext;
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
	): Promise<{ results: RetrievalResult[]; diagnostics: RetrievalDiagnostic[] }> {
		const topK = options.topK ?? this.defaultTopK;
		const methods = this.registry.list();
		const { results: byMethod, diagnostics } = await this.retriever.retrieveWithDiagnostics(methods, query, options);
		const ctx = this.accessContextFor(options);
		const filtered = this.filter.filter(byMethod, methods, ctx, options.scope, options.allowedScopes);
		const allDiagnostics = this.appendMinSyncUnavailableDiagnostic(methods, filtered, diagnostics);
		return {
			results: this.merger.merge(filtered, { topK, dedup: this.defaultDedup }),
			diagnostics: allDiagnostics,
		};
	}

	/**
	 * Run retrieval and return results keyed per method (unmerged) with
	 * diagnostics. Unlike {@link retrieve}, this does NOT merge or sort —
	 * callers get the raw per-method view after datasource filtering.
	 */
	async retrieveByMethod(
		query: string,
		options: RetrievalOptions = {},
	): Promise<{ byMethod: Map<string, RetrievalResult[]>; diagnostics: RetrievalDiagnostic[] }> {
		const methods = this.registry.list();
		const { results: byMethod, diagnostics } = await this.retriever.retrieveWithDiagnostics(methods, query, options);
		const ctx = this.accessContextFor(options);
		const filtered = this.filter.filter(byMethod, methods, ctx, options.scope, options.allowedScopes);
		const allDiagnostics = this.appendMinSyncUnavailableDiagnostic(methods, filtered, diagnostics);
		return { byMethod: filtered, diagnostics: allDiagnostics };
	}

	/**
	 * Post-check for MinSync binary-missing diagnostic.
	 *
	 * When a registered method named `"minsync"` fails to throw (the method returns
	 * `[]` because the binary is missing), {@link ParallelRetriever} cannot record
	 * a diagnostic. This method checks whether the minsync binary is unavailable
	 * and emits a `minsync-unavailable` diagnostic when every minsync method's
	 * result set is empty, exactly matching the existing {@link AutoRAGAgent}
	 * behavior (see `agent.src/agent/agent.ts` `retrieveWithDiagnostics`).
	 */
	private appendMinSyncUnavailableDiagnostic(
		methods: readonly RetrievalMethod[],
		filtered: Map<string, RetrievalResult[]>,
		diagnostics: RetrievalDiagnostic[],
	): RetrievalDiagnostic[] {
		if (this.isMinSyncBinaryMissing === undefined) return diagnostics;
		if (!this.isMinSyncBinaryMissing()) return diagnostics;
		if (diagnostics.some((d) => d.code === "minsync-unavailable")) return diagnostics;

		// Did any minsync method return empty results (binary missing, no throw)?
		const hasEmptyMinsync = methods.some((m) => {
			const name = m.describe().name;
			if (name !== "minsync") return false;
			const results = filtered.get(name);
			return results === undefined || results.length === 0;
		});
		if (!hasEmptyMinsync) return diagnostics;

		return [
			...diagnostics,
			{
				code: "minsync-unavailable" as const,
				severity: "warning" as const,
				message: "MinSync semantic search is unavailable; results rely on other retrieval paths.",
				source: "minsync" as const,
			},
		];
	}

	/**
	 * Build a datasource access context scoped to the caller's options.
	 * Model/tool arguments (allowedTags, allowedScopes) from options can only
	 * further restrict the trusted context, never widen it.
	 */
	private accessContextFor(options: RetrievalOptions): DatasourceAccessContext {
		// The engine's base trusted context is the fixed one. Any user-supplied
		// allowedTags/allowedScopes in options are intersected by narrowing via
		// the filter's userScope parameter, not by replacing the base context.
		// For the filter-scope path (userScope), we pass options.scope directly.
		// If options provides explicit allowedTags/allowedScopes, they narrow
		// the base context.
		const baseTags = this.accessContext.allowedTags;
		const baseScopes = this.accessContext.allowedScopes;
		const userTags = options.allowedTags;
		// Intersect: if both sides have tags, only common tags survive.
		const allowedTags =
			baseTags.length === 0 ? [] : userTags !== undefined ? baseTags.filter((t) => userTags.includes(t)) : baseTags;
		return new DatasourceAccessContext({ allowedTags, allowedScopes: baseScopes });
	}
}
