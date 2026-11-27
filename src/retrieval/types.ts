export interface RetrievalResult {
	id: string;
	content: string;
	source: string;
	score: number;
	metadata: Record<string, unknown>;
}

export interface RetrievalMethodDescriptor {
	name: string;
	type: "posix" | "vector" | "bm25" | "sparse" | "hybrid" | "visual" | "remote";
	description: string;
	status: "active" | "stub";
	capabilities: string[];
	datasourceId?: string;
	tags?: readonly string[];
}

/**
 * A fallback chain answered with a fallback because its primary method failed. `reason` is the primary's
 * error verbatim, so the degraded surface can be reported like any other skipped method.
 */
export interface RetrievalFallbackEvent {
	/** Name of the primary method that failed. */
	method: string;
	/** Name of the fallback that answered instead. */
	fallback: string;
	reason: string;
}

/** One hit dropped because the file it was indexed from no longer exists on disk. */
export interface RetrievalMissingSource {
	/** Virtual path the index knows the document by. */
	virtualPath: string;
	/** The original source path that could not be resolved. */
	sourcePath: string;
	/** Filesystem error code (`ENOENT` or `ENOTDIR`). */
	code: string;
}

export interface RetrievalSourceMissingEvent {
	/** Registered name of the method that skipped the hits. */
	method: string;
	missing: readonly RetrievalMissingSource[];
}

export interface RetrievalOptions {
	topK?: number;
	scope?: string;
	/**
	 * Time window `[since, until)` in epoch ms. Only narrows: it composes with `scope` (both must match) and never
	 * widens what a datasource's trusted config allows. The pipeline applies it to every method's results (see
	 * `time-window.ts`); a method may additionally push it into its own store to widen recall.
	 */
	since?: number;
	until?: number;
	/**
	 * Per-search filters a datasource declares through {@link DatasourceSkill.searchFilters}, by filter name. Like `scope`,
	 * they only narrow: a method that does not know a filter ignores it, and none can reach past the operator's config.
	 */
	filters?: Readonly<Record<string, readonly string[]>>;
	/**
	 * Explicit Turn-1 route override chosen by the caller (e.g. a user clicking an
	 * explicit Find / Summarize / Organize / Automate mode chip in the UI).
	 * When set, Turn 1 skips Jev's initial query-route classification and executes
	 * the requested branch directly (`local` still runs decomposition + datasource
	 * selection; `organize` loads the organize skill; `config` loads the setup
	 * skill when self-config is enabled; `extra` runs a direct single tool turn).
	 */
	routeOverride?: "local" | "organize" | "extra" | "config";
	signal?: AbortSignal;
	/**
	 * Owned by the retrieval pipeline, not by callers: invoked when a method answered through a fallback.
	 * Methods that wrap another method must not forward it.
	 */
	onFallback?: (event: RetrievalFallbackEvent) => void;
	/**
	 * Owned by the retrieval pipeline, not by callers: invoked once per query when a method skipped hits whose
	 * source file no longer exists (a stale index entry). The remaining hits are still returned, so this is a
	 * warning and not a failure of the surface. Methods that wrap another method must not forward it.
	 */
	onSourceMissing?: (event: RetrievalSourceMissingEvent) => void;
}

export interface RetrievalMethod {
	describe(): RetrievalMethodDescriptor;
	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]>;
}

export type RetrievalDiagnosticCode =
	| "retrieval-method-failed"
	| "retrieval-source-missing"
	| "rerank-failed"
	| "time-window-untimed"
	| "time-window-ceiling";

/**
 * A retrieval surface (local MinSync files or one datasource) that was not
 * searched for this query. Reported next to partial results so a caller that
 * reads only the result list can still tell the answer is incomplete.
 */
export interface RetrievalUnsearchedSurface {
	/** Surface label: "minsync" for local files, or the datasource id. */
	surface: string;
	/** Registered retrieval method names on this surface that did not run. */
	methods: string[];
	/**
	 * Why the surface did not run: the underlying error verbatim (CLI stderr,
	 * exit status, paths included). Never classified or suppressed — the operator
	 * debugging their own machine needs the real text.
	 */
	reason: string;
}

/** Diagnostic emitted by the multi-method retrieval pipeline. */
export interface RetrievalDiagnostic {
	code: RetrievalDiagnosticCode;
	severity: "info" | "warning" | "error";
	message: string;
	/** Component/method label. */
	source?: string;
	/** The underlying failure verbatim when the diagnostic reports a method that did not run. */
	reason?: string;
}

/**
 * What a time window did to the candidates, so an empty answer can say why: nothing existed in the range, results had
 * no timestamp, or the over-fetch ceiling cut the search short.
 */
export interface RetrievalWindowReport {
	readonly since?: number;
	readonly until?: number;
	/** Candidates the methods returned before the window was applied. */
	readonly examined: number;
	/** Candidates inside the window. */
	readonly matched: number;
	/** Candidates dropped because the method gave them neither `publishedAt` nor a source mtime. */
	readonly untimed: number;
	/** Some method returned as many candidates as it was asked for, so older or newer matches may be unexamined. */
	readonly ceilingHit: boolean;
	/** The depth each method was asked for. */
	readonly fetchDepth: number;
}

export interface RetrievalWithDiagnostics {
	results: Map<string, RetrievalResult[]>;
	diagnostics: RetrievalDiagnostic[];
	/** Present only when the query carried a time window. */
	window?: RetrievalWindowReport;
	/** Surfaces that were not searched for this query. Empty when every method ran. */
	unsearched: RetrievalUnsearchedSurface[];
}
