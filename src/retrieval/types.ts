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
	signal?: AbortSignal;
	/**
	 * Owned by the retrieval pipeline, not by callers: invoked when a method answered through a fallback.
	 * Methods that wrap another method must not forward it.
	 */
	onFallback?: (event: RetrievalFallbackEvent) => void;
}

export interface RetrievalMethod {
	describe(): RetrievalMethodDescriptor;
	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]>;
}

export type RetrievalDiagnosticCode =
	| "retrieval-method-failed"
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
