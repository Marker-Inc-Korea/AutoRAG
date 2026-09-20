export interface RetrievalResult {
	id: string;
	content: string;
	source: string;
	score: number;
	metadata: Record<string, unknown>;
}

export interface RetrievalMethodDescriptor {
	name: string;
	type: "posix" | "vector" | "bm25" | "hybrid" | "visual";
	description: string;
	status: "active" | "stub";
	capabilities: string[];
	datasourceId?: string;
	tags?: readonly string[];
}

export interface RetrievalOptions {
	topK?: number;
	scope?: string;
	filters?: Record<string, unknown>;
	allowedTags?: readonly string[];
	allowedScopes?: readonly string[];
	signal?: AbortSignal;
}

export interface RetrievalMethod {
	describe(): RetrievalMethodDescriptor;
	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]>;
}

export type RetrievalDiagnosticCode = "retrieval-method-failed" | "minsync-unavailable";

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

export interface RetrievalWithDiagnostics {
	results: Map<string, RetrievalResult[]>;
	diagnostics: RetrievalDiagnostic[];
	/** Surfaces that were not searched for this query. Empty when every method ran. */
	unsearched: RetrievalUnsearchedSurface[];
}

export interface NumberedResult {
	index: number;
	source: string;
	content: string;
	method: string;
}

export interface EvidenceReference {
	method: string;
	source: string;
	excerpt?: string;
	content?: string;
	retrievalResultId?: string;
	chunkIndex?: number;
	lineNumber?: number;
	stableEvidenceId: string;
}

export interface CuratedResult {
	index: number;
	content: string;
	source: string;
	method: string;
	evidenceRefs?: readonly EvidenceReference[];
}
