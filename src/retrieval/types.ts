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

export type RetrievalDiagnosticCode =
	| "retrieval-method-failed"
	| "minsync-unavailable"
	| "bm25-unavailable"
	| "bm25-degraded-fallback";

/** Path-opaque diagnostic emitted by the multi-method retrieval pipeline. */
export interface RetrievalDiagnostic {
	code: RetrievalDiagnosticCode;
	severity: "info" | "warning" | "error";
	message: string;
	/** Component/method label — never a real filesystem path. */
	source?: string;
}

export interface RetrievalWithDiagnostics {
	results: Map<string, RetrievalResult[]>;
	diagnostics: RetrievalDiagnostic[];
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
