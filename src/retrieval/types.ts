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
}

export interface RetrievalOptions {
	topK?: number;
	scope?: string;
	filters?: Record<string, unknown>;
	signal?: AbortSignal;
}

export interface RetrievalMethod {
	describe(): RetrievalMethodDescriptor;
	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]>;
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
