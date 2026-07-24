export type BenchmarkProfile = "smoke" | "full";

export type BenchmarkMethod = "bm25" | "minsync" | "hybrid";

export interface CorpusDocument {
	documentId: string;
	title: string;
	text: string;
}

export interface BenchmarkQuery {
	queryId: string;
	text: string;
}

export interface Qrel {
	queryId: string;
	documentId: string;
	relevance: number;
}

export interface RankedHit {
	documentId: string;
	score: number;
	rank: number;
}

export interface QueryRunRecord {
	schemaVersion: 1;
	method: BenchmarkMethod;
	queryId: string;
	latencyMs: number;
	hits: readonly RankedHit[];
	errorCode?: "retrieval-failed";
}
