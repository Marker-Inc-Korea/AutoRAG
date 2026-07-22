export type { MergeOptions } from "./merger.ts";
export { ParallelRetriever, ResultMerger } from "./merger.ts";
export type {
	BM25Engine,
	BM25FallbackMode,
	BM25MethodOptions,
	BM25ReadinessState,
	BM25Status,
	BM25SyncResult,
} from "./methods/bm25.ts";
export { BM25Method, BM25UnavailableError } from "./methods/bm25.ts";
export { RetrievalMethodRegistry } from "./registry.ts";
export {
	matchesVirtualPathScope,
	normalizeVirtualPath,
	normalizeVirtualPathScope,
	virtualPathScopeToRegExp,
} from "./scope.ts";
export type {
	CuratedResult,
	NumberedResult,
	RetrievalDiagnostic,
	RetrievalDiagnosticCode,
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
	RetrievalWithDiagnostics,
} from "./types.ts";
