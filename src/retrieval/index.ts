export { RetrievalEngine, type RetrievalEngineOptions } from "./engine.ts";
export type { MergeOptions } from "./merger.ts";
export { ParallelRetriever, ResultMerger } from "./merger.ts";
export { RetrievalMethodRegistry } from "./registry.ts";
export {
	matchesVirtualPathScope,
	normalizeVirtualPath,
	normalizeVirtualPathScope,
	virtualPathScopeToRegExp,
} from "./scope.ts";
export {
	classifyRetrievalSkip,
	groupUnsearchedSurfaces,
	MINSYNC_SURFACE,
	type RetrievalSkip,
	retrievalSkipAction,
	retrievalSkipMessage,
	retrievalSurfaceFor,
} from "./skip.ts";
export type {
	CuratedResult,
	NumberedResult,
	RetrievalDiagnostic,
	RetrievalDiagnosticCode,
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
	RetrievalSkipAction,
	RetrievalSkipReason,
	RetrievalUnsearchedSurface,
	RetrievalWithDiagnostics,
} from "./types.ts";
