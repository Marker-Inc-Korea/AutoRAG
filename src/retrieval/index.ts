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
	describeRetrievalError,
	groupUnsearchedSurfaces,
	MINSYNC_SURFACE,
	type RetrievalSkip,
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
	RetrievalUnsearchedSurface,
	RetrievalWithDiagnostics,
} from "./types.ts";
