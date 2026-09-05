export type { MergeOptions } from "./merger.ts";
export { ParallelRetriever, ResultMerger } from "./merger.ts";
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
