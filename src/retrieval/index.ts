export { bm25Scores, tokenize } from "./bm25.ts";
export { RetrievalEngine, type RetrievalEngineOptions } from "./engine.ts";
export { FallbackRetrievalMethod } from "./fallback.ts";
export type { MergeOptions } from "./merger.ts";
export { ParallelRetriever, ResultMerger } from "./merger.ts";
export { RetrievalMethodRegistry } from "./registry.ts";
export {
	type CreateRerankerOptions,
	createReranker,
	DEFAULT_DAZZI_RERANK_MODEL,
	DEFAULT_RERANK_API_KEY_ENV,
	DEFAULT_RERANK_MODEL,
	DEFAULT_RERANK_PROVIDER,
	DEFAULT_RERANK_TOP_N,
	defaultRerankApiKeyEnv,
	defaultRerankModel,
	OpenRouterReranker,
	type OpenRouterRerankerOptions,
	type Reranker,
	type RerankerDescriptor,
	type RerankOptions,
	SUPPORTED_RERANK_PROVIDERS,
} from "./rerank.ts";
export {
	matchesVirtualPathScope,
	normalizeVirtualPath,
	normalizeVirtualPathScope,
	virtualPathScopeToRegExp,
} from "./scope.ts";
export {
	type DatasourceCatalogEntry,
	derivedDatasourceIds,
	type RetrievalSelection,
	RetrievalSelectionError,
	type RetrievalSelectionErrorCode,
	resolveSelectedMethods,
} from "./selection.ts";
export {
	describeRetrievalError,
	groupUnsearchedSurfaces,
	MINSYNC_SURFACE,
	type RetrievalSkip,
	retrievalSurfaceFor,
} from "./skip.ts";
export type {
	RetrievalDiagnostic,
	RetrievalDiagnosticCode,
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
	RetrievalUnsearchedSurface,
	RetrievalWithDiagnostics,
} from "./types.ts";
