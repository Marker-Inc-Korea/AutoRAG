export {
	MinSyncClient,
	type MinSyncClientOptions,
	MinSyncQueryError,
	type MinSyncQueryMode,
} from "./client.ts";
export {
	AUTO_RETRIEVER_ID,
	configuredBm25Config,
	configuredSparseId,
	configuredVectorDimension,
	DEFAULT_MINSYNC_BM25_LANGUAGE,
	DEFAULT_MINSYNC_EMBEDDER_CONFIG,
	DEFAULT_MINSYNC_EMBEDDER_DIMENSION,
	DEFAULT_MINSYNC_EMBEDDER_ID,
	DEFAULT_MINSYNC_SPARSE_ID,
	isBm25Enabled,
	isBm25OnlyConfig,
	isSparseDisabled,
	isSparseOnlyEmbedder,
	MINSYNC_CONFIG_DIR,
	MINSYNC_CONFIG_FILE,
	type MinSyncEmbeddingIdentity,
	minSyncConfigPath,
	minSyncEmbeddingIdentityPath,
	NONE_SPARSE_ID,
	rewriteEmbedderConfig,
	SPARSE_ONLY_EMBEDDER_ID,
	SUPPORTED_MINSYNC_BM25_LANGUAGES,
} from "./embedder-config.ts";
export { MinSyncRequiredError } from "./errors.ts";
export {
	AUTORAG_HARDWARE_OVERRIDE_ENV,
	type CudaGpuInfo,
	evaluateMinSyncHardwareTier,
	MINSYNC_DENSE_MIN_APPLE_RAM_GIB,
	MINSYNC_DENSE_MIN_CUDA_HOST_RAM_GIB,
	MINSYNC_DENSE_MIN_CUDA_VRAM_MIB,
	MINSYNC_HARDWARE_OVERRIDE_ENV,
	MINSYNC_SPLADE_MIN_CPU_CORES,
	MINSYNC_SPLADE_MIN_RAM_GIB,
	type MinSyncHardwareProfile,
	type MinSyncHardwareRecommendation,
	type MinSyncIndexingTier,
	parseNvidiaSmiOutput,
	probeMinSyncHardware,
	type ResolvedMinSyncRetrieverPlan,
	resolveMinSyncRetrieverPlan,
} from "./hardware.ts";
export {
	type EnsureMinSyncBinaryOptions,
	ensureMinSyncBinary,
	executableName,
	fetchLatestMinSyncRelease,
	type InstalledMinSyncBinary,
	MINSYNC_VERSION,
	type MinSyncRelease,
	type MinSyncReleaseAsset,
	MinSyncReleaseError,
	selectReleaseAsset,
} from "./installer.ts";
export {
	ensureLocalEmbedder,
	LocalEmbedderError,
	type LocalEmbedderPreflightOptions,
} from "./local-embedder.ts";
export {
	MinSyncHybridMethod,
	type MinSyncHybridMethodOptions,
	MinSyncVectorMethod,
	type MinSyncVectorMethodOptions,
} from "./method.ts";
export { MINSYNC_FILES_SUBDIR, MINSYNC_SUBDIR, minSyncDocumentPath, minSyncWorkspaceRoot } from "./paths.ts";
export type {
	MinSyncBm25Config,
	MinSyncChunkerConfig,
	MinSyncDiagnostic,
	MinSyncEmbedderConfig,
	MinSyncFileFailure,
	MinSyncOptions,
	MinSyncQueryHit,
	MinSyncRetrievers,
	MinSyncSparseConfig,
	MinSyncSyncProgress,
	MinSyncSyncResult,
} from "./types.ts";
export {
	buildMinSyncPathMap,
	type MinSyncWorkspaceEntry,
	type MinSyncWorkspaceSyncResult,
	syncMinSyncWorkspace,
} from "./workspace.ts";
