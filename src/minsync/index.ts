export {
	MINSYNC_OLLAMA_MIGRATION_MESSAGE,
	MinSyncClient,
	type MinSyncClientOptions,
	MinSyncQueryError,
	type MinSyncQueryMode,
} from "./client.ts";
export {
	configuredVectorDimension,
	DEFAULT_MINSYNC_EMBEDDER_CONFIG,
	DEFAULT_MINSYNC_EMBEDDER_DIMENSION,
	DEFAULT_MINSYNC_EMBEDDER_ID,
	MINSYNC_CONFIG_DIR,
	MINSYNC_CONFIG_FILE,
	type MinSyncEmbeddingIdentity,
	minSyncConfigPath,
	minSyncEmbeddingIdentityPath,
	rewriteEmbedderConfig,
} from "./embedder-config.ts";
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
	MinSyncChunkerConfig,
	MinSyncDiagnostic,
	MinSyncEmbedderConfig,
	MinSyncOptions,
	MinSyncQueryHit,
	MinSyncSyncResult,
} from "./types.ts";
export {
	buildMinSyncPathMap,
	minSyncMirrorFingerprint,
	type MinSyncWorkspaceEntry,
	type MinSyncWorkspaceSyncResult,
	syncMinSyncWorkspace,
} from "./workspace.ts";
