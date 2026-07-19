export { MinSyncClient, type MinSyncClientOptions } from "./client.ts";
export {
	MINSYNC_CONFIG_DIR,
	MINSYNC_CONFIG_FILE,
	minSyncConfigPath,
	rewriteEmbedderConfig,
} from "./embedder-config.ts";
export {
	type EnsureMinSyncBinaryOptions,
	ensureMinSyncBinary,
	executableName,
	fetchLatestMinSyncRelease,
	type InstalledMinSyncBinary,
	type MinSyncRelease,
	type MinSyncReleaseAsset,
	MinSyncReleaseError,
	selectReleaseAsset,
} from "./installer.ts";
export { MinSyncVectorMethod, type MinSyncVectorMethodOptions } from "./method.ts";
export { MINSYNC_FILES_SUBDIR, MINSYNC_SUBDIR, minSyncDocumentPath, minSyncWorkspaceRoot } from "./paths.ts";
export type { MinSyncEmbedderConfig, MinSyncOptions, MinSyncQueryHit, MinSyncSyncResult } from "./types.ts";
export {
	buildMinSyncPathMap,
	type MinSyncWorkspaceEntry,
	type MinSyncWorkspaceSyncResult,
	syncMinSyncWorkspace,
} from "./workspace.ts";
