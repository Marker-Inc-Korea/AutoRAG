export { MinSyncClient, type MinSyncClientOptions } from "./client.ts";
export {
	type EnsureMinSyncBinaryOptions,
	ensureMinSyncBinary,
	fetchLatestMinSyncRelease,
	type InstalledMinSyncBinary,
	type MinSyncRelease,
	type MinSyncReleaseAsset,
	MinSyncReleaseError,
	selectReleaseAsset,
} from "./installer.ts";
export { MinSyncVectorMethod, type MinSyncVectorMethodOptions } from "./method.ts";
export { MINSYNC_FILES_SUBDIR, MINSYNC_SUBDIR, minSyncDocumentPath, minSyncWorkspaceRoot } from "./paths.ts";
export type { MinSyncOptions, MinSyncQueryHit, MinSyncSyncResult } from "./types.ts";
export {
	buildMinSyncPathMap,
	type MinSyncWorkspaceEntry,
	type MinSyncWorkspaceSyncResult,
	syncMinSyncWorkspace,
} from "./workspace.ts";
