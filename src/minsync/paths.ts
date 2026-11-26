import { join } from "node:path";

export const MINSYNC_SUBDIR = join(".autorag", "minsync");
export const MINSYNC_FILES_SUBDIR = "files";
export const MINSYNC_VISUAL_SUFFIX = ".autorag-visual.md";

export function minSyncWorkspaceRoot(root: string): string {
	return join(root, MINSYNC_SUBDIR);
}

export function minSyncDocumentPath(workspacePath: string, virtualPath: string): string {
	const safePath = virtualPath.replace(/^\/+/, "");
	return join(workspacePath, MINSYNC_FILES_SUBDIR, `${safePath}.md`);
}

/** Staged path of the supplementary visual-extraction mirror for `virtualPath` inside MinSync's `files/` tree. */
export function minSyncVisualDocumentPath(workspacePath: string, virtualPath: string): string {
	const safePath = virtualPath.replace(/^\/+/, "");
	return join(workspacePath, MINSYNC_FILES_SUBDIR, `${safePath}${MINSYNC_VISUAL_SUFFIX}`);
}
