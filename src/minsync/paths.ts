import { join } from "node:path";

export const MINSYNC_SUBDIR = join(".autorag", "minsync");
export const MINSYNC_FILES_SUBDIR = "files";

export function minSyncWorkspaceRoot(root: string): string {
	return join(root, MINSYNC_SUBDIR);
}

export function minSyncDocumentPath(workspacePath: string, virtualPath: string): string {
	const safePath = virtualPath.replace(/^\/+/, "");
	return join(workspacePath, MINSYNC_FILES_SUBDIR, `${safePath}.md`);
}
