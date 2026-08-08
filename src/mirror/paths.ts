import { createHash } from "node:crypto";
import { join } from "node:path";

/** Single source of truth for the workspace-local autorag directory name. */
export const AUTORAG_DIR = ".autorag";

export const PARSED_MIRROR_SUBDIR = join(AUTORAG_DIR, "parsed");
export const PARSED_FILES_SUBDIR = "files";
export const PARSED_INDEX_FILE = "index.json";

export function parsedMirrorRoot(root: string): string {
	return join(root, PARSED_MIRROR_SUBDIR);
}

export function parsedMirrorIndexPath(root: string): string {
	return join(parsedMirrorRoot(root), PARSED_INDEX_FILE);
}

export function parsedOutputPath(root: string, virtualPath: string): string {
	const digest = createHash("sha256").update(virtualPath).digest("hex");
	return join(parsedMirrorRoot(root), PARSED_FILES_SUBDIR, `${digest}.md`);
}
