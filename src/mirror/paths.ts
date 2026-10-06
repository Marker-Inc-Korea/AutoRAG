import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const PARSED_MIRROR_SUBDIR = join(".autorag", "parsed");
export const PARSED_FILES_SUBDIR = "files";
export const PARSED_INDEX_FILE = "index.json";
export const REFRESH_READINESS_FILE = "refresh-complete.json";
export const REFRESH_PROGRESS_FILE = "refresh-progress.json";

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

export function refreshReadinessPath(root: string): string {
	return join(root, ".autorag", REFRESH_READINESS_FILE);
}

export function refreshProgressPath(root: string): string {
	return join(root, ".autorag", REFRESH_PROGRESS_FILE);
}
/**
 * Whether a parsed-mirror refresh has completed at least once for the given
 * workspace. Reads the persisted readiness marker so a separate process (CLI,
 * MCP server) reaches the same verdict as the process that ran the refresh.
 */
export function isParsedRefreshComplete(workspacePath: string): boolean {
	const markerPath = refreshReadinessPath(workspacePath);
	if (!existsSync(markerPath)) return false;
	try {
		const marker: unknown = JSON.parse(readFileSync(markerPath, "utf8"));
		return (
			typeof marker === "object" &&
			marker !== null &&
			"version" in marker &&
			marker.version === 1 &&
			"completed" in marker &&
			marker.completed === true &&
			"parsed" in marker &&
			marker.parsed === true
		);
	} catch {
		return false;
	}
}
