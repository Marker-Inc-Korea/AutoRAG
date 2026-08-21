import { join } from "node:path";

/**
 * Path helpers for the Obsidian (`qmd`) datasource skill.
 *
 * All AutoRAG-managed state lives under
 * `<workspaceRoot>/.autorag/datasources/obsidian/<instanceId>/`. The external
 * `qmd` CLI is pointed at isolated config/cache dirs so multiple workspaces
 * never share an index. Callers surface opaque slash-hierarchical sources of
 * the form `/obsidian/<instance-id>/chunks/<chunk-id>`.
 */

export const AUTORAG_DIRNAME = ".autorag";
export const DATASOURCES_DIRNAME = "datasources";
export const OBSIDIAN_DATASOURCE_DIRNAME = "obsidian";
export const OBSIDIAN_SOURCE_KIND = "obsidian" as const;

export function obsidianDatasourceRoot(workspaceRoot: string, instanceId: string): string {
	return join(workspaceRoot, AUTORAG_DIRNAME, DATASOURCES_DIRNAME, OBSIDIAN_DATASOURCE_DIRNAME, instanceId);
}

/** Directory passed to `QMD_CONFIG_DIR` (holds `index.yml`). */
export function obsidianQmdConfigDir(workspaceRoot: string, instanceId: string): string {
	return join(obsidianDatasourceRoot(workspaceRoot, instanceId), "config");
}

/** Directory passed to `XDG_CACHE_HOME` (holds `qmd/index.sqlite`). */
export function obsidianQmdCacheDir(workspaceRoot: string, instanceId: string): string {
	return join(obsidianDatasourceRoot(workspaceRoot, instanceId), "cache");
}

export function obsidianSourcePath(instanceId: string, chunkId: string): string {
	return `/${OBSIDIAN_SOURCE_KIND}/${instanceId}/chunks/${chunkId}`;
}

export function parseObsidianSourcePath(
	source: string,
): { readonly instanceId: string; readonly chunkId: string } | undefined {
	const match = /^\/obsidian\/([^/]+)\/chunks\/([^/]+)$/.exec(source);
	if (match === null) return undefined;
	const [, instanceId, chunkId] = match;
	if (instanceId === undefined || chunkId === undefined) return undefined;
	return { instanceId, chunkId };
}

/** Strip leading and trailing ASCII dashes without a quadratic regex. */
export function stripEdgeDashes(value: string): string {
	let start = 0;
	let end = value.length;
	while (start < end && value[start] === "-") start += 1;
	while (end > start && value[end - 1] === "-") end -= 1;
	return value.slice(start, end);
}

/** Sanitize an instance id into a qmd collection name. */
export function toQmdCollectionName(instanceId: string): string {
	const cleaned = stripEdgeDashes(
		instanceId
			.trim()
			.toLowerCase()
			.replace(/[^a-z0-9._-]+/g, "-"),
	);
	return cleaned.length > 0 ? cleaned.slice(0, 64) : "default";
}
