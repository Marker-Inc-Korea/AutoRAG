import { join } from "node:path";

/**
 * Path helpers for the KakaoTalk (`lazykatok`) datasource skill.
 *
 * All paths are workspace-relative and live under
 * `<workspaceRoot>/.autorag/datasources/lazykatok`. The client never exposes raw
 * filesystem paths to callers; chunk identifiers are surfaced as opaque
 * slash-hierarchical sources of the form
 * `/kakao/<instance-id>/chunks/<chunk-id>`.
 */

export const AUTORAG_DIRNAME = ".autorag";
export const DATASOURCES_DIRNAME = "datasources";
export const LAZYKATOK_DATASOURCE_DIRNAME = "lazykatok";

/** Logical datasource kind for source identifiers. */
export const LAZYKATOK_SOURCE_KIND = "kakao" as const;

/**
 * Root directory for lazykatok-managed state under a workspace. This is where the
 * external `lazykatok` CLI stores its index, chunks, and sync state.
 */
export function lazykatokDatasourceRoot(workspaceRoot: string): string {
	return join(workspaceRoot, AUTORAG_DIRNAME, DATASOURCES_DIRNAME, LAZYKATOK_DATASOURCE_DIRNAME);
}

/** Directory holding the lazykatok chunk index. */
export function lazykatokIndexPath(workspaceRoot: string): string {
	return join(lazykatokDatasourceRoot(workspaceRoot), "index");
}

/** Directory holding materialized lazykatok chunks. */
export function lazykatokChunksPath(workspaceRoot: string): string {
	return join(lazykatokDatasourceRoot(workspaceRoot), "chunks");
}

/** Directory holding lazykatok sync state. */
export function lazykatokSyncPath(workspaceRoot: string): string {
	return join(lazykatokDatasourceRoot(workspaceRoot), "sync");
}

/**
 * Builds the path-opaque, slash-hierarchical source identifier for a single
 * lazykatok chunk. Callers (the LazykatokSkill methods) use this to populate
 * `RetrievalResult.source` so no real filesystem path ever leaks.
 */
export function lazykatokSourcePath(instanceId: string, chunkId: string): string {
	return `/${LAZYKATOK_SOURCE_KIND}/${instanceId}/chunks/${chunkId}`;
}

/**
 * Parses a lazykatok source identifier back into its `{ instanceId, chunkId }`
 * components. Returns `undefined` for malformed or non-kakao sources so that
 * deny decisions remain explicit rather than undefined-as-deny.
 */
export function parseLazykatokSourcePath(
	source: string,
): { readonly instanceId: string; readonly chunkId: string } | undefined {
	const match = /^\/kakao\/([^/]+)\/chunks\/([^/]+)$/.exec(source);
	if (match === null) return undefined;
	const [, instanceId, chunkId] = match;
	if (instanceId === undefined || chunkId === undefined) return undefined;
	return { instanceId, chunkId };
}
