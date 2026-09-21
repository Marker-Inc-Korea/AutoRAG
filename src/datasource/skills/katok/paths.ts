import { join } from "node:path";

/**
 * Path helpers for the KakaoTalk (`katok`) datasource skill.
 *
 * All paths are workspace-relative and live under
 * `<workspaceRoot>/.autorag/datasources/katok`. The client never exposes raw
 * filesystem paths to callers; chunk identifiers are surfaced as opaque
 * slash-hierarchical sources of the form
 * `/kakao/<instance-id>/chunks/<chunk-id>`.
 */

export const AUTORAG_DIRNAME = ".autorag";
export const DATASOURCES_DIRNAME = "datasources";
export const KATOK_DATASOURCE_DIRNAME = "katok";

/** Logical datasource kind for source identifiers. */
export const KATOK_SOURCE_KIND = "kakao" as const;

/**
 * Root directory for katok-managed state under a workspace. This is where the
 * external `katok` CLI stores its index, chunks, and sync state.
 */
export function katokDatasourceRoot(workspaceRoot: string): string {
	return join(workspaceRoot, AUTORAG_DIRNAME, DATASOURCES_DIRNAME, KATOK_DATASOURCE_DIRNAME);
}

/** Directory holding the katok chunk index. */
export function katokIndexPath(workspaceRoot: string): string {
	return join(katokDatasourceRoot(workspaceRoot), "index");
}

/** Directory holding materialized katok chunks. */
export function katokChunksPath(workspaceRoot: string): string {
	return join(katokDatasourceRoot(workspaceRoot), "chunks");
}

/** Directory holding katok sync state. */
export function katokSyncPath(workspaceRoot: string): string {
	return join(katokDatasourceRoot(workspaceRoot), "sync");
}

/**
 * Builds the path-opaque, slash-hierarchical source identifier for a single
 * katok chunk. Callers (the KatokSkill methods) use this to populate
 * `RetrievalResult.source` so no real filesystem path ever leaks.
 */
export function katokSourcePath(instanceId: string, chunkId: string): string {
	return `/${KATOK_SOURCE_KIND}/${instanceId}/chunks/${chunkId}`;
}

/**
 * Parses a katok source identifier back into its `{ instanceId, chunkId }`
 * components. Returns `undefined` for malformed or non-kakao sources so that
 * deny decisions remain explicit rather than undefined-as-deny.
 */
export function parseKatokSourcePath(
	source: string,
): { readonly instanceId: string; readonly chunkId: string } | undefined {
	const match = /^\/kakao\/([^/]+)\/chunks\/([^/]+)$/.exec(source);
	if (match === null) return undefined;
	const [, instanceId, chunkId] = match;
	if (instanceId === undefined || chunkId === undefined) return undefined;
	return { instanceId, chunkId };
}
