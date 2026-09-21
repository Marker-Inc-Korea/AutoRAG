/**
 * Slash-hierarchical source-path helpers for the datasource layer.
 *
 * Datasource sources are opaque, slash-hierarchical paths such as
 * `/kakao/<instance-id>` and `/kakao/<instance-id>/chunks/<chunk-id>`.
 * `#` fragments are NEVER produced or matched — sources are pure path trees.
 *
 * Reuses {@link normalizeVirtualPath} and {@link matchesVirtualPathScope} from
 * `src/retrieval/scope.ts` so datasource scopes compose with the existing
 * retrieval scope grammar (globs, leading-slash normalization).
 */

import { matchesVirtualPathScope, normalizeVirtualPath } from "../retrieval/scope.ts";

/** Path segment separating an instance root from its chunk sub-namespace. */
export const DATASOURCE_CHUNKS_SEGMENT = "chunks";

/**
 * Build the opaque root source for a datasource instance, e.g.
 * `/kakao/acct-1`.
 */
export function buildDatasourceInstanceSource(skillName: string, instanceId: string): string {
	return normalizeVirtualPath(`/${skillName}/${instanceId}`);
}

/**
 * Build the opaque source for a single datasource chunk, e.g.
 * `/kakao/acct-1/chunks/c-42`.
 */
export function buildDatasourceChunkSource(skillName: string, instanceId: string, chunkId: string): string {
	return normalizeVirtualPath(`/${skillName}/${instanceId}/${DATASOURCE_CHUNKS_SEGMENT}/${chunkId}`);
}
export function datasourceSourcePath(skillName: string, instanceId: string, chunkId?: string): string {
	return chunkId === undefined
		? buildDatasourceInstanceSource(skillName, instanceId)
		: buildDatasourceChunkSource(skillName, instanceId, chunkId);
}

/** True when a source string contains a `#` fragment (always invalid here). */
export function datasourceSourceHasFragment(source: string): boolean {
	return typeof source === "string" && source.includes("#");
}

/**
 * Whether a string is a valid datasource source: a non-root, normalized,
 * slash-hierarchical path with no `#` fragment.
 */
export function isDatasourceSource(source: string): boolean {
	if (typeof source !== "string" || source.length === 0) return false;
	if (datasourceSourceHasFragment(source)) return false;
	const normalized = normalizeVirtualPath(source);
	return normalized !== "/";
}

/**
 * Match a datasource source against a scope. Returns `false` when the source
 * contains a `#` fragment; otherwise delegates to the shared retrieval
 * scope matcher so globs and leading-slash normalization behave identically.
 */
export function matchesDatasourceScope(source: string, scope: string | undefined): boolean {
	if (datasourceSourceHasFragment(source)) return false;
	return matchesVirtualPathScope(source, scope);
}

export { matchesVirtualPathScope, normalizeVirtualPath } from "../retrieval/scope.ts";
