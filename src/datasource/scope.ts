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
import type { RetrievalMethod, RetrievalResult } from "../retrieval/types.ts";

/** Per-method retrieval results keyed by method name. */
export type ResultsByMethod = Map<string, RetrievalResult[]>;

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

/**
 * Narrow a per-method result map by the ordinary query `scope`.
 *
 * This is the only post-retrieval narrowing left in the datasource layer: every
 * configured datasource is searchable without permission setup. Methods whose
 * descriptor has no `datasourceId` (plain retrieval methods such as `posix`) and
 * datasource methods that do not advertise the `scoped` capability pass through
 * untouched. Scope-capable datasource methods keep only the results whose
 * source matches `scope`; `undefined` scope matches every valid source, and
 * sources containing a `#` fragment are always rejected.
 *
 * Result entries for methods not present in `methods` are passed through
 * unchanged (they cannot be classified). A new map is returned; the input is
 * not mutated.
 */
export function filterDatasourceScope(
	byMethod: ResultsByMethod,
	methods: readonly RetrievalMethod[],
	scope?: string,
): ResultsByMethod {
	const descriptors = new Map<string, RetrievalMethod>();
	for (const method of methods) {
		descriptors.set(method.describe().name, method);
	}

	const out: ResultsByMethod = new Map();
	for (const [name, results] of byMethod) {
		const method = descriptors.get(name);
		if (method === undefined) {
			// No descriptor available: cannot classify as a datasource method.
			out.set(name, results);
			continue;
		}
		const descriptor = method.describe();
		if (descriptor.datasourceId === undefined || !descriptor.capabilities.includes("scoped")) {
			// Non-datasource, or a datasource without source-scope support: pass through.
			out.set(name, results);
			continue;
		}
		out.set(
			name,
			results.filter((result) => matchesDatasourceScope(result.source, scope)),
		);
	}
	return out;
}

export { matchesVirtualPathScope, normalizeVirtualPath } from "../retrieval/scope.ts";
