import { isAbsolute, relative, resolve, sep } from "node:path";
import { planSourceRoots, type SourceRoot, sourceIdentifier } from "../filesystem/source-paths.ts";

export interface JikjiSourceRoot extends SourceRoot {
	readonly searchPath: string;
}

export function planJikjiSourceRoots(searchPaths: readonly string[]): readonly JikjiSourceRoot[] {
	return planSourceRoots(searchPaths).map((sourceRoot) => ({
		...sourceRoot,
		searchPath: sourceRoot.rootPath,
	}));
}

/**
 * Reduce search roots to the ones Jikji should index and query: duplicates
 * collapse to their first occurrence and a root nested inside another root is
 * dropped, because the outer root's index already covers it. Upstream Jikji
 * rejects `prepare` on a root that overlaps an indexed one, so passing nested
 * roots through only produces failed refreshes and duplicate answer paths.
 * Input order is preserved for the survivors.
 */
export function outermostJikjiRoots(searchPaths: readonly string[]): readonly string[] {
	const unique = [...new Set(searchPaths.map((searchPath) => resolve(searchPath)))];
	return unique.filter(
		(candidate) =>
			!unique.some((other) => {
				if (other === candidate) return false;
				// Exact-prefix match on a separator boundary: `/docs` contains
				// `/docs/..cache` but not `/docs-2`.
				return candidate.startsWith(other.endsWith(sep) ? other : `${other}${sep}`);
			}),
	);
}

export function mapJikjiPath(sourceRoot: SourceRoot, returnedPath: string): string | undefined {
	const resolved = resolveReturnedPath(sourceRoot.rootPath, returnedPath);
	if (resolved === undefined || !isWithinRoot(sourceRoot.rootPath, resolved)) return undefined;
	return sourceIdentifier(sourceRoot, resolved);
}

export function resolveReturnedPath(rootPath: string, returnedPath: string): string | undefined {
	if (returnedPath.length === 0) return undefined;
	const normalized = returnedPath.replace(/\\/g, "/");
	if (isUnsafeReturnedPath(normalized)) return undefined;
	if (normalized.split("/").includes("..")) return undefined;
	return resolve(rootPath, normalized);
}

export function isUnsafeReturnedPath(normalizedPath: string): boolean {
	return (
		normalizedPath.startsWith("/") ||
		/^[A-Za-z]:\//.test(normalizedPath) ||
		normalizedPath.startsWith("//") ||
		/^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(normalizedPath)
	);
}

function isWithinRoot(rootPath: string, sourcePath: string): boolean {
	const rel = relative(resolve(rootPath), resolve(sourcePath));
	return rel === "" || (!rel.startsWith("..") && !rel.includes(`..${sep}`) && !isAbsolute(rel));
}
