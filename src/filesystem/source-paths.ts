import { realpathSync } from "node:fs";
import { basename, isAbsolute, relative, resolve, sep } from "node:path";

export interface SourceRoot {
	readonly rootPath: string;
	readonly prefix: string;
}

export function planSourceRoots(searchPaths: readonly string[]): readonly SourceRoot[] {
	const sorted = searchPaths.map((path) => resolve(path)).sort((a, b) => a.localeCompare(b));
	const used = new Set<string>();
	return sorted.map((rootPath) => {
		const base = basename(rootPath.replace(/[/\\]+$/, "")) || "root";
		let prefix = `/${base}`;
		let suffix = 2;
		while (used.has(prefix)) {
			prefix = `/${base}-${suffix}`;
			suffix += 1;
		}
		used.add(prefix);
		return { rootPath, prefix };
	});
}

export function sourceIdentifier(sourceRoot: SourceRoot, sourcePath: string): string {
	const rel = relative(sourceRoot.rootPath, resolve(sourcePath));
	const suffix = rel === "" ? "" : `/${rel.split(sep).join("/")}`;
	return `${sourceRoot.prefix}${suffix}`;
}

/**
 * Resolved mapping from an opaque virtual source id back to a real filesystem
 * path within a configured source root. Real paths NEVER leave this module:
 * callers surface only `sourceId` (opaque) in tool content/details.
 */
export interface ResolvedVirtualSource {
	readonly root: SourceRoot;
	readonly realPath: string;
	readonly sourceId: string;
}

const URL_SCHEME_RE = /^[a-z][a-z0-9+.-]*:/i;

/**
 * Normalize an opaque virtual source id. Returns the normalized id, or
 * `undefined` when the input is syntactically invalid or unsafe (no leading
 * slash, URL scheme, fragment/query, traversal, NUL, backslash). Never throws
 * and never returns a real path — failures are signaled as `undefined`.
 */
export function normalizeVirtualPath(virtual: string | undefined | null): string | undefined {
	if (typeof virtual !== "string") return undefined;
	let v = virtual.trim();
	if (v.length === 0) return undefined;
	if (URL_SCHEME_RE.test(v)) return undefined;
	if (v.includes("#") || v.includes("?")) return undefined;
	if (v.includes("\0")) return undefined;
	if (!v.startsWith("/")) return undefined;
	if (v.includes("\\")) return undefined;
	v = v.replace(/\/+/g, "/");
	if (v.length > 1 && v.endsWith("/")) v = v.slice(0, -1);
	const segments = v === "/" ? [] : v.slice(1).split("/");
	for (const segment of segments) {
		if (segment === ".." || segment === ".") return undefined;
	}
	return v;
}

/**
 * Exact-prefix boundary match: `/docs` matches `/docs` and `/docs/...` but NOT
 * `/docs-2`. Prevents prefix confusion between sibling roots like `/docs` and
 * `/docs-2`.
 */
export function virtualRootMatches(virtual: string, root: SourceRoot): boolean {
	return virtual === root.prefix || virtual.startsWith(`${root.prefix}/`);
}

/**
 * Reverse-resolve an opaque virtual source id to a real path within a
 * configured source root. Selects the longest matching root by exact-prefix
 * boundary, then verifies containment via `realpath` against the selected
 * root's realpath. Symlinks that escape the root are rejected. Returns
 * `undefined` (path-free failure) instead of throwing when the source is
 * missing, out-of-scope, or escapes containment.
 */
export function resolveVirtualSource(
	virtual: string,
	sourceRoots: readonly SourceRoot[],
): ResolvedVirtualSource | undefined {
	const normalized = normalizeVirtualPath(virtual);
	if (normalized === undefined) return undefined;
	const candidates = sourceRoots
		.filter((root) => virtualRootMatches(normalized, root))
		.sort((a, b) => b.prefix.length - a.prefix.length);
	for (const root of candidates) {
		const rel = normalized === root.prefix ? "" : normalized.slice(root.prefix.length + 1);
		const candidate = rel.length === 0 ? resolve(root.rootPath) : resolve(root.rootPath, ...rel.split("/"));
		let realPath: string;
		let rootReal: string;
		try {
			realPath = realpathSync(candidate);
			rootReal = realpathSync(root.rootPath);
		} catch {
			continue;
		}
		const relCheck = relative(rootReal, realPath);
		if (relCheck.startsWith("..") || isAbsolute(relCheck)) {
			continue;
		}
		return { root, realPath, sourceId: normalized };
	}
	return undefined;
}
