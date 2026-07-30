import { realpathSync } from "node:fs";
import { dirname, posix, resolve, win32 } from "node:path";
import { planSourceRoots } from "../filesystem/source-paths.ts";
import { loadMirrorIndex } from "../mirror/index-store.ts";

export interface RetrievalScopeBinding {
	readonly virtualPrefix: string;
	readonly physicalRoots: readonly string[];
}

export class RetrievalScopeError extends Error {
	readonly code = "invalid-retrieval-scope";

	constructor(virtualPrefixes: readonly string[]) {
		const guidance = virtualPrefixes.length > 0 ? ` Use a virtual scope under ${virtualPrefixes.join(", ")}.` : "";
		super(`invalid-retrieval-scope: physical scope is outside the configured search roots.${guidance}`);
		this.name = "RetrievalScopeError";
	}
}

export function normalizeVirtualPath(value: string): string {
	const normalized = value.replace(/\\/g, "/").replace(/\/+/g, "/").trim();
	if (normalized.length === 0 || normalized === "/") return "/";
	const withLeadingSlash = normalized.startsWith("/") ? normalized : `/${normalized}`;
	return withLeadingSlash.length > 1 ? withLeadingSlash.replace(/\/+$/g, "") : withLeadingSlash;
}

export function normalizeVirtualPathScope(scope: string | undefined): string | undefined {
	if (scope === undefined) return undefined;
	const normalized = normalizeVirtualPath(scope);
	return normalized === "/" ? undefined : normalized;
}

export function resolveRetrievalScope(
	scope: string | undefined,
	bindings: readonly RetrievalScopeBinding[],
	platform: NodeJS.Platform = process.platform,
	passthroughVirtualPrefixes: readonly string[] = [],
): string | undefined {
	if (scope === undefined) return undefined;
	const trimmed = scope.trim();
	if (trimmed.length === 0) return undefined;
	const normalizedVirtual = normalizeVirtualPathScope(trimmed);
	if (
		normalizedVirtual !== undefined &&
		(bindings.some((binding) => virtualPathBelongsToRoot(normalizedVirtual, binding.virtualPrefix)) ||
			passthroughVirtualPrefixes.some((prefix) => virtualPathBelongsToRoot(normalizedVirtual, prefix)))
	) {
		return normalizedVirtual;
	}

	const pathApi = platform === "win32" ? win32 : posix;
	let bestMatch: { mappedScope: string; specificity: number } | undefined;
	for (const binding of bindings) {
		for (const physicalRoot of binding.physicalRoots) {
			for (const physicalScope of physicalScopeCandidates(trimmed, platform)) {
				const relativePath = pathApi.relative(pathApi.normalize(physicalRoot), pathApi.normalize(physicalScope));
				if (!isContainedRelativePath(relativePath, pathApi)) continue;
				const mappedScope =
					relativePath.length === 0
						? normalizeVirtualPath(binding.virtualPrefix)
						: normalizeVirtualPath(`${binding.virtualPrefix}/${relativePath.split(pathApi.sep).join("/")}`);
				const specificity = pathApi.normalize(physicalRoot).length;
				if (bestMatch === undefined || specificity > bestMatch.specificity) {
					bestMatch = { mappedScope, specificity };
				} else if (specificity === bestMatch.specificity && mappedScope !== bestMatch.mappedScope) {
					throw new RetrievalScopeError(bindings.map((candidate) => candidate.virtualPrefix));
				}
			}
		}
	}
	if (bestMatch !== undefined) return bestMatch.mappedScope;

	if (isUnambiguouslyPhysicalScope(trimmed, platform)) {
		throw new RetrievalScopeError(bindings.map((binding) => binding.virtualPrefix));
	}
	return normalizedVirtual;
}

export function buildRetrievalScopeBindings(
	workspaceRoot: string,
	searchPaths: readonly string[],
	configuredSearchPaths: readonly string[] = searchPaths,
): readonly RetrievalScopeBinding[] {
	const currentRoots = planSourceRoots(searchPaths);
	const persistedRoots = persistedScopeRoots(workspaceRoot);
	const configuredAliases = configuredPathAliases(configuredSearchPaths);
	const singlePersisted = persistedRoots.size === 1 ? [...persistedRoots.entries()][0] : undefined;

	return currentRoots.map((currentRoot) => {
		const exactPersisted = [...persistedRoots.entries()].find(([, roots]) =>
			roots.some((root) => samePhysicalPath(root, currentRoot.rootPath)),
		);
		const persisted = exactPersisted ?? (currentRoots.length === 1 ? singlePersisted : undefined);
		const virtualPrefix = persisted?.[0] ?? currentRoot.prefix;
		const physicalRoots = new Set<string>([currentRoot.rootPath, ...(persisted?.[1] ?? [])]);
		for (const alias of configuredAliases) {
			if (samePhysicalPath(alias.canonical, currentRoot.rootPath)) {
				physicalRoots.add(alias.configured);
				physicalRoots.add(alias.canonical);
			}
		}
		return { virtualPrefix, physicalRoots: [...physicalRoots] };
	});
}

export function matchesVirtualPathScope(virtualPath: string, scope: string | undefined): boolean {
	const normalizedScope = normalizeVirtualPathScope(scope);
	if (normalizedScope === undefined) return true;
	return virtualPathScopeToRegExp(normalizedScope).test(normalizeVirtualPath(virtualPath));
}

export function virtualPathScopeToRegExp(scope: string): RegExp {
	const normalized = normalizeVirtualPath(scope);
	const scoped = hasGlob(normalized) || looksLikeFileScope(normalized) ? normalized : `${normalized}/**`;
	const pattern = scoped
		.split("/")
		.map((segment, index) => {
			if (index === 0) return "";
			if (segment === "**") return "(?:.*)";
			return escapeRegExp(segment).replace(/\\\*/g, "[^/]*");
		})
		.join("/")
		.replace(/\/\(\?:\.\*\)$/u, "(?:/.*)?");
	return new RegExp(`^${pattern}$`);
}

function hasGlob(scope: string): boolean {
	return scope.includes("*");
}

function virtualPathBelongsToRoot(scope: string, virtualPrefix: string): boolean {
	const normalizedRoot = normalizeVirtualPath(virtualPrefix);
	return scope === normalizedRoot || scope.startsWith(`${normalizedRoot}/`);
}

function isContainedRelativePath(relativePath: string, pathApi: typeof posix | typeof win32): boolean {
	return (
		relativePath === "" ||
		(relativePath !== ".." && !relativePath.startsWith(`..${pathApi.sep}`) && !pathApi.isAbsolute(relativePath))
	);
}

function isUnambiguouslyPhysicalScope(scope: string, platform: NodeJS.Platform): boolean {
	if (platform === "win32") return win32.isAbsolute(scope);
	return posix.isAbsolute(scope);
}

function physicalScopeCandidates(scope: string, platform: NodeJS.Platform): readonly string[] {
	if (platform !== process.platform || hasGlob(scope)) return [scope];
	try {
		return [scope, realpathSync(scope)];
	} catch {
		return [scope];
	}
}

function configuredPathAliases(paths: readonly string[]): readonly { configured: string; canonical: string }[] {
	return paths.map((path) => {
		const configured = resolve(path);
		try {
			return { configured, canonical: realpathSync(configured) };
		} catch {
			return { configured, canonical: configured };
		}
	});
}

function persistedScopeRoots(workspaceRoot: string): ReadonlyMap<string, readonly string[]> {
	const roots = new Map<string, Set<string>>();
	for (const entry of Object.values(loadMirrorIndex(workspaceRoot).entries)) {
		const normalized = normalizeVirtualPath(entry.virtualPath);
		const segments = normalized.slice(1).split("/");
		const virtualPrefix = segments[0] ? `/${segments[0]}` : "/";
		let physicalRoot = entry.sourcePath;
		for (let index = 1; index < segments.length; index += 1) {
			physicalRoot = dirname(physicalRoot);
		}
		const candidates = roots.get(virtualPrefix) ?? new Set<string>();
		candidates.add(physicalRoot);
		roots.set(virtualPrefix, candidates);
	}
	return new Map([...roots].map(([prefix, physicalRoots]) => [prefix, [...physicalRoots]]));
}

function samePhysicalPath(left: string, right: string): boolean {
	const pathApi = process.platform === "win32" ? win32 : posix;
	return pathApi.relative(pathApi.normalize(left), pathApi.normalize(right)) === "";
}

function looksLikeFileScope(scope: string): boolean {
	const finalSegment = scope.split("/").at(-1) ?? "";
	return /\.[^/.]+$/u.test(finalSegment);
}

function escapeRegExp(literal: string): string {
	return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
