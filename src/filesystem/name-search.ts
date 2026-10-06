import { type Dirent, realpathSync } from "node:fs";
import { lstat, readdir } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";

/**
 * Literal (never regex) file-name search over configured filesystem roots.
 *
 * This is a discovery-only backend: it walks directories directly and never
 * consults a parsed mirror/index, so a name can be found before any refresh
 * has run. It never opens file contents — only directory listings are read.
 *
 * Path safety is the whole point of this module:
 * - configured roots are `realpath`-pinned; missing roots degrade to a
 *   diagnostic instead of an exception;
 * - `request.root` may only select a configured root or a real subdirectory
 *   *inside* one (realpath containment), so it can never widen scope;
 * - child symlinks are never followed, so a symlink inside a root cannot leak
 *   results from outside it;
 * - when nothing can be searched, the result is an empty page plus a warning
 *   diagnostic, never an arbitrary global `/` scan.
 */

/** A single matched entry, with its canonical absolute filesystem path. */
export interface FileNameSearchMatch {
	readonly path: string;
	readonly type: "file" | "folder";
}

/** A non-fatal problem encountered while resolving or walking roots. */
export interface FileNameSearchDiagnostic {
	readonly code: string;
	readonly severity: "warning";
	readonly source: string;
	readonly message: string;
}

export interface FileNameSearchRequest {
	readonly query: string;
	/** Configured root or subdirectory (absolute or root-relative) to limit the walk to. */
	readonly root?: string;
	/** Match the query against the path relative to the search root instead of the file name. */
	readonly matchPath?: boolean;
	/** Case-sensitive matching. Defaults to case-insensitive. */
	readonly matchCase?: boolean;
	readonly kind?: "files" | "folders";
	readonly maxResults?: number;
	readonly offset?: number;
	readonly signal?: AbortSignal;
}

export interface FileNameSearchResult {
	readonly ok: true;
	readonly backend: "filesystem" | "everything";
	readonly results: readonly FileNameSearchMatch[];
	readonly truncated: boolean;
	readonly diagnostics: readonly FileNameSearchDiagnostic[];
}

/** Directory names owned by AutoRAG/git/tooling, never corpus content. */
const SKIP_DIR_NAMES: Record<string, true> = {
	".git": true,
	".autorag": true,
	".jikji": true,
	node_modules: true,
};

const DEFAULT_MAX_RESULTS = 100;
const MAX_MAX_RESULTS = 1000;

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
	const truncated = Math.trunc(value);
	if (truncated < min) return min;
	if (truncated > max) return max;
	return truncated;
}

function fsErrorCode(error: unknown): string | undefined {
	if (typeof error !== "object" || error === null) return undefined;
	if (!("code" in error)) return undefined;
	const { code } = error;
	return typeof code === "string" ? code : undefined;
}

/** Exact-prefix containment on canonical real paths: `/a` contains `/a` and `/a/b`, not `/a-b`. */
function isContained(childReal: string, parentReal: string): boolean {
	const prefix = parentReal.endsWith(sep) ? parentReal : `${parentReal}${sep}`;
	return childReal === parentReal || childReal.startsWith(prefix);
}

/**
 * Canonicalize an exclusion the same way walked entries are built: pin the
 * parent chain (`realpath`) but keep the final component literal, so excluding
 * a symlinked file still matches the literal name the walker reports. Missing
 * paths keep their resolved form (excluding a not-yet-existing path is legal).
 */
function pinExcludedPath(path: string): string {
	const resolvedPath = resolve(path);
	try {
		return join(realpathSync(dirname(resolvedPath)), basename(resolvedPath));
	} catch {
		return resolvedPath;
	}
}

function isExcluded(absPath: string, excluded: readonly string[]): boolean {
	for (const excludedPath of excluded) {
		const prefix = excludedPath.endsWith(sep) ? excludedPath : `${excludedPath}${sep}`;
		if (absPath === excludedPath || absPath.startsWith(prefix)) return true;
	}
	return false;
}

interface WalkContext {
	readonly rootReal: string;
	readonly query: string;
	readonly matchPath: boolean;
	readonly matchCase: boolean;
	readonly kind: "files" | "folders" | undefined;
	readonly limit: number;
	readonly excluded: readonly string[];
	readonly signal: AbortSignal | undefined;
	readonly matches: FileNameSearchMatch[];
	readonly diagnostics: FileNameSearchDiagnostic[];
	stopped: boolean;
}

function matchesEntry(context: WalkContext, absPath: string, name: string): boolean {
	const haystackRaw = context.matchPath ? relative(context.rootReal, absPath).split(sep).join("/") : name;
	const haystack = context.matchCase ? haystackRaw : haystackRaw.toLowerCase();
	const needle = context.matchCase ? context.query : context.query.toLowerCase();
	return haystack.includes(needle);
}

function isCancelled(context: WalkContext): boolean {
	return context.stopped || context.signal?.aborted === true;
}

async function walkDirectory(context: WalkContext, directory: string): Promise<void> {
	if (isCancelled(context)) {
		context.stopped = true;
		return;
	}
	let entries: Dirent[];
	try {
		entries = await readdir(directory, { withFileTypes: true });
	} catch (error) {
		if (context.signal?.aborted === true) {
			context.stopped = true;
			return;
		}
		const code = fsErrorCode(error);
		// A raced-away directory is not a problem worth surfacing; anything else is.
		if (code === "ENOENT") return;
		context.diagnostics.push({
			code: code === "EACCES" || code === "EPERM" ? "permission-denied" : "read-error",
			severity: "warning",
			source: directory,
			message: `Could not read directory: ${error instanceof Error ? error.message : String(error)}`,
		});
		return;
	}
	// Deterministic traversal: sort each listing by name (code-unit order) before walking.
	entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
	for (const entry of entries) {
		if (isCancelled(context)) {
			context.stopped = true;
			return;
		}
		// Never follow symlinks: a child link could point outside the root.
		if (entry.isSymbolicLink()) continue;
		const absPath = join(directory, entry.name);
		if (entry.isDirectory()) {
			if (SKIP_DIR_NAMES[entry.name] === true) continue;
			if (isExcluded(absPath, context.excluded)) continue;
			if (context.kind !== "files" && matchesEntry(context, absPath, entry.name)) {
				context.matches.push({ path: absPath, type: "folder" });
				if (context.matches.length >= context.limit) {
					context.stopped = true;
					return;
				}
			}
			await walkDirectory(context, absPath);
			if (context.stopped) return;
			continue;
		}
		if (!entry.isFile()) continue;
		if (isExcluded(absPath, context.excluded)) continue;
		if (context.kind !== "folders" && matchesEntry(context, absPath, entry.name)) {
			context.matches.push({ path: absPath, type: "file" });
			if (context.matches.length >= context.limit) {
				context.stopped = true;
				return;
			}
		}
	}
}

/** Pin configured roots to canonical real paths, dropping missing ones with a diagnostic. */
async function pinConfiguredRoots(
	roots: readonly string[],
	diagnostics: FileNameSearchDiagnostic[],
): Promise<string[]> {
	const pinned: string[] = [];
	const seen = new Set<string>();
	for (const root of roots) {
		if (typeof root !== "string" || root.length === 0) continue;
		const resolvedRoot = resolve(root);
		let real: string;
		try {
			real = realpathSync(resolvedRoot);
		} catch (error) {
			diagnostics.push({
				code: "root-unavailable",
				severity: "warning",
				source: resolvedRoot,
				message: `Configured search root could not be resolved: ${error instanceof Error ? error.message : String(error)}`,
			});
			continue;
		}
		if (seen.has(real)) continue;
		seen.add(real);
		pinned.push(real);
	}
	// Drop roots nested inside another pinned root: their entries are already covered.
	const outermost = pinned
		.slice()
		.sort((a, b) => a.length - b.length)
		.filter((candidate, index, all) => !all.slice(0, index).some((other) => isContained(candidate, other)));
	return outermost;
}

/**
 * Resolve `request.root` to real directories contained in a configured root.
 * Returns `undefined` when the request selects nothing in scope.
 */
async function resolveRequestRoots(requestRoot: string, pinnedRoots: readonly string[]): Promise<string[] | undefined> {
	const candidates = isAbsolute(requestRoot) ? [requestRoot] : pinnedRoots.map((root) => resolve(root, requestRoot));
	const allowed: string[] = [];
	for (const candidate of candidates) {
		let real: string;
		try {
			real = realpathSync(candidate);
		} catch {
			continue;
		}
		if (!pinnedRoots.some((root) => isContained(real, root))) continue;
		if (!allowed.includes(real)) allowed.push(real);
	}
	return allowed.length === 0 ? undefined : allowed;
}

/** Resolve one requested root inside the configured roots for backend routing. */
export async function resolveConfiguredFileSearchRoot(
	roots: readonly string[],
	requestRoot: string,
): Promise<string | undefined> {
	const diagnostics: FileNameSearchDiagnostic[] = [];
	const pinnedRoots = await pinConfiguredRoots(roots ?? [], diagnostics);
	const resolvedRoots = await resolveRequestRoots(requestRoot, pinnedRoots);
	return resolvedRoots?.length === 1 ? resolvedRoots[0] : undefined;
}

/**
 * Keep provider-returned file-name hits inside configured roots and exclusions.
 * Revalidates lstat/realpath so an external index cannot widen MCP scope.
 */
export async function filterFileNameSearchMatches(
	roots: readonly string[],
	candidates: readonly FileNameSearchMatch[],
	excludePaths: readonly string[] = [],
): Promise<FileNameSearchMatch[]> {
	const diagnostics: FileNameSearchDiagnostic[] = [];
	const pinnedRoots = await pinConfiguredRoots(roots ?? [], diagnostics);
	const excluded = (excludePaths ?? []).map(pinExcludedPath);
	const matches: FileNameSearchMatch[] = [];
	for (const candidate of candidates) {
		let stat: Awaited<ReturnType<typeof lstat>>;
		let real: string;
		try {
			stat = await lstat(candidate.path);
			if (stat.isSymbolicLink()) continue;
			real = realpathSync(candidate.path);
		} catch {
			continue;
		}
		if (!pinnedRoots.some((root) => isContained(real, root))) continue;
		if (isExcluded(real, excluded)) continue;
		const segments = real.split(sep);
		if (segments.some((segment) => SKIP_DIR_NAMES[segment] === true)) continue;
		matches.push({ path: real, type: stat.isDirectory() ? "folder" : "file" });
	}
	return matches;
}

export async function searchFileNames(
	roots: readonly string[],
	request: FileNameSearchRequest,
	excludePaths: readonly string[] = [],
): Promise<FileNameSearchResult> {
	const diagnostics: FileNameSearchDiagnostic[] = [];
	const signal = request.signal;

	const pinnedRoots = await pinConfiguredRoots(roots ?? [], diagnostics);
	const excluded = (excludePaths ?? []).map(pinExcludedPath);
	const maxResults = clampInt(request.maxResults, 1, MAX_MAX_RESULTS, DEFAULT_MAX_RESULTS);
	const offset = clampInt(request.offset, 0, Number.MAX_SAFE_INTEGER, 0);

	if (signal?.aborted === true) {
		diagnostics.push({
			code: "search-cancelled",
			severity: "warning",
			source: request.root ?? "",
			message: "The file-name search was cancelled before it completed.",
		});
		return { ok: true, backend: "filesystem", results: [], truncated: false, diagnostics };
	}

	let searchRoots = pinnedRoots;
	if (typeof request.root === "string" && request.root.length > 0) {
		const resolvedRoots = await resolveRequestRoots(request.root, pinnedRoots);
		if (resolvedRoots === undefined) {
			diagnostics.push({
				code: "root-out-of-scope",
				severity: "warning",
				source: request.root,
				message: "The requested search root is not inside any configured search root.",
			});
			return { ok: true, backend: "filesystem", results: [], truncated: false, diagnostics };
		}
		searchRoots = resolvedRoots;
	}
	// A root inside a user exclusion is not searched at all.
	searchRoots = searchRoots.filter((root) => !isExcluded(root, excluded));

	const matches: FileNameSearchMatch[] = [];
	const limit = offset + maxResults + 1;
	for (const rootReal of searchRoots) {
		if (signal?.aborted) break;
		const context: WalkContext = {
			rootReal,
			query: request.query ?? "",
			matchPath: request.matchPath === true,
			matchCase: request.matchCase === true,
			kind: request.kind,
			limit,
			excluded,
			signal,
			matches,
			diagnostics,
			stopped: false,
		};
		await walkDirectory(context, rootReal);
		if (context.stopped) break;
	}

	if (signal?.aborted) {
		diagnostics.push({
			code: "search-cancelled",
			severity: "warning",
			source: request.root ?? "",
			message: "The file-name search was cancelled before it completed.",
		});
	}

	// `limit` carries one lookahead slot, so seeing past `offset + maxResults`
	// proves the page is truncated without a second pass.
	const truncated = matches.length > offset + maxResults;
	const results = matches.slice(offset, offset + maxResults);
	return { ok: true, backend: "filesystem", results, truncated, diagnostics };
}
