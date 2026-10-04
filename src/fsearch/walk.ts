import type { Dirent } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import { basename, join, sep } from "node:path";
import type { FSearchEntry, FSearchSearchRequest, FSearchSort } from "./client.ts";

/**
 * Graceful-degradation backend for hosts without fsearch-cli: a bounded,
 * slow filesystem walk over the configured search folders. It supports a
 * deliberate subset of the fsearch-cli surface — substring (or regex) name
 * matching, kind/path filters, and in-memory sorting — and skips AutoRAG
 * state directories (`.autorag`) the same way the index excludes them.
 */

export interface FSearchWalkOptions {
	/** Directory entries to visit before giving up. Default 500k. */
	readonly maxVisited?: number;
	/** Wall-clock budget. Default 20s. */
	readonly deadlineMs?: number;
	/** Matches to collect before giving up. Default 10k. */
	readonly maxMatches?: number;
}

export interface FSearchWalkResult {
	readonly entries: FSearchEntry[];
	readonly visited: number;
	readonly truncated: boolean;
}

const DEFAULT_MAX_VISITED = 500_000;
const DEFAULT_DEADLINE_MS = 20_000;
const DEFAULT_MAX_MATCHES = 10_000;

const SORT_ORDERS: Record<FSearchSort, (a: FSearchEntry, b: FSearchEntry) => number> = {
	"name-ascending": (a, b) => compare(a.name, b.name),
	"name-descending": (a, b) => compare(b.name, a.name),
	"path-ascending": (a, b) => compare(a.path, b.path),
	"path-descending": (a, b) => compare(b.path, a.path),
	"size-ascending": (a, b) => (a.size ?? -1) - (b.size ?? -1),
	"size-descending": (a, b) => (b.size ?? -1) - (a.size ?? -1),
	"date-modified-ascending": (a, b) => compare(a.dateModified ?? "", b.dateModified ?? ""),
	"date-modified-descending": (a, b) => compare(b.dateModified ?? "", a.dateModified ?? ""),
};

function compare(a: string, b: string): number {
	return a < b ? -1 : a > b ? 1 : 0;
}

export async function walkFileSearch(
	folders: readonly string[],
	request: FSearchSearchRequest,
	options: FSearchWalkOptions = {},
): Promise<FSearchWalkResult> {
	const maxVisited = options.maxVisited ?? DEFAULT_MAX_VISITED;
	const maxMatches = options.maxMatches ?? DEFAULT_MAX_MATCHES;
	const deadline = Date.now() + (options.deadlineMs ?? DEFAULT_DEADLINE_MS);
	const matches: (path: string) => boolean = request.regex
		? (() => {
				const pattern = new RegExp(request.query, request.matchCase ? "" : "i");
				return (path) => pattern.test(request.matchPath ? path : basename(path));
			})()
		: (() => {
				const needle = request.matchCase ? request.query : request.query.toLowerCase();
				return (path) => {
					const haystack = request.matchPath ? path : basename(path);
					return (request.matchCase ? haystack : haystack.toLowerCase()).includes(needle);
				};
			})();
	const pathPrefix =
		request.path === undefined ? undefined : request.path.endsWith(sep) ? request.path : `${request.path}${sep}`;
	const underPath = (path: string) => pathPrefix === undefined || path === request.path || path.startsWith(pathPrefix);

	const collected: FSearchEntry[] = [];
	let visited = 0;
	let truncated = false;
	const pending: string[] = [...folders];
	outer: while (pending.length > 0) {
		const dir = pending.pop()!;
		if (Date.now() >= deadline) {
			truncated = true;
			break;
		}
		let dirents: Dirent[];
		try {
			dirents = await readdir(dir, { withFileTypes: true });
		} catch {
			continue;
		}
		for (const dirent of dirents) {
			if (visited >= maxVisited) {
				truncated = true;
				break outer;
			}
			visited += 1;
			const path = join(dir, dirent.name);
			if (dirent.isDirectory()) {
				if (dirent.name === ".autorag") continue;
				pending.push(path);
			}
			if (!underPath(path)) continue;
			if (request.kind === "folders" && !dirent.isDirectory()) continue;
			if (request.kind === "files" && dirent.isDirectory()) continue;
			if (!matches(path)) continue;
			try {
				const info = await stat(path);
				if (request.kind === "folders" && !info.isDirectory()) continue;
				if (request.kind === "files" && info.isDirectory()) continue;
				collected.push({
					path,
					name: basename(path),
					type: info.isDirectory() ? "folder" : "file",
					size: info.isDirectory() ? undefined : info.size,
					dateModified: info.mtime.toISOString(),
				});
			} catch {
				// Vanished or unreadable between readdir and stat; skip it.
			}
			if (collected.length >= maxMatches) {
				truncated = true;
				break outer;
			}
		}
	}
	// The root folders themselves are candidates too (fsearch-cli indexes the
	// included roots), but they carry no parent dirent, so test them here.
	for (const folder of folders) {
		if (!underPath(folder) || !matches(folder)) continue;
		if (request.kind === "files") continue;
		try {
			const info = await stat(folder);
			if (!info.isDirectory()) continue;
			collected.push({
				path: folder,
				name: basename(folder),
				type: "folder",
				size: undefined,
				dateModified: info.mtime.toISOString(),
			});
		} catch {
			// Unreadable root; skip it.
		}
	}
	if (request.sort !== undefined) collected.sort(SORT_ORDERS[request.sort]);
	const offset = request.offset ?? 0;
	const maxResults = request.maxResults ?? 100;
	const entries = collected.slice(offset, offset + maxResults);
	return { entries, visited, truncated };
}
