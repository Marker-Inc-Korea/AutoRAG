import { stat } from "node:fs/promises";
import { resolve } from "node:path";
import type { DupeyScanResult } from "./cli.ts";

export interface ExactDuplicateFilterResult {
	readonly excluded: ReadonlySet<string>;
	readonly keepers: ReadonlySet<string>;
	readonly errors: readonly string[];
}

/** Selects the newest copy for each exact canonical-text hash.
 *
 * Uses dupey's own latest-copy ranking — the most recent internal document
 * modification time, falling back to filesystem mtime when a document records
 * none — so AutoRAG and dupey never disagree about which copy is newest.
 * Filesystem mtime is consulted only when dupey supplies no ranking.
 */
export async function selectExactDuplicateExclusions(
	root: string,
	scan: DupeyScanResult,
	isUnavailable?: (path: string) => boolean,
): Promise<ExactDuplicateFilterResult> {
	const byHash = new Map<string, string[]>();
	for (const file of scan.files) {
		if (typeof file.content_hash !== "string" || file.content_hash.length === 0) continue;
		const path = resolve(root, file.path);
		const group = byHash.get(file.content_hash) ?? [];
		group.push(path);
		byHash.set(file.content_hash, group);
	}
	const dupeyRanking = dupeyExactRanking(scan);
	const excluded = new Set<string>();
	const keepers = new Set<string>();
	const errors = [...scan.errors.map((error) => JSON.stringify(error))];
	for (const paths of byHash.values()) {
		if (paths.length < 2) continue;
		const ranked = await rankExactCandidates(paths, dupeyRanking, errors);
		// User-excluded copies are handled by the caller's own exclusion pass; they
		// must not win keeper selection and drag the remaining copies out of the index.
		const candidates = isUnavailable === undefined ? ranked : ranked.filter((path) => !isUnavailable(path));
		const keeper = candidates[0];
		if (!keeper) continue;
		keepers.add(keeper);
		for (const candidate of candidates.slice(1)) excluded.add(candidate);
	}
	return { excluded, keepers, errors };
}

/** Absolute path → rank index derived from dupey's exact-family latest-copy picks. */
function dupeyExactRanking(scan: DupeyScanResult): Map<string, number> {
	const ranking = new Map<string, number>();
	for (const family of scan.families) {
		if (family.relation !== "exact") continue;
		const ranked: unknown = family.pick?.ranked;
		if (!Array.isArray(ranked)) continue;
		const entries: readonly unknown[] = ranked;
		entries.forEach((entry, index) => {
			if (!entry || typeof entry !== "object" || !("path" in entry)) return;
			const path = entry.path;
			if (typeof path === "string") ranking.set(resolve(path), index);
		});
	}
	return ranking;
}

async function rankExactCandidates(
	paths: readonly string[],
	dupeyRanking: ReadonlyMap<string, number>,
	errors: string[],
): Promise<string[]> {
	if (paths.every((path) => dupeyRanking.has(path))) {
		return [...paths].sort((a, b) => (dupeyRanking.get(a) ?? 0) - (dupeyRanking.get(b) ?? 0) || a.localeCompare(b));
	}
	const ranked = await Promise.all(
		paths.map(async (path) => {
			try {
				const info = await stat(path);
				return { path, mtimeMs: info.mtimeMs };
			} catch {
				errors.push("dupey exact duplicate path disappeared before indexing.");
				return { path, mtimeMs: -1 };
			}
		}),
	);
	ranked.sort((a, b) => b.mtimeMs - a.mtimeMs || a.path.localeCompare(b.path));
	return ranked.map(({ path }) => path);
}
