import { stat } from "node:fs/promises";
import { resolve } from "node:path";
import type { DupeyScanResult } from "./cli.ts";

export interface ExactDuplicateFilterResult {
	readonly excluded: ReadonlySet<string>;
	readonly keepers: ReadonlySet<string>;
	readonly errors: readonly string[];
}

/** Selects the newest filesystem copy for each exact canonical-text hash. */
export async function selectExactDuplicateExclusions(
	root: string,
	scan: DupeyScanResult,
): Promise<ExactDuplicateFilterResult> {
	const byHash = new Map<string, string[]>();
	for (const file of scan.files) {
		if (typeof file.content_hash !== "string" || file.content_hash.length === 0) continue;
		const path = resolve(root, file.path);
		const group = byHash.get(file.content_hash) ?? [];
		group.push(path);
		byHash.set(file.content_hash, group);
	}
	const excluded = new Set<string>();
	const keepers = new Set<string>();
	const errors = [...scan.errors.map((error) => JSON.stringify(error))];
	for (const paths of byHash.values()) {
		if (paths.length < 2) continue;
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
		const keeper = ranked[0]?.path;
		if (!keeper) continue;
		keepers.add(keeper);
		for (const candidate of ranked.slice(1)) excluded.add(candidate.path);
	}
	return { excluded, keepers, errors };
}

