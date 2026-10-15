import type { Dirent } from "node:fs";
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { DEFAULT_CONNECTOR_SYNC_TIMEOUT_MS } from "../datasource/connector-timeouts.ts";

/** Upper bound for a size-scaled `jikji prepare` budget: the connector sync budget (30 min). */
export const MAX_PREPARE_TIMEOUT_MS = DEFAULT_CONNECTOR_SYNC_TIMEOUT_MS;

/** `base + perFile * fileCount`, capped at {@link MAX_PREPARE_TIMEOUT_MS} (never below `base`). */
export function scaledPrepareTimeoutMs(baseMs: number, perFileMs: number, fileCount: number): number {
	const cap = Math.max(baseMs, MAX_PREPARE_TIMEOUT_MS);
	return Math.min(baseMs + perFileMs * fileCount, cap);
}

/** Files that make the scaled budget reach its cap; counting past this is wasted work. */
export function filesToReachCap(baseMs: number, perFileMs: number): number {
	const cap = Math.max(baseMs, MAX_PREPARE_TIMEOUT_MS);
	return Math.ceil((cap - baseMs) / perFileMs);
}

/**
 * Count regular files under `root`, stopping once `limit` is reached so a huge
 * tree costs at most `limit` entries of walking. Symlinks are not followed and
 * hidden entries are skipped unless `includeHidden` (matching what `jikji
 * prepare` indexes by default). `exclude` globs are ignored: over-counting only
 * lengthens the budget.
 */
export async function countFilesBounded(root: string, limit: number, includeHidden: boolean): Promise<number> {
	let count = 0;
	const pending = [root];
	while (pending.length > 0 && count < limit) {
		const directory = pending.pop() as string;
		let entries: Dirent[];
		try {
			entries = await readdir(directory, { withFileTypes: true });
		} catch {
			continue;
		}
		for (const entry of entries) {
			if (!includeHidden && entry.name.startsWith(".")) continue;
			if (entry.isDirectory()) pending.push(join(directory, entry.name));
			else if (entry.isFile() && ++count >= limit) return count;
		}
	}
	return count;
}
