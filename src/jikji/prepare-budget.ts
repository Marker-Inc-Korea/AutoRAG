import type { Dirent } from "node:fs";
import { readdir } from "node:fs/promises";
import { join } from "node:path";

/**
 * Largest delay `setTimeout` honors (2^31 - 1 ms, ~24.8 days); Node fires a
 * larger delay after 1 ms. This is a platform limit, not a policy cap.
 */
export const MAX_TIMER_MS = 2_147_483_647;

/**
 * `base + perFile * fileCount`. There is deliberately no policy ceiling: the
 * budget grows with the root, so a huge root is never cut by a fixed wall while
 * a hung run on a small root still is. Clamped only to the timer limit.
 */
export function scaledPrepareTimeoutMs(baseMs: number, perFileMs: number, fileCount: number): number {
	return Math.min(baseMs + perFileMs * fileCount, MAX_TIMER_MS);
}

/**
 * Count regular files under `root`. Symlinks are not followed and hidden
 * entries are skipped unless `includeHidden` (matching what `jikji prepare`
 * indexes by default). `exclude` globs are ignored: over-counting only
 * lengthens the budget. Stops early when `signal` aborts.
 */
export async function countFiles(root: string, includeHidden: boolean, signal?: AbortSignal): Promise<number> {
	let count = 0;
	const pending = [root];
	while (pending.length > 0 && signal?.aborted !== true) {
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
			else if (entry.isFile()) count += 1;
		}
	}
	return count;
}
