import { randomUUID } from "node:crypto";
import { mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

/**
 * Write `content` to `path` through a sibling temp file and an atomic rename.
 * On any failure the temp file is removed before the error is rethrown, so a full
 * disk or an I/O error never leaves `*.tmp` orphans in the mirror.
 */
export function writeFileAtomic(path: string, content: string): void {
	mkdirSync(dirname(path), { recursive: true });
	const tmp = `${path}.${randomUUID()}.tmp`;
	try {
		writeFileSync(tmp, content);
		renameSync(tmp, path);
	} catch (error) {
		try {
			rmSync(tmp, { force: true });
		} catch {
			// Best effort: the original write error is the one worth reporting.
		}
		throw error;
	}
}
