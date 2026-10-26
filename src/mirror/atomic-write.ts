import { randomUUID } from "node:crypto";
import { mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

/**
 * Errors a rename raises when the destination is momentarily held by someone else. On Windows
 * `MoveFileEx(REPLACE_EXISTING)` reports that as `ERROR_ACCESS_DENIED` (Node: `EPERM`) while an editor,
 * antivirus scan, search indexer, or another AutoRAG process has the destination open without delete-sharing.
 */
const RETRYABLE_RENAME_CODES: Record<string, true> = { EPERM: true, EACCES: true, EBUSY: true };
const DEFAULT_RENAME_RETRIES = 5;
const DEFAULT_RENAME_BACKOFF_MS = 25;

export interface RenameRetryOptions {
	/** Retries after the first attempt. Default 5, so six attempts in total. */
	readonly retries?: number;
	/** First wait in milliseconds; it doubles each retry (25/50/100/200/400 by default). */
	readonly backoffMs?: number;
}

/** True for a rename error that a short wait can clear (`EPERM`, `EACCES`, `EBUSY`). */
export function isRetryableRenameError(error: unknown): boolean {
	if (typeof error !== "object" || error === null || !("code" in error)) return false;
	return typeof error.code === "string" && Object.hasOwn(RETRYABLE_RENAME_CODES, error.code);
}

/**
 * `renameSync` that waits and retries while the destination is transiently held (see
 * {@link RETRYABLE_RENAME_CODES}). Any other error, and the last retryable one, is rethrown unchanged.
 */
export function renameWithRetry(from: string, to: string, options: RenameRetryOptions = {}): void {
	const retries = options.retries ?? DEFAULT_RENAME_RETRIES;
	const backoffMs = options.backoffMs ?? DEFAULT_RENAME_BACKOFF_MS;
	for (let attempt = 0; ; attempt += 1) {
		try {
			renameSync(from, to);
			return;
		} catch (error) {
			if (attempt >= retries || !isRetryableRenameError(error)) throw error;
			sleepSync(backoffMs * 2 ** attempt);
		}
	}
}

function sleepSync(milliseconds: number): void {
	if (milliseconds <= 0) return;
	Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
}

/**
 * Write `content` to `path` through a sibling temp file and an atomic rename.
 * A destination that stays held through every rename retry is overwritten in place instead: replacing by
 * rename needs delete rights on the destination, writing to an existing file does not.
 * On any other failure the temp file is removed before the error is rethrown, so a full
 * disk or an I/O error never leaves `*.tmp` orphans in the mirror.
 */
export function writeFileAtomic(path: string, content: string): void {
	mkdirSync(dirname(path), { recursive: true });
	const tmp = `${path}.${randomUUID()}.tmp`;
	try {
		writeFileSync(tmp, content);
		try {
			renameWithRetry(tmp, path);
		} catch (error) {
			if (!isRetryableRenameError(error)) throw error;
			writeFileSync(path, content);
			rmSync(tmp, { force: true });
		}
	} catch (error) {
		try {
			rmSync(tmp, { force: true });
		} catch {
			// Best effort: the original write error is the one worth reporting.
		}
		throw error;
	}
}
