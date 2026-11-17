import { lstatSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";

const REFRESH_LOCK_WAIT_TIMEOUT_MS = 0;
const REFRESH_LOCK_STALE_MS = 10 * 60 * 1000;

export function refreshLockPath(root: string): string {
	return join(root, ".autorag", "refresh.lock");
}

export class RefreshBusyError extends Error {
	constructor() {
		super("A refresh is already running for this workspace");
		this.name = "RefreshBusyError";
	}
}

function isMissingEntryError(error: unknown): boolean {
	return error instanceof Error && "code" in error && error.code === "ENOENT";
}

/** Acquire the cross-process refresh lock, or return undefined when another refresh owns it. */
export function acquireRefreshLock(root: string): FileLockHandle | undefined {
	mkdirSync(join(root, ".autorag"), { recursive: true });
	const lockPath = refreshLockPath(root);
	let stats: ReturnType<typeof lstatSync> | undefined;
	try {
		stats = lstatSync(lockPath);
	} catch (error) {
		if (!isMissingEntryError(error)) throw error;
	}
	if (stats !== undefined && !stats.isDirectory()) {
		throw new Error(
			"The refresh lock at .autorag/refresh.lock exists but is not a lock directory. Move it aside or delete it, then run the command again.",
		);
	}
	try {
		return acquireFileLock(lockPath, {
			timeoutMs: REFRESH_LOCK_WAIT_TIMEOUT_MS,
			staleMs: REFRESH_LOCK_STALE_MS,
			timeoutError: () => new RefreshBusyError(),
		});
	} catch (error) {
		if (error instanceof RefreshBusyError) return undefined;
		throw error;
	}
}
