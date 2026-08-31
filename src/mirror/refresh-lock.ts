import { lstatSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import { AUTORAG_DIR } from "./paths.ts";

/**
 * Cross-process mutual exclusion for the refresh pipeline.
 *
 * An in-process guard is not enough. The CLI builds a fresh agent in a fresh process for every
 * command, so the realistic collision is `autorag watch` in one terminal and `autorag refresh` in
 * another: two processes, one index directory. Without a lock on disk they can interleave their
 * artifact and fingerprint commits, leaving a fingerprint that describes a newer corpus than the
 * artifact it points at. The next refresh then matches that fingerprint and silently keeps the
 * stale artifact.
 *
 * The locking primitive is `acquireFileLock`, the same one used for the memory store and the pi
 * models file. It takes a lock directory containing a uniquely named owner marker, so a stale
 * reaper can only unlink the exact marker it inspected and `rmdir` cannot succeed once a new owner
 * has installed its own. A hand-rolled read-then-unlink scheme cannot offer that guarantee.
 */

/**
 * Lock path for a workspace root: directly under `.autorag`, *not* inside any index directory.
 *
 * Placement is load-bearing. `index reset` deletes whole index subtrees with a recursive `rmSync`,
 * so a lock stored inside one of them would be destroyed by the very operation it must exclude —
 * the running refresh would keep a handle to a directory that no longer exists and a third process
 * could then take the "free" lock. `.autorag` itself is never removed by reset, so the lock
 * survives every scoped reset and stays a single rendezvous point for both writers.
 */
export function refreshLockPath(root: string): string {
	return `${join(root, AUTORAG_DIR, "refresh")}.lock`;
}

/**
 * Refresh is not queued behind another refresh: a caller that cannot take the lock is told the
 * pipeline is busy and returns immediately. Waiting would stack watch ticks behind a long rebuild
 * for no benefit, since the running refresh already covers the newer corpus.
 */
const REFRESH_LOCK_WAIT_TIMEOUT_MS = 0;

/**
 * How long a lock may sit untouched before a reaper may consider it abandoned.
 *
 * The reaper also requires the recorded owner process to be gone, so this bound only decides how
 * long a *dead* owner's lock lingers; a live owner is never reaped regardless of age. It is set far
 * above any refresh observed here, where the heaviest cold rebuild on the largest synthetic corpus
 * finishes in under two seconds. Reclaiming too early would reintroduce the corruption this lock
 * prevents, so the bias is deliberate.
 */
const REFRESH_LOCK_STALE_MS = 10 * 60 * 1000;

export class RefreshBusyError extends Error {
	constructor() {
		super("A refresh is already running for this workspace");
		this.name = "RefreshBusyError";
	}
}

/** True when the filesystem entry does not exist. */
function isMissingEntryError(error: unknown): boolean {
	return error instanceof Error && "code" in error && error.code === "ENOENT";
}

/** Take the refresh lock, or return undefined when another refresh already holds it. */
export function acquireRefreshLock(root: string): FileLockHandle | undefined {
	// `.autorag` may not exist yet on a first refresh; the lock lives directly under it.
	mkdirSync(join(root, AUTORAG_DIR), { recursive: true });

	// A refresh lock is a directory containing an owner marker. Anything else at the lock path is a
	// structural anomaly, not contention: `acquireFileLock`'s legacy-file reaper would unlink a
	// stale-looking regular file (deleting whatever the user put there), and a regular file that
	// parses as an owner with a live pid would read as contention forever. Refuse loudly, leave the
	// entry untouched, and tell the user exactly what to fix.
	const lockPath = refreshLockPath(root);
	let stats: ReturnType<typeof lstatSync> | undefined;
	try {
		stats = lstatSync(lockPath);
	} catch (error) {
		if (!isMissingEntryError(error)) throw error;
	}
	if (stats !== undefined && !stats.isDirectory()) {
		throw new Error(
			`The refresh lock at ${join(AUTORAG_DIR, "refresh.lock")} exists but is not a lock directory. ` +
				"A lock is a directory with an owner marker; move the file aside or delete it, then run the command again.",
		);
	}

	try {
		return acquireFileLock(lockPath, {
			timeoutMs: REFRESH_LOCK_WAIT_TIMEOUT_MS,
			staleMs: REFRESH_LOCK_STALE_MS,
			timeoutError: () => new RefreshBusyError(),
		});
	} catch (error) {
		// Contention is the only expected failure. Anything else (permissions, a full disk) is a real
		// fault and must surface rather than masquerade as a busy pipeline.
		if (error instanceof RefreshBusyError) return undefined;
		throw error;
	}
}
