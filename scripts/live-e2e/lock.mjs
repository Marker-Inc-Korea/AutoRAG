/**
 * scripts/live-e2e/lock.mjs — Filesystem-exclusive lock for live-e2e.
 *
 * Extracted from env.mjs so the lock/fingerprint concerns stay separable
 * and env.mjs stays under the size gate. Exports are identical to what
 * env.mjs re-exports for runner.mjs and tests.
 *
 * This is a plain JS ESM module (.mjs).  Type annotations use JSDoc.
 */

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = resolve(__dirname, "..", "..");
const E2E_DIR = join(REPO_ROOT, ".autorag-e2e");

const LOCK_DIR = join(E2E_DIR, "locks");
const LOCK_FILE = join(LOCK_DIR, "live-e2e.lock");

/**
 * @typedef {'acquired' | { kind: 'held', reason: string }} LockResult
 */

/**
 * Filesystem-based exclusive lock using mkdir atomicity with stale detection.
 *
 * mkdirSync on an existing path throws EEXIST — only one process can create
 * a given directory atomically.  The lock is a fixed-name directory.
 * The winner writes its PID inside for diagnostic purposes.
 *
 * **Stale lock recovery**: if the lock exists and the PID inside is dead
 * (via kill(pid, 0)), the stale lock is removed and acquisition retried.
 * This prevents a SIGKILL'd holder from permanently blocking the lock.
 *
 * @param {number} [holdMs=0] - How long (ms) to hold the lock before releasing.
 * @returns {{ kind: 'acquired' } | { kind: 'held', reason: string }}
 */
export function tryLock(holdMs = 0) {
	// Ensure lock root exists
	mkdirSync(LOCK_DIR, { recursive: true });

	// Check for and break stale locks before attempting acquisition.
	// If breakStaleLock removes the lock, the next mkdirSync will succeed.
	for (let attempt = 0; attempt < 2; attempt++) {
		breakStaleLock();

		try {
			// Atomic test-and-set on a SINGLE fixed name.
			mkdirSync(LOCK_FILE);

			// Write PID for diagnostics (best-effort)
			try {
				writeFileSync(join(LOCK_FILE, "pid"), String(process.pid));
			} catch { /* ignore — lock is ours regardless */ }

			// Hold if requested
			if (holdMs > 0) {
				spawnSync("sleep", [String(Math.ceil(holdMs / 1000))], {
					stdio: "inherit",
					timeout: holdMs + 5000,
				});
			}

			// Release: remove the lock directory
			rmSync(LOCK_FILE, { recursive: true, force: true });

			return { kind: "acquired" };
		} catch (/** @type {unknown} */ e) {
			// mkdir failed — EEXIST, lock held by another process.
			// Continue to next attempt (second attempt will break stale lock again).
		}
	}

	// Both attempts failed — genuinely held by a live process.
	let heldBy = "unknown";
	try {
		const pidPath = join(LOCK_FILE, "pid");
		if (existsSync(pidPath)) {
			heldBy = readFileSync(pidPath, "utf-8").trim();
		}
	} catch { /* ignore */ }
	return { kind: "held", reason: "live-e2e-lock-held: lock held by pid " + heldBy };
}

/**
 * Check if an existing lock directory holds a stale PID.
 * If so, remove the lock directory so a new acquirer can create it.
 *
 * Uses kill(pid, 0): returns 0 if the process exists, -1 with ESRCH if not.
 * This is safe — we never send a signal, just probe existence.
 *
 * The lock is considered stale (and broken) when:
 * - PID file exists, PID is numeric, valid, and kill(pid, 0) reports dead
 * - PID equals our own process PID (restart/crash)
 *
 * The lock is NOT broken when:
 * - No PID file (lock creation is in progress — extremely short window)
 * - PID is alive
 *
 * Missing-PID conservatism accepts a <1ms orphan window where a newly
 * created lock has no PID yet; the next caller will see it and break it
 * if the PID is never written (crashed creator).
 */
function breakStaleLock() {
	if (!existsSync(LOCK_FILE)) {
		return;
	}

	// Read PID from the lock directory
	let pid = 0;
	try {
		const pidPath = join(LOCK_FILE, "pid");
		if (existsSync(pidPath)) {
			pid = Number(readFileSync(pidPath, "utf-8").trim());
		} else {
			// No PID file yet — holder is in the <1ms window between mkdir
			// and writeFileSync.  Conservatively DO NOT break.
			return;
		}
	} catch {
		// Unreadable — conservatively don't break
		return;
	}

	// Invalid PID
	if (!Number.isFinite(pid) || pid <= 0) {
		rmSync(LOCK_FILE, { recursive: true, force: true });
		return;
	}

	// Our own PID — previous incarnation crashed / restarted
	if (pid === process.pid) {
		rmSync(LOCK_FILE, { recursive: true, force: true });
		return;
	}

	// Probe if the process exists using kill(pid, 0)
	try {
		const result = spawnSync("kill", ["-0", String(pid)], {
			stdio: "ignore",
			timeout: 1000,
		});
		if (result.status !== 0) {
			// Process does not exist — stale lock, break it
			rmSync(LOCK_FILE, { recursive: true, force: true });
		}
	} catch {
		// kill command failed — can't probe, conservatively leave lock
	}
}