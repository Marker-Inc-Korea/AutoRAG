import { randomUUID } from "node:crypto";
import {
	closeSync,
	lstatSync,
	mkdirSync,
	openSync,
	readdirSync,
	readFileSync,
	rmdirSync,
	unlinkSync,
	writeSync,
} from "node:fs";
import { join } from "node:path";

const LOCK_WAIT_BUFFER = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));

interface FileLockOwner {
	readonly token: string;
	readonly pid: number;
	readonly createdAt: number;
}

interface LockFileSnapshot {
	readonly path: string;
	readonly contents: string;
	readonly modifiedAt: number;
	readonly owner?: FileLockOwner;
	readonly device: number;
	readonly inode: number;
}

type LockFileInspection =
	| { readonly kind: "missing" }
	| { readonly kind: "other" }
	| ({ readonly kind: "file" } & LockFileSnapshot);

export interface FileLockOptions {
	readonly timeoutMs: number;
	readonly staleMs: number;
	readonly retryMs?: number;
	readonly timeoutError: () => Error;
}

export interface FileLockHandle {
	readonly path: string;
	readonly contents: string;
	assertOwned(): void;
	release(): void;
}

export class FileLockOwnershipError extends Error {
	constructor(lockPath: string) {
		super(`File lock ownership changed before commit: ${lockPath}`);
		this.name = "FileLockOwnershipError";
	}
}

function hasErrorCode(error: unknown, code: string): boolean {
	return error instanceof Error && "code" in error && error.code === code;
}

function hasAnyErrorCode(error: unknown, codes: readonly string[]): boolean {
	return codes.some((code) => hasErrorCode(error, code));
}

function removeFileIfPresent(path: string): void {
	try {
		unlinkSync(path);
	} catch (error) {
		if (!hasErrorCode(error, "ENOENT")) throw error;
	}
}

function isFileLockOwner(value: unknown): value is FileLockOwner {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
	const record = value as Record<string, unknown>;
	return (
		typeof record.token === "string" &&
		record.token.length > 0 &&
		typeof record.pid === "number" &&
		Number.isInteger(record.pid) &&
		record.pid > 0 &&
		typeof record.createdAt === "number" &&
		Number.isFinite(record.createdAt)
	);
}

function parseFileLockOwner(contents: string): FileLockOwner | undefined {
	try {
		const parsed: unknown = JSON.parse(contents);
		return isFileLockOwner(parsed) ? parsed : undefined;
	} catch (error) {
		if (error instanceof SyntaxError) return undefined;
		throw error;
	}
}

function inspectRegularFile(path: string): LockFileInspection {
	let before: ReturnType<typeof lstatSync>;
	try {
		before = lstatSync(path);
	} catch (error) {
		if (hasErrorCode(error, "ENOENT")) return { kind: "missing" };
		throw error;
	}
	if (!before.isFile()) return { kind: "other" };

	let contents: string;
	try {
		contents = readFileSync(path, "utf8");
	} catch (error) {
		if (hasErrorCode(error, "ENOENT")) return { kind: "missing" };
		if (hasAnyErrorCode(error, ["EISDIR", "EINVAL"])) return { kind: "other" };
		throw error;
	}

	let after: ReturnType<typeof lstatSync>;
	try {
		after = lstatSync(path);
	} catch (error) {
		if (hasErrorCode(error, "ENOENT")) return { kind: "missing" };
		throw error;
	}
	if (
		!after.isFile() ||
		before.dev !== after.dev ||
		before.ino !== after.ino ||
		before.size !== after.size ||
		before.mtimeMs !== after.mtimeMs
	) {
		return { kind: "other" };
	}

	return {
		kind: "file",
		path,
		contents,
		modifiedAt: after.mtimeMs,
		owner: parseFileLockOwner(contents),
		device: after.dev,
		inode: after.ino,
	};
}

function isProcessAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		if (hasErrorCode(error, "ESRCH")) return false;
		if (hasErrorCode(error, "EPERM")) return true;
		throw error;
	}
}

function isStaleSnapshot(snapshot: LockFileSnapshot, staleMs: number): boolean {
	if (Date.now() - snapshot.modifiedAt < staleMs) return false;
	return snapshot.owner === undefined || !isProcessAlive(snapshot.owner.pid);
}

function removeExactSnapshot(snapshot: LockFileSnapshot): boolean {
	const current = inspectRegularFile(snapshot.path);
	if (current.kind === "missing") return true;
	if (
		current.kind !== "file" ||
		current.device !== snapshot.device ||
		current.inode !== snapshot.inode ||
		current.contents !== snapshot.contents
	) {
		return false;
	}

	try {
		unlinkSync(snapshot.path);
		return true;
	} catch (error) {
		if (hasErrorCode(error, "ENOENT")) return true;
		if (hasAnyErrorCode(error, ["EISDIR", "EPERM"])) return false;
		throw error;
	}
}

function removeEmptyLockDirectory(lockPath: string): boolean {
	try {
		rmdirSync(lockPath);
		return true;
	} catch (error) {
		if (hasErrorCode(error, "ENOENT")) return true;
		if (hasAnyErrorCode(error, ["ENOTEMPTY", "EEXIST", "ENOTDIR", "EPERM"])) return false;
		throw error;
	}
}

function reapStaleLockDirectory(lockPath: string, staleMs: number): boolean {
	let names: string[];
	try {
		names = readdirSync(lockPath);
	} catch (error) {
		if (hasErrorCode(error, "ENOENT")) return true;
		if (hasErrorCode(error, "ENOTDIR")) return false;
		throw error;
	}
	if (names.length === 0) return removeEmptyLockDirectory(lockPath);

	const staleSnapshots: LockFileSnapshot[] = [];
	for (const name of names) {
		const inspection = inspectRegularFile(join(lockPath, name));
		if (inspection.kind === "missing") continue;
		if (inspection.kind !== "file" || !isStaleSnapshot(inspection, staleMs)) return false;
		staleSnapshots.push(inspection);
	}
	for (const snapshot of staleSnapshots) {
		if (!removeExactSnapshot(snapshot)) return false;
	}
	return removeEmptyLockDirectory(lockPath);
}

function reapLegacyFileLock(lockPath: string, staleMs: number): boolean {
	const inspection = inspectRegularFile(lockPath);
	if (inspection.kind === "missing") return true;
	if (inspection.kind !== "file" || !isStaleSnapshot(inspection, staleMs)) return false;
	return removeExactSnapshot(inspection);
}

function reapStaleLock(lockPath: string, staleMs: number): boolean {
	let stats: ReturnType<typeof lstatSync>;
	try {
		stats = lstatSync(lockPath);
	} catch (error) {
		if (hasErrorCode(error, "ENOENT")) return true;
		throw error;
	}
	if (stats.isDirectory()) return reapStaleLockDirectory(lockPath, staleMs);
	if (stats.isFile()) return reapLegacyFileLock(lockPath, staleMs);
	return false;
}

function createLockFile(lockPath: string, contents: string): void {
	const descriptor = openSync(lockPath, "wx", 0o600);
	let descriptorOpen = true;
	try {
		writeSync(descriptor, contents, null, "utf8");
		closeSync(descriptor);
		descriptorOpen = false;
	} catch (error) {
		try {
			if (descriptorOpen) closeSync(descriptor);
		} finally {
			removeFileIfPresent(lockPath);
		}
		throw error;
	}
}

function tryCreateOwnerMarker(lockPath: string, owner: FileLockOwner, contents: string): string | undefined {
	try {
		mkdirSync(lockPath, { mode: 0o700 });
	} catch (error) {
		if (hasErrorCode(error, "EEXIST")) return undefined;
		throw error;
	}

	const markerName = `owner-${owner.token}.json`;
	const markerPath = join(lockPath, markerName);
	let acquired = false;
	try {
		try {
			createLockFile(markerPath, contents);
		} catch (error) {
			if (hasAnyErrorCode(error, ["ENOENT", "EEXIST", "ENOTDIR"])) return undefined;
			throw error;
		}

		let names: string[];
		try {
			names = readdirSync(lockPath);
		} catch (error) {
			if (hasAnyErrorCode(error, ["ENOENT", "ENOTDIR"])) return undefined;
			throw error;
		}
		const marker = inspectRegularFile(markerPath);
		if (names.length !== 1 || names[0] !== markerName || marker.kind !== "file" || marker.contents !== contents) {
			return undefined;
		}
		acquired = true;
		return markerPath;
	} finally {
		if (!acquired) {
			removeFileIfPresent(markerPath);
			removeEmptyLockDirectory(lockPath);
		}
	}
}

function createHandle(lockPath: string, markerPath: string, contents: string): FileLockHandle {
	let released = false;
	return {
		path: lockPath,
		contents,
		assertOwned: () => {
			const marker = inspectRegularFile(markerPath);
			if (released || marker.kind !== "file" || marker.contents !== contents) {
				throw new FileLockOwnershipError(lockPath);
			}
		},
		release: () => {
			if (released) return;
			released = true;
			const marker = inspectRegularFile(markerPath);
			if (marker.kind === "file" && marker.contents === contents) removeExactSnapshot(marker);
			removeEmptyLockDirectory(lockPath);
		},
	};
}

export function acquireFileLock(lockPath: string, options: FileLockOptions): FileLockHandle {
	const retryMs = options.retryMs ?? 10;
	const deadline = Date.now() + options.timeoutMs;
	const maxAttempts = Math.ceil(options.timeoutMs / retryMs);
	let attempts = 0;

	for (;;) {
		const owner: FileLockOwner = { token: randomUUID(), pid: process.pid, createdAt: Date.now() };
		const contents = `${JSON.stringify(owner)}\n`;
		const markerPath = tryCreateOwnerMarker(lockPath, owner, contents);
		if (markerPath !== undefined) return createHandle(lockPath, markerPath, contents);

		// Unique marker names are never reused. A stale reaper can unlink only the marker it inspected,
		// and POSIX rmdir cannot remove the directory after a fresh owner has installed its marker.
		if (reapStaleLock(lockPath, options.staleMs)) continue;
		if (Date.now() >= deadline || attempts >= maxAttempts) throw options.timeoutError();
		Atomics.wait(LOCK_WAIT_BUFFER, 0, 0, retryMs);
		attempts++;
	}
}
