import { readFileSync, rmSync } from "node:fs";
import { writeFileAtomic } from "../mirror/atomic-write.ts";
import { refreshProgressPath } from "../mirror/paths.ts";

/** Persist display progress every 25 files; this is not the engine's recovery checkpoint. */
export const REFRESH_PROGRESS_EVERY = 25;

export type RefreshProgressPhase =
	| "parsed"
	| "minsync"
	| "datasources"
	| "jikji"
	| "everything"
	| "fsearch"
	| "finalizing";

export interface RefreshProgressCounts {
	readonly scanned: number;
	readonly written: number;
	readonly deleted: number;
	readonly skipped: number;
}

export interface PersistedRefreshProgress {
	readonly version: 1;
	readonly runId: string;
	readonly pid: number;
	readonly state: "running" | "success" | "failed";
	readonly phase: RefreshProgressPhase;
	readonly startedAt: string;
	readonly updatedAt: string;
	readonly finishedAt?: string;
	readonly sourceFiles?: { readonly total: number };
	readonly parsedCounts?: RefreshProgressCounts;
	readonly minsync?: { readonly synced?: number; readonly total?: number };
	readonly error?: string;
}

export function readRefreshProgress(root: string): PersistedRefreshProgress | undefined {
	try {
		const parsed: unknown = JSON.parse(readFileSync(refreshProgressPath(root), "utf8"));
		return isPersistedRefreshProgress(parsed) ? parsed : undefined;
	} catch (error) {
		if (
			error instanceof SyntaxError ||
			(typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT")
		) {
			return undefined;
		}
		throw error;
	}
}

export function writeRefreshProgress(root: string, progress: PersistedRefreshProgress): void {
	writeFileAtomic(refreshProgressPath(root), `${JSON.stringify(progress, null, 2)}\n`);
}

export function updateRefreshProgress(
	progress: PersistedRefreshProgress,
	update: Partial<Omit<PersistedRefreshProgress, "version" | "runId" | "pid" | "startedAt" | "updatedAt">>,
): PersistedRefreshProgress {
	return {
		...progress,
		...update,
		updatedAt: new Date().toISOString(),
	};
}

export function removeRefreshProgress(root: string): void {
	rmSync(refreshProgressPath(root), { force: true });
}

export function isRefreshOwnerAlive(progress: PersistedRefreshProgress): boolean {
	if (progress.state !== "running") return false;
	try {
		process.kill(progress.pid, 0);
		return true;
	} catch (error) {
		if (error instanceof Error) return false;
		throw error;
	}
}

function isPersistedRefreshProgress(value: unknown): value is PersistedRefreshProgress {
	if (!isRecord(value)) return false;
	const record = value;
	return (
		record.version === 1 &&
		typeof record.runId === "string" &&
		typeof record.pid === "number" &&
		Number.isInteger(record.pid) &&
		(record.state === "running" || record.state === "success" || record.state === "failed") &&
		isRefreshProgressPhase(record.phase) &&
		typeof record.startedAt === "string" &&
		typeof record.updatedAt === "string" &&
		(record.finishedAt === undefined || typeof record.finishedAt === "string") &&
		(record.error === undefined || typeof record.error === "string") &&
		isSourceFiles(record.sourceFiles) &&
		isParsedCounts(record.parsedCounts) &&
		isMinsync(record.minsync)
	);
}

function isRefreshProgressPhase(value: unknown): value is RefreshProgressPhase {
	return (
		value === "parsed" ||
		value === "minsync" ||
		value === "datasources" ||
		value === "jikji" ||
		value === "everything" ||
		value === "fsearch" ||
		value === "finalizing"
	);
}

function isSourceFiles(value: unknown): boolean {
	return value === undefined || (isRecord(value) && typeof value.total === "number" && Number.isInteger(value.total));
}

function isParsedCounts(value: unknown): boolean {
	return (
		value === undefined ||
		(isRecord(value) &&
			typeof value.scanned === "number" &&
			typeof value.written === "number" &&
			typeof value.deleted === "number" &&
			typeof value.skipped === "number")
	);
}

function isMinsync(value: unknown): boolean {
	return (
		value === undefined ||
		(isRecord(value) &&
			(value.synced === undefined ||
				(typeof value.synced === "number" && Number.isInteger(value.synced) && value.synced >= 0)) &&
			(value.total === undefined ||
				(typeof value.total === "number" && Number.isInteger(value.total) && value.total >= 0)))
	);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}
