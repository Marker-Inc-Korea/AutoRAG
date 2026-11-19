import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

export const PARSED_MIRROR_SUBDIR = join(".autorag", "parsed");
export const PARSED_FILES_SUBDIR = "files";
export const REFRESH_READINESS_FILE = "refresh-complete.json";
export const REFRESH_PROGRESS_FILE = "refresh-progress.json";

export interface PersistedDuplicateScanWarning {
	readonly code: "duplicate-scan-failed";
	readonly severity: "warning";
	readonly source: string;
	readonly reason: "timeout" | "scan-failed";
	readonly message: string;
}

export function parsedMirrorRoot(root: string): string {
	return join(root, PARSED_MIRROR_SUBDIR);
}

export function parsedOutputPath(root: string, virtualPath: string): string {
	const digest = createHash("sha256").update(virtualPath).digest("hex");
	return join(parsedMirrorRoot(root), PARSED_FILES_SUBDIR, `${digest}.md`);
}

export function refreshReadinessPath(root: string): string {
	return join(root, ".autorag", REFRESH_READINESS_FILE);
}

export function refreshProgressPath(root: string): string {
	return join(root, ".autorag", REFRESH_PROGRESS_FILE);
}
/**
 * Whether a parsed-mirror refresh has completed at least once for the given
 * workspace. Reads the persisted readiness marker so a separate process (CLI,
 * MCP server) reaches the same verdict as the process that ran the refresh.
 */
export function isParsedRefreshComplete(workspacePath: string): boolean {
	const markerPath = refreshReadinessPath(workspacePath);
	if (!existsSync(markerPath)) return false;
	try {
		const marker: unknown = JSON.parse(readFileSync(markerPath, "utf8"));
		return (
			typeof marker === "object" &&
			marker !== null &&
			"version" in marker &&
			marker.version === 1 &&
			"completed" in marker &&
			marker.completed === true &&
			"parsed" in marker &&
			marker.parsed === true
		);
	} catch {
		return false;
	}
}

/**
 * Record completion of a parsed-mirror refresh, along with any `duplicate-scan-failed`
 * warnings from that run so a separate `status` process can surface them without running
 * dupey. A subsequent completed parsed refresh overwrites the marker, clearing recovered roots.
 */
export function writeParsedRefreshReadiness(
	workspacePath: string,
	duplicateScanWarnings: readonly PersistedDuplicateScanWarning[] = [],
): void {
	const markerPath = refreshReadinessPath(workspacePath);
	mkdirSync(dirname(markerPath), { recursive: true });
	const payload =
		duplicateScanWarnings.length === 0
			? { version: 1, completed: true, parsed: true }
			: { version: 1, completed: true, parsed: true, duplicateScanWarnings };
	writeFileSync(markerPath, `${JSON.stringify(payload)}\n`);
}

function isPersistedDuplicateScanWarning(value: unknown): value is PersistedDuplicateScanWarning {
	if (typeof value !== "object" || value === null) return false;
	const record = value as Record<string, unknown>;
	return (
		record.code === "duplicate-scan-failed" &&
		record.severity === "warning" &&
		typeof record.source === "string" &&
		(record.reason === "timeout" || record.reason === "scan-failed") &&
		typeof record.message === "string"
	);
}

/**
 * Read `duplicate-scan-failed` warnings persisted by the last completed parsed refresh.
 * Returns an empty array when the marker is absent, unreadable, or recorded no warnings.
 */
export function readPersistedDuplicateScanWarnings(workspacePath: string): readonly PersistedDuplicateScanWarning[] {
	const markerPath = refreshReadinessPath(workspacePath);
	if (!existsSync(markerPath)) return [];
	try {
		const marker: unknown = JSON.parse(readFileSync(markerPath, "utf8"));
		if (typeof marker !== "object" || marker === null || !("duplicateScanWarnings" in marker)) return [];
		const warnings = (marker as { readonly duplicateScanWarnings?: unknown }).duplicateScanWarnings;
		if (!Array.isArray(warnings)) return [];
		return warnings.filter(isPersistedDuplicateScanWarning);
	} catch {
		return [];
	}
}
