import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { parsedMirrorIndexPath } from "./paths.ts";

export interface ParsedMirrorEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly outputPath: string;
	readonly parserName: string;
	readonly sourceMtimeNs: number;
	readonly sourceSizeBytes: number;
	readonly updatedAt: string;
	/** SHA-256 of the normalized parsed markdown written to `outputPath`. */
	readonly contentSha256?: string;
}

/** Why refresh deliberately left a source out of the parsed mirror. */
export type ParsedMirrorSkipReason = "duplicate-excluded" | "parser-skipped" | "parser-failed";

/**
 * A source the last refresh decided not to mirror, pinned to the exact version it
 * decided on. Staleness checks compare against this record instead of re-deriving
 * the decision, so a deliberately skipped file is not reported as a new source.
 */
export interface ParsedMirrorSkipEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly reason: ParsedMirrorSkipReason;
	readonly sourceMtimeNs: number;
	readonly sourceSizeBytes: number;
	readonly updatedAt: string;
}

export interface ParsedMirrorIndex {
	readonly version: 1;
	readonly entries: Readonly<Record<string, ParsedMirrorEntry>>;
	/** Deliberate skips recorded by the last refresh. Absent in indexes written before this field existed. */
	readonly skipped?: Readonly<Record<string, ParsedMirrorSkipEntry>>;
}

export function emptyMirrorIndex(): ParsedMirrorIndex {
	return { version: 1, entries: {} };
}

export function loadMirrorIndex(root: string): ParsedMirrorIndex {
	const path = parsedMirrorIndexPath(root);
	if (!existsSync(path)) return emptyMirrorIndex();
	const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
	return isParsedMirrorIndex(parsed) ? parsed : emptyMirrorIndex();
}

export function saveMirrorIndex(root: string, index: ParsedMirrorIndex): void {
	const path = parsedMirrorIndexPath(root);
	mkdirSync(dirname(path), { recursive: true });
	const tmp = `${path}.${randomUUID()}.tmp`;
	writeFileSync(tmp, `${JSON.stringify(index, null, 2)}\n`);
	renameSync(tmp, path);
}

function isParsedMirrorIndex(value: unknown): value is ParsedMirrorIndex {
	if (!isRecord(value) || value.version !== 1 || !isRecord(value.entries)) return false;
	if (value.skipped !== undefined && !isParsedMirrorSkipRecord(value.skipped)) return false;
	return Object.entries(value.entries).every(
		([key, entry]) => isParsedMirrorEntry(entry) && key === entry.virtualPath,
	);
}

function isParsedMirrorSkipRecord(value: unknown): boolean {
	if (!isRecord(value)) return false;
	return Object.entries(value).every(([key, entry]) => isParsedMirrorSkipEntry(entry) && key === entry.virtualPath);
}

function isParsedMirrorSkipEntry(value: unknown): value is ParsedMirrorSkipEntry {
	return (
		isRecord(value) &&
		typeof value.virtualPath === "string" &&
		typeof value.sourcePath === "string" &&
		isParsedMirrorSkipReason(value.reason) &&
		typeof value.sourceMtimeNs === "number" &&
		typeof value.sourceSizeBytes === "number" &&
		typeof value.updatedAt === "string"
	);
}

function isParsedMirrorSkipReason(value: unknown): value is ParsedMirrorSkipReason {
	return value === "duplicate-excluded" || value === "parser-skipped" || value === "parser-failed";
}

function isParsedMirrorEntry(value: unknown): value is ParsedMirrorEntry {
	return (
		isRecord(value) &&
		typeof value.virtualPath === "string" &&
		typeof value.sourcePath === "string" &&
		typeof value.outputPath === "string" &&
		typeof value.parserName === "string" &&
		typeof value.sourceMtimeNs === "number" &&
		typeof value.sourceSizeBytes === "number" &&
		typeof value.updatedAt === "string" &&
		(value.contentSha256 === undefined || typeof value.contentSha256 === "string")
	);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}
