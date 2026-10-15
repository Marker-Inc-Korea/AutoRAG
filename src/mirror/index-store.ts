import { existsSync, readFileSync } from "node:fs";
import { writeFileAtomic } from "./atomic-write.ts";
import { parsedMirrorIndexPath } from "./paths.ts";

export interface ParsedMirrorEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly outputPath: string;
	readonly parserName: string;
	readonly sourceMtimeNs: number;
	readonly sourceSizeBytes: number;
	readonly updatedAt: string;
}

/** Why refresh deliberately left a source out of the parsed mirror. */
export type ParsedMirrorSkipReason =
	| "duplicate-excluded"
	| "user-excluded"
	| "parser-skipped"
	| "parser-failed"
	| "source-unreadable";

/**
 * A source the last refresh decided not to mirror, pinned to the exact version it
 * decided on. Staleness checks compare against this record instead of re-deriving
 * the decision, so a deliberately skipped file is not reported as a new source.
 *
 * Retryable skips (`parser-failed`, `source-unreadable`) also carry the parser stamp, so refresh
 * can honor the record without reading or parsing the file again.
 */
export interface ParsedMirrorSkipEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly reason: ParsedMirrorSkipReason;
	readonly sourceMtimeNs: number;
	readonly sourceSizeBytes: number;
	readonly updatedAt: string;
	/** `Parser.name` the skip was decided with. */
	readonly parserName?: string;
	/** `Parser.version` the skip was decided with. Absent on records written before version stamps. */
	readonly parserVersion?: string;
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
	writeFileAtomic(parsedMirrorIndexPath(root), `${JSON.stringify(index, null, 2)}\n`);
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
		typeof value.updatedAt === "string" &&
		(value.parserName === undefined || typeof value.parserName === "string") &&
		(value.parserVersion === undefined || typeof value.parserVersion === "string")
	);
}

function isParsedMirrorSkipReason(value: unknown): value is ParsedMirrorSkipReason {
	return (
		value === "duplicate-excluded" ||
		value === "user-excluded" ||
		value === "parser-skipped" ||
		value === "parser-failed" ||
		value === "source-unreadable"
	);
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
		typeof value.updatedAt === "string"
	);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}
