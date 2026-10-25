import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { writeFileAtomic } from "./atomic-write.ts";
import { parsedMirrorIndexPath, parsedMirrorRoot, parsedOutputPath } from "./paths.ts";

export interface ParsedMirrorEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly outputPath: string;
	readonly parserName: string;
	readonly sourceMtimeNs: number;
	readonly sourceSizeBytes: number;
	readonly updatedAt: string;
	readonly parserVersion?: string;
	readonly sourceVersion?: string;
	readonly parserFingerprint?: string;
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
	/** Output-affecting parser options under which this skip was decided. */
	readonly parserFingerprint?: string;
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

/** Stable opaque stage IDs: filesystem names may contain backslashes, controls or drive-like colons. */
export function mirrorStageKey(virtualPath: string): string {
	return encodeURIComponent(virtualPath.slice(1));
}

export function mirrorStageVirtualPath(key: string): string {
	return `/${decodeURIComponent(key)}`;
}

function readOptionalText(path: string): string | undefined {
	try {
		return readFileSync(path, "utf8");
	} catch (error) {
		if (error instanceof Error && (error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
		throw error;
	}
}

export function loadMirrorIndex(root: string): ParsedMirrorIndex {
	const path = parsedMirrorIndexPath(root);
	const text = readOptionalText(path);
	const parsed: unknown = text === undefined ? undefined : JSON.parse(text);
	const index = isParsedMirrorIndex(parsed) ? parsed : emptyMirrorIndex();
	const entries: Record<string, ParsedMirrorEntry> = { ...index.entries };
	const skipped = { ...index.skipped };
	const stageDir = join(parsedMirrorRoot(root), "stage");
	// Read-only projection, never a resume planner. Stable source metadata travels
	// in MinSync's opaque version so a hard host crash cannot orphan committed
	// mirrors before index.json is published. MinSync alone validates completions.
	const project = (key: string, version: unknown, updatedAt: string, fingerprint: unknown): void => {
		if (typeof version !== "string") return;
		let source: unknown;
		try {
			source = JSON.parse(version);
		} catch {
			return;
		}
		const virtualPath = mirrorStageVirtualPath(key);
		const outputPath = parsedOutputPath(root, virtualPath);
		const entry = isRecord(source)
			? {
					...source,
					outputPath,
					updatedAt,
					...(typeof fingerprint === "string" ? { parserFingerprint: fingerprint } : {}),
				}
			: undefined;
		if (!isParsedMirrorEntry(entry) || entry.virtualPath !== virtualPath) return;
		if (!existsSync(outputPath)) {
			delete entries[virtualPath];
			return;
		}
		entries[entry.virtualPath] = entry;
		delete skipped[entry.virtualPath];
	};
	const snapshotPath = join(stageDir, "snapshot.json");
	const snapshotText = readOptionalText(snapshotPath);
	if (snapshotText !== undefined) {
		const snapshot: unknown = JSON.parse(snapshotText);
		if (
			isRecord(snapshot) &&
			snapshot.source_id === "autorag-parsed-mirror" &&
			snapshot.stage === "parse" &&
			isRecord(snapshot.items) &&
			typeof snapshot.updated_at === "string"
		) {
			for (const [key, item] of Object.entries(snapshot.items))
				if (isRecord(item)) project(key, item.version, snapshot.updated_at, item.fingerprint);
		}
	}
	const journalPath = join(stageDir, "progress.jsonl");
	const journalText = readOptionalText(journalPath);
	if (journalText !== undefined) {
		const text = journalText;
		// A torn final append is not a record. A corrupt middle invalidates the
		// entire projection; the core applies the same conservative rule on open.
		const lines = text
			.slice(0, text.lastIndexOf("\n") + 1)
			.trimEnd()
			.split("\n");
		let records: unknown[] = [];
		try {
			records = lines.map((line) => JSON.parse(line));
		} catch {
			/* No trusted projection. */
		}
		const [header, ...items] = records;
		if (
			isRecord(header) &&
			header.source_id === "autorag-parsed-mirror" &&
			header.stage === "parse" &&
			typeof header.created_at === "string"
		) {
			for (const item of items) {
				if (!isRecord(item) || typeof item.item !== "string") continue;
				if (item.outcome === "committed") project(item.item, item.version, header.created_at, header.fingerprint);
				else if (item.outcome === "deleted" || item.outcome === "failed")
					delete entries[mirrorStageVirtualPath(item.item)];
			}
		}
	}
	return { version: 1, entries, skipped };
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
