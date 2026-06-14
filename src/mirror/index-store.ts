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
}

export interface ParsedMirrorIndex {
	readonly version: 1;
	readonly entries: Readonly<Record<string, ParsedMirrorEntry>>;
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
	return Object.entries(value.entries).every(
		([key, entry]) => isParsedMirrorEntry(entry) && key === entry.virtualPath,
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
