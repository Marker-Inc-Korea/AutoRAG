import { type JikjiSourceRoot, mapJikjiPath } from "./path-map.ts";
import type { JikjiPrepareResult } from "./types.ts";

export const JIKJI_FILE_MAP_ITEM_CAP = 200;
export const JIKJI_FILE_MAP_TOTAL_CHAR_CAP = 4096;
export const JIKJI_FILE_MAP_FIELD_CHAR_CAP = 256;

export interface JikjiFileMapEntry {
	readonly path: string;
	readonly label?: string;
}

export interface JikjiFileMapSummary {
	readonly entries: readonly JikjiFileMapEntry[];
	readonly truncated: boolean;
	readonly diagnostics: readonly string[];
}

export interface JikjiFileMapOptions {
	readonly sourceRoots?: readonly JikjiSourceRoot[];
}
export interface JikjiFileMapInput {
	readonly result: JikjiPrepareResult;
	readonly sourceRoots: readonly JikjiSourceRoot[];
}

export function summarizeJikjiFileMapsBySource(inputs: readonly JikjiFileMapInput[]): JikjiFileMapSummary {
	if (inputs.length === 0) {
		return { entries: [], truncated: false, diagnostics: ["Jikji did not return a file map."] };
	}
	const entries: JikjiFileMapEntry[] = [];
	const diagnostics: string[] = [];
	let truncated = false;
	for (const input of inputs) {
		const summary = summarizeJikjiFileMaps([input.result], { sourceRoots: input.sourceRoots });
		diagnostics.push(...summary.diagnostics);
		for (const entry of summary.entries) {
			if (entries.some((existing) => existing.path === entry.path)) continue;
			if (entries.length >= JIKJI_FILE_MAP_ITEM_CAP) {
				truncated = true;
				break;
			}
			entries.push(entry);
		}
		truncated = truncated || summary.truncated;
	}
	return { entries, truncated, diagnostics: uniqueStrings(diagnostics) };
}

interface CandidateEntry {
	readonly path: string;
	readonly label?: string;
}

const DRIVE_PATH = /^[A-Za-z]:[\\/]/u;
const URLISH = /^[A-Za-z][A-Za-z0-9+.-]*:/u;

export function summarizeJikjiFileMaps(
	results: readonly JikjiPrepareResult[] | undefined,
	options: JikjiFileMapOptions = {},
): JikjiFileMapSummary {
	if (results === undefined || results.length === 0) {
		return { entries: [], truncated: false, diagnostics: ["Jikji did not return a file map."] };
	}
	const entries: JikjiFileMapEntry[] = [];
	const diagnostics: string[] = [];
	let truncated = false;
	for (const result of results) {
		if (!result.ok) {
			diagnostics.push("Jikji prepare did not complete successfully; file-map output was ignored.");
			continue;
		}
		const parsed = parseJikjiFileMapStdout(result.stdout, options);
		diagnostics.push(...parsed.diagnostics);
		for (const entry of parsed.entries) {
			if (entries.some((existing) => existing.path === entry.path)) continue;
			if (entries.length >= JIKJI_FILE_MAP_ITEM_CAP) {
				truncated = true;
				break;
			}
			entries.push(entry);
		}
	}
	return { entries, truncated, diagnostics: uniqueStrings(diagnostics) };
}

export function parseJikjiFileMapStdout(stdout: string, options: JikjiFileMapOptions = {}): JikjiFileMapSummary {
	const trimmed = stdout.trim();
	if (trimmed.length === 0) {
		return { entries: [], truncated: false, diagnostics: ["Jikji prepare produced no file-map JSON."] };
	}
	let value: unknown;
	try {
		value = JSON.parse(trimmed);
	} catch {
		return { entries: [], truncated: false, diagnostics: ["Jikji file-map JSON was not parseable."] };
	}
	const rawEntries = extractCandidateEntries(value);
	if (rawEntries === undefined) {
		return { entries: [], truncated: false, diagnostics: ["Jikji file-map JSON used an unknown shape."] };
	}
	const entries: JikjiFileMapEntry[] = [];
	let truncated = false;
	for (const raw of rawEntries) {
		const entry = sanitizeEntry(raw, options);
		if (entry === undefined) continue;
		if (entries.some((existing) => existing.path === entry.path)) continue;
		if (entries.length >= JIKJI_FILE_MAP_ITEM_CAP) {
			truncated = true;
			break;
		}
		entries.push(entry);
	}
	const diagnostics = entries.length === 0 ? ["Jikji file-map JSON contained no safe entries."] : [];
	return { entries, truncated, diagnostics };
}

export function renderJikjiFileMapContext(summary: JikjiFileMapSummary): string {
	if (summary.entries.length === 0) {
		return "No sanitized Jikji file map is available yet. Continue using the active retrieval tools; do not rely on raw Jikji output.";
	}
	let text = [
		"Sanitized Jikji file map prepared from configured source directories. Use it only to choose promising files/scopes for AutoRAG tools; it is not retrieval evidence.",
		"<jikji_file_map>",
	].join("\n");
	let written = 0;
	for (const entry of summary.entries) {
		const line = entry.label === undefined ? `- ${entry.path}` : `- ${entry.path} — ${entry.label}`;
		if (text.length + line.length + "\n</jikji_file_map>".length > JIKJI_FILE_MAP_TOTAL_CHAR_CAP) {
			break;
		}
		text += `\n${line}`;
		written += 1;
	}
	text += "\n</jikji_file_map>";
	if (summary.truncated || written < summary.entries.length) {
		text += "\nJikji file map was capped; search broadly when needed.";
	}
	return text;
}

function extractCandidateEntries(value: unknown): readonly CandidateEntry[] | undefined {
	if (!isRecord(value)) return undefined;
	const keys = Object.keys(value);
	if (keys.length !== 1) return undefined;
	const key = keys[0];
	if (key !== "files" && key !== "fileMap") return undefined;
	const files = value[key];
	if (!Array.isArray(files)) return undefined;
	return files.filter(isCandidateEntry);
}

function isCandidateEntry(value: unknown): value is CandidateEntry {
	if (!isRecord(value)) return false;
	if (typeof value.path !== "string") return false;
	return value.label === undefined || typeof value.label === "string";
}

function sanitizeEntry(entry: CandidateEntry, options: JikjiFileMapOptions): JikjiFileMapEntry | undefined {
	const path = sanitizeOpaquePath(entry.path, options);
	if (path === undefined) return undefined;
	const label = sanitizeLabel(entry.label);
	return label === undefined ? { path } : { path, label };
}

function sanitizeOpaquePath(value: string, options: JikjiFileMapOptions): string | undefined {
	const path = value.trim();
	if (path.length === 0) return undefined;
	if (path.includes("\0") || path.includes("?") || path.includes("#")) return undefined;
	if (DRIVE_PATH.test(path) || URLISH.test(path) || path.includes("\\") || path.startsWith("/")) return undefined;
	if (path.split("/").some((part) => part === ".." || part === ".")) return undefined;
	if (options.sourceRoots !== undefined && options.sourceRoots.length > 0) {
		for (const root of options.sourceRoots) {
			const mapped = mapJikjiPath(root, path);
			if (mapped !== undefined) return truncateField(mapped);
		}
		return undefined;
	}
	return truncateField(`/${path.replace(/\/+/gu, "/")}`);
}

function sanitizeLabel(value: string | undefined): string | undefined {
	if (typeof value !== "string") return undefined;
	const raw = value.trim();
	if (raw.includes("/") || raw.includes("\\") || raw.includes("..") || DRIVE_PATH.test(raw) || URLISH.test(raw)) {
		return undefined;
	}
	const label = raw
		.replace(/[\0\r\n\t<>{}`]/gu, " ")
		.replace(/\s+/gu, " ")
		.trim();
	if (label.length === 0) return undefined;
	if (/\b(ignore|instruction|prompt|system|assistant|tool|jikji_file_map)\b/iu.test(label)) return undefined;
	return truncateField(label);
}

function truncateField(value: string): string {
	return value.length <= JIKJI_FILE_MAP_FIELD_CHAR_CAP
		? value
		: `${value.slice(0, JIKJI_FILE_MAP_FIELD_CHAR_CAP - 1)}…`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function uniqueStrings(values: readonly string[]): readonly string[] {
	return [...new Set(values)];
}
