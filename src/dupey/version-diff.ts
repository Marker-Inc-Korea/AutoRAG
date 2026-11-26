import { readFile, stat } from "node:fs/promises";
import { basename, resolve } from "node:path";
import type { Api, AssistantMessage, Model } from "@earendil-works/pi-ai";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import { createDefaultParserRegistry, type ParserRegistry } from "../parser/index.ts";
import type { DupeyRelation } from "./cli.ts";

export type DiffLineKind = "context" | "added" | "removed";

export interface VersionDiffLine {
	readonly kind: DiffLineKind;
	readonly text: string;
	readonly previousLine: number | null;
	readonly currentLine: number | null;
}

export interface VersionDiffHunk {
	readonly previousStart: number;
	readonly previousCount: number;
	readonly currentStart: number;
	readonly currentCount: number;
	readonly lines: readonly VersionDiffLine[];
}

export interface TextLineDiff {
	readonly identical: boolean;
	readonly addedCount: number;
	readonly removedCount: number;
	readonly hunks: readonly VersionDiffHunk[];
	readonly truncated: boolean;
}

export interface VersionDiffSummary {
	readonly addedCount: number;
	readonly removedCount: number;
	readonly addedHighlights: readonly string[];
	readonly removedHighlights: readonly string[];
}

export type VersionDiffStatus = "changed" | "identical" | "unsupported" | "error";

export interface DocumentVersionDiff {
	readonly previousPath: string;
	readonly currentPath: string;
	readonly status: VersionDiffStatus;
	readonly summary: VersionDiffSummary;
	readonly hunks: readonly VersionDiffHunk[];
	readonly truncated: boolean;
	readonly error?: string;
}

export interface VersionHistoryInputMember {
	readonly path: string;
	readonly relation?: DupeyRelation | "latest";
	/** Optional override for last-modified time (epoch ms); falls back to fs.stat mtimeMs. */
	readonly mtimeMs?: number;
}

export interface VersionHistoryEntry {
	readonly path: string;
	readonly name: string;
	readonly mtimeMs: number;
	readonly relation: DupeyRelation | "latest";
	readonly isLatest: boolean;
	readonly previousPath: string | null;
	readonly diff: DocumentVersionDiff | null;
}

export interface VersionHistoryResult {
	/** Ordered newest-first by last modified timestamp. */
	readonly versions: readonly VersionHistoryEntry[];
}

export interface DiffTextOptions {
	readonly contextLines?: number;
	readonly maxHunks?: number;
	readonly maxLinesPerSide?: number;
}

export interface SummarizeDiffOptions {
	readonly maxHighlights?: number;
	readonly maxLineLength?: number;
}

export interface CompareDocumentVersionsOptions extends DiffTextOptions, SummarizeDiffOptions {
	readonly registry?: ParserRegistry;
	readonly maxFileBytes?: number;
}

const DEFAULT_CONTEXT_LINES = 2;
const DEFAULT_MAX_HUNKS = 40;
const DEFAULT_MAX_LINES_PER_SIDE = 4_000;
const DEFAULT_MAX_HIGHLIGHTS = 3;
const DEFAULT_MAX_LINE_LENGTH = 84;
const DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024;
/** Cap on Myers edit distance before switching to a bounded fast fallback. */
const MYERS_MAX_D = 1_500;

interface EditOp {
	readonly kind: DiffLineKind;
	readonly text: string;
	readonly previousLine: number | null;
	readonly currentLine: number | null;
}

function splitNormalizedLines(text: string): string[] {
	const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
	if (normalized.length === 0) return [];
	const trimmed = normalized.endsWith("\n") ? normalized.slice(0, -1) : normalized;
	if (trimmed.length === 0) return [];
	return trimmed.split("\n");
}

/**
 * Myers O((N+M)D) line diff with prefix/suffix trimming and bounded fallback.
 * Deterministic and dependency-free.
 */
function computeLineEdits(prev: readonly string[], curr: readonly string[]): EditOp[] {
	let start = 0;
	while (start < prev.length && start < curr.length && prev[start] === curr[start]) {
		start += 1;
	}
	let prevEnd = prev.length;
	let currEnd = curr.length;
	while (prevEnd > start && currEnd > start && prev[prevEnd - 1] === curr[currEnd - 1]) {
		prevEnd -= 1;
		currEnd -= 1;
	}

	const prefixOps: EditOp[] = [];
	for (let i = 0; i < start; i += 1) {
		prefixOps.push({ kind: "context", text: prev[i]!, previousLine: i + 1, currentLine: i + 1 });
	}

	const suffixOps: EditOp[] = [];
	const suffixLen = prev.length - prevEnd;
	for (let i = 0; i < suffixLen; i += 1) {
		suffixOps.push({
			kind: "context",
			text: prev[prevEnd + i]!,
			previousLine: prevEnd + i + 1,
			currentLine: currEnd + i + 1,
		});
	}

	const a = prev.slice(start, prevEnd);
	const b = curr.slice(start, currEnd);
	if (a.length === 0 && b.length === 0) {
		return [...prefixOps, ...suffixOps];
	}
	if (a.length === 0) {
		const added: EditOp[] = b.map((text, idx) => ({
			kind: "added",
			text,
			previousLine: null,
			currentLine: start + idx + 1,
		}));
		return [...prefixOps, ...added, ...suffixOps];
	}
	if (b.length === 0) {
		const removed: EditOp[] = a.map((text, idx) => ({
			kind: "removed",
			text,
			previousLine: start + idx + 1,
			currentLine: null,
		}));
		return [...prefixOps, ...removed, ...suffixOps];
	}

	const middleOps = myersEdits(a, b, start, start);
	return [...prefixOps, ...middleOps, ...suffixOps];
}

function myersEdits(a: readonly string[], b: readonly string[], prevOffset: number, currOffset: number): EditOp[] {
	const n = a.length;
	const m = b.length;
	const maxD = Math.min(n + m, MYERS_MAX_D);
	const offset = maxD;
	const v = new Int32Array(2 * maxD + 1);
	const trace: Int32Array[] = [];

	let foundD = -1;
	for (let d = 0; d <= maxD; d += 1) {
		trace.push(v.slice());
		for (let k = -d; k <= d; k += 2) {
			const idx = k + offset;
			let x: number;
			if (k === -d || (k !== d && v[idx - 1]! < v[idx + 1]!)) {
				x = v[idx + 1]!;
			} else {
				x = v[idx - 1]! + 1;
			}
			let y = x - k;
			while (x < n && y < m && a[x] === b[y]) {
				x += 1;
				y += 1;
			}
			v[idx] = x;
			if (x >= n && y >= m) {
				foundD = d;
				break;
			}
		}
		if (foundD !== -1) break;
	}

	if (foundD === -1) {
		// Beyond edit-distance cap: report clean replace block without quadratic blowup.
		const removed: EditOp[] = a.map((text, i) => ({
			kind: "removed",
			text,
			previousLine: prevOffset + i + 1,
			currentLine: null,
		}));
		const added: EditOp[] = b.map((text, i) => ({
			kind: "added",
			text,
			previousLine: null,
			currentLine: currOffset + i + 1,
		}));
		return [...removed, ...added];
	}

	const reversed: EditOp[] = [];
	let x = n;
	let y = m;
	for (let d = foundD; d > 0; d -= 1) {
		const prevV = trace[d]!;
		const k = x - y;
		const idx = k + offset;
		let prevK: number;
		if (k === -d || (k !== d && prevV[idx - 1]! < prevV[idx + 1]!)) {
			prevK = k + 1;
		} else {
			prevK = k - 1;
		}
		const prevX = prevV[prevK + offset]!;
		const prevY = prevX - prevK;

		while (x > prevX && y > prevY) {
			x -= 1;
			y -= 1;
			reversed.push({
				kind: "context",
				text: a[x]!,
				previousLine: prevOffset + x + 1,
				currentLine: currOffset + y + 1,
			});
		}

		if (x === prevX) {
			y -= 1;
			reversed.push({
				kind: "added",
				text: b[y]!,
				previousLine: null,
				currentLine: currOffset + y + 1,
			});
		} else {
			x -= 1;
			reversed.push({
				kind: "removed",
				text: a[x]!,
				previousLine: prevOffset + x + 1,
				currentLine: null,
			});
		}
	}

	while (x > 0 && y > 0) {
		x -= 1;
		y -= 1;
		reversed.push({
			kind: "context",
			text: a[x]!,
			previousLine: prevOffset + x + 1,
			currentLine: currOffset + y + 1,
		});
	}

	reversed.reverse();
	return reversed;
}

export function diffTextLines(previousText: string, currentText: string, options: DiffTextOptions = {}): TextLineDiff {
	const contextLines = Math.max(0, options.contextLines ?? DEFAULT_CONTEXT_LINES);
	const maxHunks = Math.max(1, options.maxHunks ?? DEFAULT_MAX_HUNKS);
	const maxLinesPerSide = Math.max(100, options.maxLinesPerSide ?? DEFAULT_MAX_LINES_PER_SIDE);

	const rawPrev = splitNormalizedLines(previousText);
	const rawCurr = splitNormalizedLines(currentText);
	let truncated = rawPrev.length > maxLinesPerSide || rawCurr.length > maxLinesPerSide;
	const prev = rawPrev.length > maxLinesPerSide ? rawPrev.slice(0, maxLinesPerSide) : rawPrev;
	const curr = rawCurr.length > maxLinesPerSide ? rawCurr.slice(0, maxLinesPerSide) : rawCurr;

	const ops = computeLineEdits(prev, curr);
	let addedCount = 0;
	let removedCount = 0;
	const changeIndices: number[] = [];
	for (let i = 0; i < ops.length; i += 1) {
		const op = ops[i]!;
		if (op.kind === "added") {
			addedCount += 1;
			changeIndices.push(i);
		} else if (op.kind === "removed") {
			removedCount += 1;
			changeIndices.push(i);
		}
	}

	if (changeIndices.length === 0) {
		return { identical: true, addedCount: 0, removedCount: 0, hunks: [], truncated };
	}

	// Merge adjacent change windows separated by at most 2 * contextLines.
	const ranges: Array<{ start: number; end: number }> = [];
	for (const idx of changeIndices) {
		const start = Math.max(0, idx - contextLines);
		const end = Math.min(ops.length - 1, idx + contextLines);
		const last = ranges[ranges.length - 1];
		if (last && start <= last.end + 1) {
			last.end = Math.max(last.end, end);
		} else {
			ranges.push({ start, end });
		}
	}

	if (ranges.length > maxHunks) {
		truncated = true;
	}

	const hunks: VersionDiffHunk[] = ranges.slice(0, maxHunks).map(({ start, end }) => {
		const lines = ops.slice(start, end + 1);
		let previousStart = 0;
		let previousCount = 0;
		let currentStart = 0;
		let currentCount = 0;
		for (const line of lines) {
			if (line.previousLine !== null) {
				if (previousStart === 0) previousStart = line.previousLine;
				previousCount += 1;
			}
			if (line.currentLine !== null) {
				if (currentStart === 0) currentStart = line.currentLine;
				currentCount += 1;
			}
		}
		return {
			previousStart: previousStart || 1,
			previousCount,
			currentStart: currentStart || 1,
			currentCount,
			lines,
		};
	});

	return {
		identical: false,
		addedCount,
		removedCount,
		hunks,
		truncated,
	};
}

/** Strip lightweight markdown prefixes (headings, bullets, blockquotes) so a short highlight reads cleanly. */
function cleanHighlightLine(raw: string, maxLen: number): string | null {
	let text = raw.trim();
	if (text.length === 0) return null;
	// Skip markdown horizontal rules, table divider lines, or bare fence markers.
	if (/^([-*_]{3,}|[:|\-\s]+|```[\w-]*)$/.test(text)) return null;
	// Strip leading markdown markers (headings, list bullets, quotes).
	text = text.replace(/^(#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)/, "").trim();
	// Strip inline markdown emphasis/code wrappers cleanly without mangling inner text.
	text = text
		.replace(/\*\*(.+?)\*\*/g, "$1")
		.replace(/`([^`]+)`/g, "$1")
		.replace(/\s+/g, " ")
		.trim();
	if (text.length === 0) return null;
	if (text.length <= maxLen) return text;
	return `${text.slice(0, Math.max(1, maxLen - 1)).trimEnd()}…`;
}

export function summarizeVersionDiff(diff: TextLineDiff, options: SummarizeDiffOptions = {}): VersionDiffSummary {
	const maxHighlights = Math.max(1, options.maxHighlights ?? DEFAULT_MAX_HIGHLIGHTS);
	const maxLineLength = Math.max(16, options.maxLineLength ?? DEFAULT_MAX_LINE_LENGTH);
	const addedHighlights: string[] = [];
	const removedHighlights: string[] = [];
	const seenAdded = new Set<string>();
	const seenRemoved = new Set<string>();

	for (const hunk of diff.hunks) {
		for (const line of hunk.lines) {
			if (line.kind === "added" && addedHighlights.length < maxHighlights) {
				const cleaned = cleanHighlightLine(line.text, maxLineLength);
				if (cleaned !== null && !seenAdded.has(cleaned)) {
					seenAdded.add(cleaned);
					addedHighlights.push(cleaned);
				}
			} else if (line.kind === "removed" && removedHighlights.length < maxHighlights) {
				const cleaned = cleanHighlightLine(line.text, maxLineLength);
				if (cleaned !== null && !seenRemoved.has(cleaned)) {
					seenRemoved.add(cleaned);
					removedHighlights.push(cleaned);
				}
			}
			if (addedHighlights.length >= maxHighlights && removedHighlights.length >= maxHighlights) {
				break;
			}
		}
		if (addedHighlights.length >= maxHighlights && removedHighlights.length >= maxHighlights) {
			break;
		}
	}

	return {
		addedCount: diff.addedCount,
		removedCount: diff.removedCount,
		addedHighlights,
		removedHighlights,
	};
}

let cachedDefaultRegistry: ParserRegistry | undefined;

function getDefaultRegistry(): ParserRegistry {
	cachedDefaultRegistry ??= createDefaultParserRegistry({ ocr: { enabled: false } });
	return cachedDefaultRegistry;
}

type ExtractedTextResult =
	| { readonly ok: true; readonly text: string }
	| { readonly ok: false; readonly reason: "unsupported" | "error"; readonly message?: string };

async function extractDocumentText(
	filePath: string,
	registry: ParserRegistry,
	maxFileBytes: number,
): Promise<ExtractedTextResult> {
	const parser = registry.getForVirtualPath(filePath);
	if (parser === undefined) {
		return { ok: false, reason: "unsupported" };
	}
	try {
		const info = await stat(filePath);
		if (!info.isFile()) {
			return { ok: false, reason: "error", message: "Path is not a file." };
		}
		if (info.size > maxFileBytes) {
			return { ok: false, reason: "unsupported", message: "File exceeds diff size limit." };
		}
		const bytes = await readFile(filePath);
		const parsed = await parser.parse({
			virtualPath: filePath,
			sourcePath: filePath,
			bytes,
		});
		return { ok: true, text: parsed.markdown };
	} catch (error) {
		return {
			ok: false,
			reason: "error",
			message: error instanceof Error ? error.message : String(error),
		};
	}
}

const EMPTY_SUMMARY: VersionDiffSummary = {
	addedCount: 0,
	removedCount: 0,
	addedHighlights: [],
	removedHighlights: [],
};

function failedDiff(
	previousPath: string,
	currentPath: string,
	failed: Extract<ExtractedTextResult, { readonly ok: false }>,
): DocumentVersionDiff {
	return {
		previousPath,
		currentPath,
		status: failed.reason === "unsupported" ? "unsupported" : "error",
		summary: EMPTY_SUMMARY,
		hunks: [],
		truncated: false,
		...(failed.message !== undefined ? { error: failed.message } : {}),
	};
}

/**
 * Compare two document versions (`previousPath` → `currentPath`) by extracting their
 * text through AutoRAG's parser registry and computing a structured line diff + concise summary.
 */
export async function compareDocumentVersions(
	previousPath: string,
	currentPath: string,
	options: CompareDocumentVersionsOptions = {},
): Promise<DocumentVersionDiff> {
	const prevAbs = resolve(previousPath);
	const currAbs = resolve(currentPath);
	const registry = options.registry ?? getDefaultRegistry();
	const maxFileBytes = options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;

	const [prevExtracted, currExtracted] = await Promise.all([
		extractDocumentText(prevAbs, registry, maxFileBytes),
		extractDocumentText(currAbs, registry, maxFileBytes),
	]);

	if (!prevExtracted.ok) return failedDiff(prevAbs, currAbs, prevExtracted);
	if (!currExtracted.ok) return failedDiff(prevAbs, currAbs, currExtracted);

	const diff = diffTextLines(prevExtracted.text, currExtracted.text, options);
	const summary = summarizeVersionDiff(diff, options);
	return {
		previousPath: prevAbs,
		currentPath: currAbs,
		status: diff.identical ? "identical" : "changed",
		summary,
		hunks: diff.hunks,
		truncated: diff.truncated,
	};
}

/**
 * Build a chronological version history for a dupey family.
 *
 * Members are ordered newest-first by last modified timestamp (`mtimeMs`, with path
 * tie-breaking), so the newest file is flagged `isLatest: true`. Each version is
 * compared against its immediate chronological predecessor (or `null` for the oldest base version).
 */
export async function buildVersionHistory(
	members: readonly VersionHistoryInputMember[],
	options: CompareDocumentVersionsOptions = {},
): Promise<VersionHistoryResult> {
	if (members.length === 0) return { versions: [] };

	const registry = options.registry ?? getDefaultRegistry();
	const maxFileBytes = options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;

	// Deduplicate by resolved path and collect mtimeMs.
	const seen = new Set<string>();
	const resolvedMembers: Array<{
		readonly path: string;
		readonly relation: DupeyRelation | "latest";
		readonly mtimeMs: number;
	}> = [];

	await Promise.all(
		members.map(async (member) => {
			const abs = resolve(member.path);
			if (seen.has(abs)) return;
			seen.add(abs);
			let mtimeMs = member.mtimeMs;
			if (mtimeMs === undefined || !Number.isFinite(mtimeMs)) {
				try {
					const info = await stat(abs);
					mtimeMs = info.mtimeMs;
				} catch {
					mtimeMs = 0;
				}
			}
			resolvedMembers.push({
				path: abs,
				relation: member.relation ?? "near",
				mtimeMs,
			});
		}),
	);

	// Sort newest-first by mtimeMs, deterministic tie-break by path.
	resolvedMembers.sort((a, b) => b.mtimeMs - a.mtimeMs || a.path.localeCompare(b.path));

	// Extract each member's text once so adjacent pairs reuse parsed output.
	const extractedMap = new Map<string, ExtractedTextResult>();
	await Promise.all(
		resolvedMembers.map(async (member) => {
			extractedMap.set(member.path, await extractDocumentText(member.path, registry, maxFileBytes));
		}),
	);

	const versions: VersionHistoryEntry[] = resolvedMembers.map((member, index) => {
		const isLatest = index === 0;
		const predecessor = resolvedMembers[index + 1];
		if (predecessor === undefined) {
			return {
				path: member.path,
				name: basename(member.path),
				mtimeMs: member.mtimeMs,
				relation: isLatest ? "latest" : member.relation,
				isLatest,
				previousPath: null,
				diff: null,
			};
		}

		const prevExtracted = extractedMap.get(predecessor.path)!;
		const currExtracted = extractedMap.get(member.path)!;

		let diff: DocumentVersionDiff;
		if (!prevExtracted.ok) {
			diff = failedDiff(predecessor.path, member.path, prevExtracted);
		} else if (!currExtracted.ok) {
			diff = failedDiff(predecessor.path, member.path, currExtracted);
		} else {
			const textDiff = diffTextLines(prevExtracted.text, currExtracted.text, options);
			const summary = summarizeVersionDiff(textDiff, options);
			diff = {
				previousPath: predecessor.path,
				currentPath: member.path,
				status: textDiff.identical ? "identical" : "changed",
				summary,
				hunks: textDiff.hunks,
				truncated: textDiff.truncated,
			};
		}

		return {
			path: member.path,
			name: basename(member.path),
			mtimeMs: member.mtimeMs,
			relation: isLatest ? "latest" : member.relation,
			isLatest,
			previousPath: predecessor.path,
			diff,
		};
	});

	return { versions };
}

/** Prompt + minimal system instruction in, short plain-text summary out. */
export type VersionDiffCompleter = (input: {
	readonly systemPrompt: string;
	readonly userPrompt: string;
}) => Promise<string>;

export interface VersionDiffSummaryModel {
	readonly model: Model<Api>;
	readonly apiKey?: string;
}

export interface SummarizeDiffWithLlmInput {
	readonly previousName: string;
	readonly currentName: string;
	readonly addedCount: number;
	readonly removedCount: number;
	readonly hunks: readonly VersionDiffHunk[];
	/** UI language hint (e.g. "ko", "en", "ja", "zh"). Defaults to matching the diff's language. */
	readonly language?: string;
	/** Max character budget for diff lines included in the prompt. Defaults to 3500. */
	readonly maxDiffChars?: number;
}

const DEFAULT_LLM_DIFF_CHARS = 3_500;
const MAX_LLM_SUMMARY_CHARS = 240;

const LANGUAGE_LABELS: Readonly<Record<string, string>> = {
	ko: "Korean",
	en: "English",
	ja: "Japanese",
	zh: "Simplified Chinese",
};

/** Keep the system prompt ultra-short for fast, low-token sLLM inference. */
export function buildVersionDiffSystemPrompt(language?: string): string {
	const lang = language !== undefined ? LANGUAGE_LABELS[language] : undefined;
	const langRule = lang !== undefined ? ` Reply in ${lang}.` : " Match the document language.";
	return `Summarize the document diff in 1-2 ultra-short sentences covering key additions and removals. Plain text only, no bullets or preamble.${langRule}`;
}

/** Format compact diff hunks within a tight character budget for the sLLM. */
export function formatCompactDiffForLlm(
	hunks: readonly VersionDiffHunk[],
	maxChars: number = DEFAULT_LLM_DIFF_CHARS,
): string {
	const lines: string[] = [];
	let used = 0;
	for (const hunk of hunks) {
		for (const line of hunk.lines) {
			if (line.kind === "context") continue;
			const trimmed = line.text.trim();
			if (trimmed.length === 0) continue;
			const prefix = line.kind === "added" ? "+" : "-";
			const entry = `${prefix} ${trimmed.length > 180 ? `${trimmed.slice(0, 179)}…` : trimmed}`;
			if (used + entry.length + 1 > maxChars) {
				if (lines.length === 0) lines.push(entry.slice(0, maxChars));
				return lines.join("\n");
			}
			lines.push(entry);
			used += entry.length + 1;
		}
	}
	return lines.join("\n");
}

export function buildVersionDiffUserPrompt(input: SummarizeDiffWithLlmInput): string {
	const diffBody = formatCompactDiffForLlm(input.hunks, input.maxDiffChars ?? DEFAULT_LLM_DIFF_CHARS);
	return `${input.previousName} -> ${input.currentName} (+${input.addedCount}/-${input.removedCount})\n${diffBody}`;
}

/** Clean model output into a tight, single-block 1–2 sentence summary. */
export function cleanVersionDiffSummary(raw: string): string | null {
	let text = raw
		.replace(/```[\s\S]*?```/g, " ")
		.replace(/^\s*[-*•]\s*/gm, "")
		.replace(/\s+/g, " ")
		.trim();
	if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
		text = text.slice(1, -1).trim();
	}
	if (text.length === 0) return null;
	if (text.length <= MAX_LLM_SUMMARY_CHARS) return text;
	return `${text.slice(0, MAX_LLM_SUMMARY_CHARS - 1).trimEnd()}…`;
}

/**
 * Summarize a document version diff into 1–2 short sentences using a small chat LLM
 * (e.g. the query-decomposition sLLM). Returns `null` if there are no diff lines or empty output.
 */
export async function summarizeVersionDiffWithLlm(
	complete: VersionDiffCompleter,
	input: SummarizeDiffWithLlmInput,
): Promise<string | null> {
	const diffBody = formatCompactDiffForLlm(input.hunks, input.maxDiffChars ?? DEFAULT_LLM_DIFF_CHARS);
	if (diffBody.length === 0) return null;
	const raw = await complete({
		systemPrompt: buildVersionDiffSystemPrompt(input.language),
		userPrompt: `${input.previousName} -> ${input.currentName} (+${input.addedCount}/-${input.removedCount})\n${diffBody}`,
	});
	return cleanVersionDiffSummary(raw);
}

/** Completer over a pi-ai model (reusing the query-decomposition sLLM): minimal thinking, tiny output budget. */
export function createModelVersionDiffCompleter(
	target: VersionDiffSummaryModel,
	signal?: AbortSignal,
): VersionDiffCompleter {
	return async ({ systemPrompt, userPrompt }) => {
		const result: AssistantMessage = await completeSimple(
			target.model,
			{
				systemPrompt,
				messages: [{ role: "user", content: userPrompt, timestamp: Date.now() }],
			},
			{
				...(target.apiKey !== undefined ? { apiKey: target.apiKey } : {}),
				...(signal !== undefined ? { signal } : {}),
				temperature: 0,
				reasoning: "minimal",
				maxTokens: 256,
			},
		);
		if (result.stopReason === "error" || result.stopReason === "aborted") {
			throw new Error(result.errorMessage ?? `version diff summary ${result.stopReason}`);
		}
		return result.content
			.filter((part): part is Extract<(typeof result.content)[number], { type: "text" }> => part.type === "text")
			.map((part) => part.text)
			.join("");
	};
}
