/** Where a retrieved chunk sits inside its parsed document. Every field is measured, never model-written. */
export interface ChunkLocation {
	/** 1-based first line of the chunk in the parsed document. */
	readonly lineStart: number;
	/** 1-based last line of the chunk in the parsed document. */
	readonly lineEnd: number;
	/** Nearest article or heading line at or before the chunk's first line. */
	readonly section?: string;
}

/** Korean statute headings (`제5조(…)`, `제2장 …`) and markdown headings. */
const ARTICLE_LINE = /^제\s*\d+(?:의\d+)?\s*[조장절관](?:\s|\(|$)/u;
const MARKDOWN_HEADING = /^#{1,6}\s+(.+?)\s*#*$/u;
const MAX_SECTION_CHARS = 80;
const MAX_CONTEXT_CHARS = 120;
const MAX_CONTEXT_VALUE_CHARS = 32;

const CHAT_KEYS = ["chatName", "channelName", "guildName"] as const;
const SENDER_KEYS = ["senderNickname", "senderName", "userName", "authorName", "from"] as const;
const TIME_KEYS = ["startedAt", "publishedAt", "timestamp", "date", "createdAt"] as const;

function sectionLabel(line: string): string | undefined {
	const text = line.trim();
	const heading = MARKDOWN_HEADING.exec(text);
	if (heading?.[1] !== undefined) return heading[1].slice(0, MAX_SECTION_CHARS);
	return ARTICLE_LINE.test(text) ? text.slice(0, MAX_SECTION_CHARS) : undefined;
}

/**
 * Find `chunk` in `document` by its non-blank lines (compared trimmed, so
 * indentation and blank-line differences between the chunker's output and the
 * file do not matter) and report the 1-based line range plus the section it
 * sits in.
 *
 * Returns undefined unless the chunk matches exactly one place: a chunk the
 * chunker rewrote (table flattening) or one that occurs twice has no single
 * correct line, and reporting a guess would be worse than reporting nothing.
 */
export function locateChunk(document: string, chunk: string): ChunkLocation | undefined {
	const wanted = chunk
		.split("\n")
		.map((line) => line.trim())
		.filter((line) => line.length > 0);
	if (wanted.length === 0) return undefined;
	const lines = document.split("\n").map((line) => line.trim());

	let found: { start: number; end: number } | undefined;
	for (let start = 0; start < lines.length; start++) {
		if (lines[start] !== wanted[0]) continue;
		let end = start;
		let matched = 1;
		for (let next = start + 1; matched < wanted.length && next < lines.length; next++) {
			if (lines[next] === "") continue;
			if (lines[next] !== wanted[matched]) break;
			matched++;
			end = next;
		}
		if (matched !== wanted.length) continue;
		if (found !== undefined) return undefined;
		found = { start, end };
	}
	if (found === undefined) return undefined;

	let section: string | undefined;
	for (let index = found.start; index >= 0 && section === undefined; index--) {
		section = sectionLabel(lines[index] ?? "");
	}
	return {
		lineStart: found.start + 1,
		lineEnd: found.end + 1,
		...(section !== undefined ? { section } : {}),
	};
}

/** One stable text form of a location, for tool output the model reads. Undefined when nothing is known. */
export function describeLocation(location: {
	readonly lineStart?: number;
	readonly lineEnd?: number;
	readonly section?: string;
}): string | undefined {
	const parts: string[] = [];
	if (location.lineStart !== undefined) {
		parts.push(
			location.lineEnd !== undefined && location.lineEnd !== location.lineStart
				? `lines ${location.lineStart}-${location.lineEnd}`
				: `line ${location.lineStart}`,
		);
	}
	if (location.section !== undefined) parts.push(`section "${location.section}"`);
	return parts.length > 0 ? parts.join(", ") : undefined;
}

/** ` (lines 7-8, section "…")` for a retrieval result whose backend measured a location; empty otherwise. */
export function locationSuffix(metadata: Readonly<Record<string, unknown>>): string {
	const described = describeMeasuredLocation(metadata);
	return described === undefined ? "" : ` (${described})`;
}

/**
 * ` (lines 7-8; chat: "…", from: "…", at: 2026-…, subject: "…")` for an agent-facing
 * result row. Location is measured by the retrieval backend; the remaining values
 * come from a small source-metadata allowlist and are bounded so one row cannot
 * flood the model's context. Unknown metadata is deliberately ignored.
 */
export function resultContextSuffix(metadata: Readonly<Record<string, unknown>>): string {
	const location = describeMeasuredLocation(metadata);
	const sourceContext = describeSourceMetadata(metadata);
	const parts = [location, sourceContext].filter((part): part is string => part !== undefined);
	return parts.length === 0 ? "" : ` (${parts.join("; ")})`;
}

function describeMeasuredLocation(metadata: Readonly<Record<string, unknown>>): string | undefined {
	const lineStart = typeof metadata.lineNumber === "number" ? metadata.lineNumber : undefined;
	const lineEnd = typeof metadata.lineEnd === "number" ? metadata.lineEnd : undefined;
	const section = typeof metadata.section === "string" && metadata.section.length > 0 ? metadata.section : undefined;
	return describeLocation({
		...(lineStart !== undefined ? { lineStart } : {}),
		...(lineEnd !== undefined ? { lineEnd } : {}),
		...(section !== undefined ? { section } : {}),
	});
}

function describeSourceMetadata(metadata: Readonly<Record<string, unknown>>): string | undefined {
	const parts: string[] = [];
	const chat = firstText(metadata, CHAT_KEYS);
	if (chat !== undefined) parts.push(`chat: ${quoted(chat)}`);

	const sender = firstText(metadata, SENDER_KEYS);
	if (sender !== undefined) parts.push(`from: ${quoted(sender)}`);

	const time = firstTime(metadata, TIME_KEYS);
	if (time !== undefined) parts.push(`at: ${time}`);

	const subject = firstText(metadata, ["subject"] as const);
	if (subject !== undefined) parts.push(`subject: ${quoted(subject)}`);

	return boundedParts(parts);
}

function firstText(metadata: Readonly<Record<string, unknown>>, keys: readonly string[]): string | undefined {
	for (const key of keys) {
		const value = metadata[key];
		if (typeof value !== "string") continue;
		const compact = compactText(value, MAX_CONTEXT_VALUE_CHARS);
		if (compact.length > 0) return compact;
	}
	return undefined;
}

function firstTime(metadata: Readonly<Record<string, unknown>>, keys: readonly string[]): string | undefined {
	for (const key of keys) {
		const normalized = normalizeTimestamp(metadata[key]);
		if (normalized !== undefined) return normalized;
	}
	return undefined;
}

function normalizeTimestamp(value: unknown): string | undefined {
	if (typeof value === "number" && Number.isFinite(value)) return isoTime(value);
	if (typeof value !== "string") return undefined;
	const text = value.trim();
	if (text.length === 0) return undefined;

	// Slack timestamps are decimal epoch seconds; mail and chat timestamps are
	// normally ISO strings. Normalize every parseable value to an ISO instant.
	if (/^[+-]?\d+(?:\.\d+)?$/u.test(text)) {
		const numeric = Number(text);
		if (!Number.isFinite(numeric)) return undefined;
		return isoTime(Math.abs(numeric) < 100_000_000_000 ? numeric * 1_000 : numeric);
	}
	const parsed = Date.parse(text);
	return Number.isFinite(parsed) ? isoTime(parsed) : compactText(text, MAX_CONTEXT_VALUE_CHARS);
}

function isoTime(milliseconds: number): string | undefined {
	const date = new Date(milliseconds);
	return Number.isFinite(date.getTime()) ? date.toISOString() : undefined;
}

function quoted(value: string): string {
	return JSON.stringify(value);
}

function compactText(value: string, maxChars: number): string {
	const compact = value.replace(/\s+/gu, " ").trim();
	if (compact.length <= maxChars) return compact;
	return `${compact.slice(0, Math.max(0, maxChars - 1))}…`;
}

function boundedParts(parts: readonly string[]): string | undefined {
	if (parts.length === 0) return undefined;
	const shown: string[] = [];
	for (const part of parts) {
		const candidate = [...shown, part].join(", ");
		if (candidate.length <= MAX_CONTEXT_CHARS) {
			shown.push(part);
			continue;
		}
		if (shown.length === 0) return `${part.slice(0, MAX_CONTEXT_CHARS - 1)}…`;
		break;
	}
	const omitted = shown.length < parts.length;
	return `${shown.join(", ")}${omitted ? "…" : ""}`.slice(0, MAX_CONTEXT_CHARS);
}
