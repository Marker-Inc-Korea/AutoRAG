import { type ParsedPages, pageRange } from "../parser/pages.ts";

/** Where a retrieved chunk sits inside its parsed document. Every field is measured, never model-written. */
export interface ChunkLocation {
	/** 1-based first line of the chunk in the parsed document. */
	readonly lineStart: number;
	/** 1-based last line of the chunk in the parsed document. */
	readonly lineEnd: number;
	/**
	 * 1-based first page of the chunk, measured by the parser. Only set for a document with real pages
	 * (`"layout"` page mode) when both ends of the chunk land on a page the parser placed.
	 */
	readonly pageStart?: number;
	/** 1-based last page of the chunk; differs from `pageStart` for a chunk that spans a page break. */
	readonly pageEnd?: number;
	/** Nearest article or heading line at or before the chunk's first line. */
	readonly section?: string;
}

/** Korean statute headings (`제5조(…)`, `제2장 …`) and markdown headings. */
const ARTICLE_LINE = /^제\s*\d+(?:의\d+)?\s*[조장절관](?:\s|\(|$)/u;
const MARKDOWN_HEADING = /^#{1,6}\s+(.+?)\s*#*$/u;
const MAX_SECTION_CHARS = 80;

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
 * With the document's `pages`, the page range comes from the same line range: the chunk's first and last
 * line must each fall on a page the parser placed, and the page mode must be `"layout"`. A `"section"`
 * page is a section-level approximation, and a line on an unplaced page (a table continued across pages)
 * belongs to no known page, so either way no page is reported rather than a guess.
 *
 * Returns undefined unless the chunk matches exactly one place: a chunk the
 * chunker rewrote (table flattening) or one that occurs twice has no single
 * correct line, and reporting a guess would be worse than reporting nothing.
 */
export function locateChunk(document: string, chunk: string, pages?: ParsedPages): ChunkLocation | undefined {
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
	const lineStart = found.start + 1;
	const lineEnd = found.end + 1;
	const range = pages?.mode === "layout" ? pageRange(pages, lineStart, lineEnd) : undefined;
	return {
		lineStart,
		lineEnd,
		...(range ?? {}),
		...(section !== undefined ? { section } : {}),
	};
}

/** One stable text form of a location, for tool output the model reads. Undefined when nothing is known. */
export function describeLocation(location: {
	readonly lineStart?: number;
	readonly lineEnd?: number;
	readonly pageStart?: number;
	readonly pageEnd?: number;
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
	if (location.pageStart !== undefined) {
		parts.push(
			location.pageEnd !== undefined && location.pageEnd !== location.pageStart
				? `pages ${location.pageStart}-${location.pageEnd}`
				: `page ${location.pageStart}`,
		);
	}
	if (location.section !== undefined) parts.push(`section "${location.section}"`);
	return parts.length > 0 ? parts.join(", ") : undefined;
}

/** ` (lines 7-8, section "…")` for a retrieval result whose backend measured a location; empty otherwise. */
export function locationSuffix(metadata: Readonly<Record<string, unknown>>): string {
	const lineStart = typeof metadata.lineNumber === "number" ? metadata.lineNumber : undefined;
	const lineEnd = typeof metadata.lineEnd === "number" ? metadata.lineEnd : undefined;
	const pageStart = typeof metadata.pageStart === "number" ? metadata.pageStart : undefined;
	const pageEnd = typeof metadata.pageEnd === "number" ? metadata.pageEnd : undefined;
	const section = typeof metadata.section === "string" && metadata.section.length > 0 ? metadata.section : undefined;
	const described = describeLocation({
		...(lineStart !== undefined ? { lineStart } : {}),
		...(lineEnd !== undefined ? { lineEnd } : {}),
		...(pageStart !== undefined ? { pageStart } : {}),
		...(pageEnd !== undefined ? { pageEnd } : {}),
		...(section !== undefined ? { section } : {}),
	});
	return described === undefined ? "" : ` (${described})`;
}
