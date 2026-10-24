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
	const lineStart = typeof metadata.lineNumber === "number" ? metadata.lineNumber : undefined;
	const lineEnd = typeof metadata.lineEnd === "number" ? metadata.lineEnd : undefined;
	const section = typeof metadata.section === "string" && metadata.section.length > 0 ? metadata.section : undefined;
	const described = describeLocation({
		...(lineStart !== undefined ? { lineStart } : {}),
		...(lineEnd !== undefined ? { lineEnd } : {}),
		...(section !== undefined ? { section } : {}),
	});
	return described === undefined ? "" : ` (${described})`;
}
