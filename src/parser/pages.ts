/** One page's extent in the parsed markdown, in 1-based lines. Every field is measured, never guessed. */
export interface PageSpan {
	/** 1-based page number (a slide number for PPTX). */
	readonly page: number;
	readonly lineStart: number;
	readonly lineEnd: number;
}

/**
 * `"layout"` spans are real pages (Hancom-saved HWP/HWPX, PDF, PPTX slides). `"section"` spans are a
 * section-level approximation for generated files and must never be reported as a page number.
 */
export type PageMode = "layout" | "section";

/** Where each page of a parsed document sits in its markdown. A page that could not be placed has no span. */
export interface ParsedPages {
	readonly mode: PageMode;
	/** Page or slide count the parser reported; can exceed `spans.length` (empty or unplaceable pages). */
	readonly pageCount?: number;
	/** Ascending by page and by line; neighbours may share one boundary line (a paragraph across a page break). */
	readonly spans: readonly PageSpan[];
}

export interface PageMarkdownInput {
	readonly page: number;
	readonly markdown: string;
}

function pageLines(markdown: string): string[] {
	const lines = markdown.split("\n");
	let start = 0;
	let end = lines.length;
	while (start < end && lines[start]?.trim() === "") start++;
	while (end > start && lines[end - 1]?.trim() === "") end--;
	return lines.slice(start, end);
}

const WHITESPACE = /\s/u;

function stripWhitespace(text: string): string {
	let out = "";
	for (const char of text) if (!WHITESPACE.test(char)) out += char;
	return out;
}

/** The document without whitespace, with the line each remaining character came from. */
interface FlatDocument {
	readonly text: string;
	/** 1-based line of the character at each index of `text`. */
	readonly lineOf: Int32Array;
	/** Exclusive end index in `text` of the line each character belongs to. */
	readonly lineEndOf: Int32Array;
	/** Indexes of `text` where a document line begins. */
	readonly lineStarts: ReadonlySet<number>;
}

function flatten(document: string): FlatDocument {
	let text = "";
	const lineOf: number[] = [];
	const lineEndOf: number[] = [];
	const lineStarts = new Set<number>();
	for (const [index, line] of document.split("\n").entries()) {
		const flat = stripWhitespace(line);
		if (flat.length === 0) continue;
		lineStarts.add(text.length);
		text += flat;
		for (let offset = 0; offset < flat.length; offset++) {
			lineOf.push(index + 1);
			lineEndOf.push(text.length);
		}
	}
	return { text, lineOf: Int32Array.from(lineOf), lineEndOf: Int32Array.from(lineEndOf), lineStarts };
}

/**
 * Place each page's markdown inside the document's markdown, scanning forward so repeated page text
 * resolves in order. Text is compared with whitespace ignored: kordoc renders a paragraph that crosses a
 * page break as one line of the document, so two neighbouring pages can share that line, and a page's own
 * line breaks need not match the document's.
 *
 * A match must start at the beginning of a document line or exactly where the previous page ended (the
 * shared line), and must end at the end of a line or leave a remainder the next page continues with. A
 * page with no markdown (a table continued across pages leaves the middle pages empty), a page that is not
 * in the document, and a page number that does not advance are left without a span, so a neighbour is never
 * moved onto them. After a page that had text but could not be placed, the next page is accepted only when
 * its text occurs once in the rest of the document: with a gap of unplaced text, an earlier duplicate must
 * not be taken for it.
 *
 * Undefined when no page could be placed.
 */
export function alignPages(
	document: string,
	pages: readonly PageMarkdownInput[],
	mode: PageMode,
	pageCount?: number,
): ParsedPages | undefined {
	const flat = flatten(document);
	const wanted = pages.map((page) => ({ page: page.page, text: stripWhitespace(page.markdown) }));
	const spans: PageSpan[] = [];
	let cursor = 0;
	let lastPage = 0;
	let needUnique = false;

	const validAt = (at: number, text: string, next: string | undefined): boolean => {
		if (at !== cursor && !flat.lineStarts.has(at)) return false;
		const end = at + text.length;
		if (end >= flat.text.length) return true;
		const remainder = flat.text.slice(end, flat.lineEndOf[end - 1]);
		return remainder.length === 0 || next?.startsWith(remainder) === true;
	};
	const find = (from: number, text: string, next: string | undefined): number => {
		for (let at = flat.text.indexOf(text, from); at !== -1; at = flat.text.indexOf(text, at + 1)) {
			if (validAt(at, text, next)) return at;
		}
		return -1;
	};

	for (const [index, page] of wanted.entries()) {
		if (page.text.length === 0 || page.page <= lastPage) continue;
		const next = wanted.slice(index + 1).find((candidate) => candidate.text.length > 0)?.text;
		const at = find(cursor, page.text, next);
		if (at === -1 || (needUnique && find(at + 1, page.text, next) !== -1)) {
			needUnique = true;
			continue;
		}
		needUnique = false;
		const end = at + page.text.length;
		spans.push({ page: page.page, lineStart: flat.lineOf[at] as number, lineEnd: flat.lineOf[end - 1] as number });
		cursor = end;
		lastPage = page.page;
	}
	if (spans.length === 0) return undefined;
	return { mode, ...(pageCount !== undefined ? { pageCount } : {}), spans };
}

/**
 * The page range of lines `lineStart..lineEnd`: the first page placed on `lineStart` and the last page
 * placed on `lineEnd`. A line two pages share (a paragraph across a break) belongs to both, so a chunk on
 * that line alone reports both pages. Undefined when either end is on a line no placed page covers.
 */
export function pageRange(
	pages: ParsedPages,
	lineStart: number,
	lineEnd: number,
): { readonly pageStart: number; readonly pageEnd: number } | undefined {
	const first = pages.spans.find((span) => lineStart >= span.lineStart && lineStart <= span.lineEnd);
	const last = [...pages.spans].reverse().find((span) => lineEnd >= span.lineStart && lineEnd <= span.lineEnd);
	return first === undefined || last === undefined ? undefined : { pageStart: first.page, pageEnd: last.page };
}

/**
 * Join per-page markdown into one document and report where each non-empty page landed. Pages that are
 * empty keep their number out of `spans` but still count toward `pageCount`, so the next page keeps its own
 * number. The lines are counted while joining, never searched for, so a repeated page text cannot be
 * mistaken for another page.
 */
export function joinPages(
	pages: readonly PageMarkdownInput[],
	mode: PageMode,
	pageCount: number,
): { readonly markdown: string; readonly pages: ParsedPages | undefined } {
	const spans: PageSpan[] = [];
	const parts: string[] = [];
	let line = 1;
	for (const page of pages) {
		const lines = pageLines(page.markdown);
		if (lines.length === 0) continue;
		const text = lines.join("\n");
		spans.push({ page: page.page, lineStart: line, lineEnd: line + lines.length - 1 });
		parts.push(text);
		line += lines.length + 1;
	}
	return {
		markdown: parts.join("\n\n"),
		pages: spans.length === 0 ? undefined : { mode, pageCount, spans },
	};
}
