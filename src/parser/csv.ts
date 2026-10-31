import { basename, extname } from "node:path";
import { classifyText, renderWorkbook, type SheetCell } from "./spreadsheet-render.ts";
import { decodeText, normalizeMarkdown } from "./text.ts";
import { type ParseDiagnostic, type ParseInput, type ParseOutput, Parser } from "./types.ts";

/** Delimited-text extensions the CSV parser owns. */
export const DELIMITED_EXTENSIONS = [".csv", ".tsv"] as const;

/** Rows kept per file, header included. Beyond this the file is indexed partially, with a warning. */
const DEFAULT_MAX_ROWS = 100_000;
const SNIFF_LINES = 50;
/** Candidates for a `.csv` separator, in tie-break order. */
const CSV_DELIMITERS = [",", ";", "\t", "|"] as const;
/** Excel's optional first line naming the separator, e.g. `sep=;`. */
const SEPARATOR_DIRECTIVE = /^sep=(.)\r?\n/i;

export interface CsvParserOptions {
	/** Rows kept per file, header included. Default 100,000. */
	readonly maxRows?: number;
}

/**
 * Parses `.csv` and `.tsv` files into the same row lines as workbooks (see `spreadsheet-render.ts`):
 * the file name stands in for the sheet name, titles above a table become its caption, and each value
 * is paired with its column label when the header is unambiguous.
 *
 * Bytes are decoded like plain text (UTF-8 with or without BOM, or CP949/EUC-KR as Korean Excel saves
 * CSV). Fields follow RFC 4180: quoted fields may hold separators, line breaks and doubled quotes, and
 * rows may have different lengths. `.tsv` is tab-separated; a `.csv` uses Excel's `sep=` line when present,
 * otherwise the separator (`,` `;` tab `|`) that splits the first lines most consistently outside quotes.
 * Values stay exactly as written: no number, date or formula conversion.
 */
export class CsvParser extends Parser {
	readonly name = "csv";
	readonly extensions = DELIMITED_EXTENSIONS;
	readonly #maxRows: number;

	constructor(options: CsvParserOptions = {}) {
		super();
		this.#maxRows = options.maxRows ?? DEFAULT_MAX_ROWS;
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		const extension = extname(input.virtualPath);
		const format = extension.slice(1).toLowerCase();
		let text = decodeText(input.bytes).replace(/^\uFEFF/, "");
		const directive = SEPARATOR_DIRECTIVE.exec(text);
		if (directive !== null) text = text.slice(directive[0].length);
		const delimiter = directive?.[1] ?? (format === "tsv" ? "\t" : sniffDelimiter(text));
		const { rows, total } = parseRows(text, delimiter, this.#maxRows);
		const diagnostics: ParseDiagnostic[] =
			total > this.#maxRows
				? [
						{
							code: "parser-warning",
							severity: "warning",
							message: `"${basename(input.virtualPath)}": indexed the first ${this.#maxRows} of ${total} rows`,
						},
					]
				: [];
		const grid = { name: basename(input.virtualPath, extension), rows: rows.map((row) => row.map(toCell)) };
		return {
			markdown: normalizeMarkdown(renderWorkbook([grid])),
			metadata: { parser: this.name, format },
			...(diagnostics.length > 0 ? { diagnostics } : {}),
		};
	}
}

function toCell(text: string): SheetCell | undefined {
	return text.trim() === "" ? undefined : { text, kind: classifyText(text) };
}

/** RFC 4180 rows: keeps the first `maxRows` rows and counts the rest. */
function parseRows(text: string, delimiter: string, maxRows: number): { rows: string[][]; total: number } {
	const rows: string[][] = [];
	let total = 0;
	let row: string[] = [];
	let index = 0;
	while (index < text.length) {
		let field: string;
		if (text[index] === '"') {
			const quoted = readQuoted(text, index + 1);
			const end = nextBoundary(text, quoted.end, delimiter);
			// Text after the closing quote is kept, as spreadsheet apps do with `"a"b`.
			field = quoted.value + text.slice(quoted.end, end);
			index = end;
		} else {
			const end = nextBoundary(text, index, delimiter);
			field = text.slice(index, end);
			index = end;
		}
		row.push(field);
		if (index >= text.length) break;
		if (text[index] === delimiter) {
			index++;
			if (index >= text.length) row.push("");
			continue;
		}
		index += text[index] === "\r" && text[index + 1] === "\n" ? 2 : 1;
		if (total < maxRows) rows.push(row);
		total++;
		row = [];
	}
	if (row.length > 0) {
		if (total < maxRows) rows.push(row);
		total++;
	}
	return { rows, total };
}

/** Reads a quoted field starting after its opening quote; `""` is a literal quote. */
function readQuoted(text: string, start: number): { value: string; end: number } {
	let value = "";
	let cursor = start;
	for (;;) {
		const quote = text.indexOf('"', cursor);
		if (quote === -1) return { value: value + text.slice(cursor), end: text.length };
		value += text.slice(cursor, quote);
		if (text[quote + 1] !== '"') return { value, end: quote + 1 };
		value += '"';
		cursor = quote + 2;
	}
}

function nextBoundary(text: string, from: number, delimiter: string): number {
	for (let index = from; index < text.length; index++) {
		const char = text[index];
		if (char === delimiter || char === "\n" || char === "\r") return index;
	}
	return text.length;
}

/** The separator that splits the first lines into the same number of fields most often. */
function sniffDelimiter(text: string): string {
	const lines = leadingLines(text);
	let best: string = CSV_DELIMITERS[0];
	let bestScore = 0;
	for (const candidate of CSV_DELIMITERS) {
		const counts = lines.map((line) => countOutsideQuotes(line, candidate));
		const first = counts[0] ?? 0;
		if (first === 0) continue;
		const score = counts.filter((count) => count === first).length;
		if (score > bestScore) {
			best = candidate;
			bestScore = score;
		}
	}
	return best;
}

/** The first lines of the file, split on line breaks outside quotes. */
function leadingLines(text: string): string[] {
	const lines: string[] = [];
	let start = 0;
	let quoted = false;
	for (let index = 0; index < text.length && lines.length < SNIFF_LINES; index++) {
		const char = text[index];
		if (char === '"') quoted = !quoted;
		else if (!quoted && (char === "\n" || char === "\r")) {
			if (index > start) lines.push(text.slice(start, index));
			start = index + 1;
		}
	}
	if (lines.length < SNIFF_LINES && start < text.length) lines.push(text.slice(start));
	return lines;
}

function countOutsideQuotes(line: string, delimiter: string): number {
	let count = 0;
	let quoted = false;
	for (const char of line) {
		if (char === '"') quoted = !quoted;
		else if (!quoted && char === delimiter) count++;
	}
	return count;
}
