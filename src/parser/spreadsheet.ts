import { extname } from "node:path";
import * as XLSX from "xlsx";
import { ParseError } from "./errors.ts";
import { type CellKind, classifyText, renderWorkbook, type SheetCell, type SheetGrid } from "./spreadsheet-render.ts";
import { normalizeMarkdown } from "./text.ts";
import { type ParseDiagnostic, type ParseInput, type ParseOutput, Parser } from "./types.ts";

/** Workbook extensions SheetJS decodes for the spreadsheet parser. */
export const SPREADSHEET_EXTENSIONS = [".xlsx", ".xlsm", ".xlsb", ".xls", ".ods"] as const;

/** Rows read per sheet (header included). Beyond this the sheet is indexed partially, with a warning. */
const DEFAULT_MAX_ROWS_PER_SHEET = 100_000;
const ZIP_SIGNATURE = [0x50, 0x4b] as const;
const OLE_SIGNATURE = [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1] as const;
const ZIP_FORMATS: Readonly<Record<string, true>> = { xlsx: true, xlsm: true, xlsb: true, ods: true };

export interface SpreadsheetParserOptions {
	/** Rows read per sheet, header included. Default 100,000. */
	readonly maxRowsPerSheet?: number;
}

/**
 * Parses Excel (`.xlsx`, `.xlsm`, `.xlsb`, legacy `.xls`) and OpenDocument (`.ods`) workbooks with
 * SheetJS, then renders every row as one self-describing line (see `spreadsheet-render.ts`).
 *
 * Cell text is what the spreadsheet shows: cached formula results with their number formats applied
 * (`742,000`, `32.1%`, dates). A format that scales or hides the stored number (`#,##0,"천원"`) keeps
 * the stored value next to it, Korean-numeral formats (`[DBNum1]`) fall back to the plain number, and a
 * formula saved without a result is shown as its formula (`=SUM(B2:B3)`).
 */
export class SpreadsheetParser extends Parser {
	readonly name = "spreadsheet";
	readonly extensions = SPREADSHEET_EXTENSIONS;
	readonly version = `1+sheetjs-${XLSX.version}`;
	readonly #maxRowsPerSheet: number;

	constructor(options: SpreadsheetParserOptions = {}) {
		super();
		this.#maxRowsPerSheet = options.maxRowsPerSheet ?? DEFAULT_MAX_ROWS_PER_SHEET;
	}

	async parse(input: ParseInput): Promise<ParseOutput> {
		const format = extname(input.virtualPath).slice(1).toLowerCase();
		let workbook: XLSX.WorkBook;
		try {
			assertWorkbookSignature(format, input.bytes);
			workbook = XLSX.read(input.bytes, {
				type: "array",
				dense: true,
				cellFormula: true,
				cellNF: true,
				cellDates: false,
				sheetRows: this.#maxRowsPerSheet,
			});
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		}
		const date1904 = workbook.Workbook?.WBProps?.date1904 === true;
		const diagnostics: ParseDiagnostic[] = [];
		const grids = workbook.SheetNames.map((name, index) => {
			const sheet = workbook.Sheets[name];
			const fullRef = sheet?.["!fullref"];
			if (fullRef !== undefined) {
				const full = XLSX.utils.decode_range(fullRef);
				const total = full.e.r - full.s.r + 1;
				if (total > this.#maxRowsPerSheet) {
					diagnostics.push({
						code: "parser-warning",
						severity: "warning",
						message: `sheet "${name}": indexed the first ${this.#maxRowsPerSheet} of ${total} rows`,
					});
				}
			}
			return toGrid(name, sheet, workbook.Workbook?.Sheets?.[index]?.Hidden, date1904);
		});
		return {
			markdown: normalizeMarkdown(renderWorkbook(grids)),
			metadata: { parser: this.name, format },
			...(diagnostics.length > 0 ? { diagnostics } : {}),
		};
	}
}

/**
 * SheetJS reads almost any bytes as a one-cell CSV sheet, so the container is checked first: OOXML and
 * ODS are ZIP packages, and `.xls` is an OLE2 compound file or the HTML/XML a web export saves as `.xls`.
 */
function assertWorkbookSignature(format: string, bytes: Uint8Array): void {
	if (ZIP_FORMATS[format] === true) {
		if (!startsWith(bytes, ZIP_SIGNATURE)) throw new Error(`not a ${format} workbook: missing ZIP signature`);
		return;
	}
	if (startsWith(bytes, OLE_SIGNATURE)) return;
	const head = new TextDecoder()
		.decode(bytes.subarray(0, 64))
		.replace(/^\uFEFF/, "")
		.trimStart();
	if (!head.startsWith("<")) throw new Error(`not a ${format} workbook: neither an OLE2 file nor HTML/XML markup`);
}

function startsWith(bytes: Uint8Array, prefix: readonly number[]): boolean {
	return prefix.every((byte, index) => bytes[index] === byte);
}

function toGrid(
	name: string,
	sheet: XLSX.WorkSheet | undefined,
	hidden: number | undefined,
	date1904: boolean,
): SheetGrid {
	const data = (sheet?.["!data"] ?? []) as readonly (readonly (XLSX.CellObject | undefined)[] | undefined)[];
	return {
		name,
		hidden: hidden !== undefined && hidden !== 0,
		rows: data.map((row) => row?.map((cell) => (cell === undefined ? undefined : toCell(cell, date1904)))),
		merges: (sheet?.["!merges"] ?? []).map((merge) => ({
			top: merge.s.r,
			left: merge.s.c,
			bottom: merge.e.r,
			right: merge.e.c,
		})),
	};
}

function toCell(cell: XLSX.CellObject, date1904: boolean): SheetCell | undefined {
	const comment = (cell.c ?? [])
		.map((note) => note.t?.trim() ?? "")
		.filter((text) => text !== "")
		.join(" / ");
	const shown = displayed(cell, date1904);
	if (shown === undefined) return undefined;
	return comment === "" ? shown : { ...shown, comment };
}

function displayed(cell: XLSX.CellObject, date1904: boolean): SheetCell | undefined {
	if (cell.v === undefined || cell.v === null) {
		return cell.f === undefined ? undefined : { text: `=${cell.f}`, kind: "formula" };
	}
	switch (cell.t) {
		case "n":
			return typeof cell.v === "number" ? formatNumber(cell.v, cell.z, date1904) : textCell(String(cell.v));
		case "b":
			return { text: cell.v === true ? "TRUE" : "FALSE", kind: "boolean" };
		case "e":
			return { text: cell.w ?? "#ERROR", kind: "error" };
		case "d":
			return { text: cell.w ?? String(cell.v), kind: "date" };
		case "z":
			return undefined;
		default:
			return textCell(String(cell.v));
	}
}

function textCell(text: string): SheetCell | undefined {
	return text.trim() === "" ? undefined : { text, kind: classifyText(text) };
}

function formatNumber(value: number, format: XLSX.NumberFormat | undefined, date1904: boolean): SheetCell {
	const plain = formatGeneral(value);
	const code = typeof format === "string" ? format : format === undefined ? "General" : XLSX.SSF.get_table()[format];
	if (code === undefined || code === "General" || /\[DBNum\d\]/i.test(code)) return { text: plain, kind: "number" };
	const kind: CellKind = XLSX.SSF.is_date(code) ? "date" : "number";
	let shown: string;
	try {
		shown = XLSX.SSF.format(code, value, { date1904 }).trim();
	} catch {
		return { text: plain, kind };
	}
	if (shown === "") return { text: plain, kind };
	return { text: kind === "number" && hidesMagnitude(code) ? `${shown} (${plain})` : shown, kind };
}

/** Excel's General display without its 11-digit rounding: integers stay exact, floats lose binary noise. */
function formatGeneral(value: number): string {
	if (Number.isInteger(value)) return String(value);
	return String(Number(value.toPrecision(15)));
}

/** True when a trailing thousands comma divides the shown number (`#,##0,"천원"`, `0.0,,`). */
function hidesMagnitude(code: string): boolean {
	const bare = code
		.replace(/"[^"]*"/g, "")
		.replace(/\\./g, "")
		.replace(/\[[^\]]*\]/g, "");
	return /[0#?],+(?![0#?,])/.test(bare);
}
