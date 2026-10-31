/**
 * Renders decoded spreadsheet grids as retrieval-ready text.
 *
 * Every spreadsheet row becomes one line that names its sheet, cell range and table caption and
 * pairs each value with its column label:
 *
 *   [예산총괄!A4:F4 · 2026년 연구개발비 편성표 (단위: 천원)] 구분: 직접비 | 비목: 인건비 | 편성액: 742,000
 *
 * Chunkers split mirror text between lines, so a value never reaches a chunk without its labels.
 * A row too long for one retrieval chunk is split into several lines that each repeat the row key.
 *
 * Structure is inferred per block of consecutive non-empty rows:
 * - leading rows holding a single value (titles such as "… (단위: 천원)") become the block caption,
 *   and a title block directly above a table captions that table;
 * - the first remaining row is the header only when the evidence is strong: a two-level header under
 *   horizontally merged group cells, or a text header above at least three data rows with a column
 *   whose values are mostly numbers, dates or formulas. Otherwise the rows are kept as written, so a
 *   form-style label/value block is never relabelled with invented column names;
 * - a block without its own header that continues the previous table's columns reuses that header;
 * - a one-column vertical merge repeats its value on every covered row; horizontal merges and
 *   blocks merged across rows and columns (side notes) are shown once.
 */

export type CellKind = "text" | "number" | "date" | "boolean" | "error" | "formula";

export interface SheetCell {
	/** Display text as the spreadsheet shows it; may contain line breaks. */
	readonly text: string;
	readonly kind: CellKind;
	readonly comment?: string;
}

/** A merged range, 0-based and inclusive. */
export interface SheetMerge {
	readonly top: number;
	readonly left: number;
	readonly bottom: number;
	readonly right: number;
}

export interface SheetGrid {
	readonly name: string;
	readonly hidden?: boolean;
	/** Row-major cells. Rows and cells may be missing. */
	readonly rows: readonly (readonly (SheetCell | undefined)[] | undefined)[];
	readonly merges?: readonly SheetMerge[];
}

/**
 * Upper bound on a rendered line, in MinSync's `estimate_tokens_upper` units. It keeps several lines
 * per chunk under both MinSync's 4096-character recursive chunks and the 2048-token embedder budget.
 */
export const MAX_LINE_TOKENS = 768;

const MIN_HEADER_DATA_ROWS = 3;
const MIN_TYPED_SHARE = 0.8;
/** Excel's column limit (XFD). */
const MAX_COLUMNS = 16_384;
const MAX_HEADER_CELL_CHARS = 60;
const MAX_CAPTION_CHARS = 120;
const MAX_CAPTION_BLOCK_ROWS = 3;
const MAX_CAPTION_GAP_ROWS = 2;
const COLUMN_STRIDE = 1 << 20;
const NUMBER_TEXT =
	/^[-+]?[₩$€£¥]?\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s?(?:%|원|천원|만원|백만원|억원)?$|^[-+]?\.\d+%?$/;
const DATE_TEXT = /^\d{4}[-./]\d{1,2}(?:[-./]\d{1,2})?\.?$/;
const YEAR_TEXT = /^(?:19|20)\d{2}$/;

/** MinSync's `estimate_tokens_upper`: half-tokens 1 per ASCII char, 2 per 2-byte, 4 per wider char. */
export function estimateLineTokens(text: string): number {
	let halves = 0;
	for (const char of text) {
		const code = char.codePointAt(0) ?? 0;
		halves += code < 0x80 ? 1 : code < 0x800 ? 2 : 4;
	}
	return Math.ceil(halves / 2);
}

/** Kind of a value that arrived as text: numbers and dates typed as strings still count as data. */
export function classifyText(text: string): CellKind {
	const trimmed = text.trim();
	if (NUMBER_TEXT.test(trimmed)) return "number";
	if (DATE_TEXT.test(trimmed)) return "date";
	return "text";
}

export function renderWorkbook(sheets: readonly SheetGrid[]): string {
	return sheets.map(renderSheet).join("\n\n");
}

interface Resolved {
	readonly cell: SheetCell;
	readonly row: number;
	readonly col: number;
	readonly merge: SheetMerge | undefined;
}

interface Band {
	readonly top: number;
	readonly bottom: number;
	readonly left: number;
	readonly right: number;
}

interface Header {
	readonly labels: ReadonlyMap<number, string>;
	/** Columns whose data is mostly typed (numbers, dates, formulas); used to recognise continuations. */
	readonly typedColumns: readonly number[];
}

interface TableContext {
	readonly band: Band;
	readonly header: Header;
	readonly captions: readonly string[];
}

interface Item {
	readonly col: number;
	readonly label?: string;
	readonly value: string;
}

class SheetView {
	readonly rowCount: number;
	readonly colCount: number;
	readonly #grid: SheetGrid;
	readonly #merges = new Map<number, SheetMerge>();

	constructor(grid: SheetGrid) {
		this.#grid = grid;
		this.rowCount = grid.rows.length;
		const widest = grid.rows.reduce((max, row) => Math.max(max, row?.length ?? 0), 0);
		// A merged box reaches past the last stored cell when its other cells are empty (G2:H6 with H unset).
		const merges = (grid.merges ?? []).filter((merge) => merge.top < this.rowCount && merge.left < widest);
		this.colCount = merges.reduce((max, merge) => Math.max(max, Math.min(merge.right, MAX_COLUMNS - 1) + 1), widest);
		for (const merge of merges) {
			const bottom = Math.min(merge.bottom, this.rowCount - 1);
			const right = Math.min(merge.right, this.colCount - 1);
			for (let row = merge.top; row <= bottom; row++) {
				for (let col = merge.left; col <= right; col++) this.#merges.set(row * COLUMN_STRIDE + col, merge);
			}
		}
	}

	/** The value shown at a position: its own cell, or the top-left cell of the merged range covering it. */
	resolve(row: number, col: number): Resolved | undefined {
		const merge = this.#merges.get(row * COLUMN_STRIDE + col);
		const anchorRow = merge?.top ?? row;
		const anchorCol = merge?.left ?? col;
		const cell = this.#grid.rows[anchorRow]?.[anchorCol];
		if (cell === undefined || clean(cell.text) === "") return undefined;
		return { cell, row: anchorRow, col: anchorCol, merge };
	}
}

function renderSheet(grid: SheetGrid): string {
	const view = new SheetView(grid);
	const heading = `## ${grid.name}${grid.hidden === true ? " (hidden)" : ""}`;
	const blocks: string[] = [];
	let pendingCaptions: readonly string[] = [];
	let previous: TableContext | undefined;
	const bands = findBands(view);
	for (const [index, band] of bands.entries()) {
		const rows = range(band.top, band.bottom);
		if (rows.every((row) => singleValue(view, row, band) !== undefined)) {
			const next = bands[index + 1];
			const captionsNext =
				next !== undefined &&
				rows.length <= MAX_CAPTION_BLOCK_ROWS &&
				next.top - band.bottom - 1 <= MAX_CAPTION_GAP_ROWS &&
				!range(next.top, next.bottom).every((row) => singleValue(view, row, next) !== undefined);
			if (captionsNext) {
				pendingCaptions = rows.map((row) => singleValue(view, row, band) as string);
				continue;
			}
			blocks.push(
				rows
					.map((row) => `[${grid.name}!${cellRange(band.left, band.right, row)}] ${singleValue(view, row, band)}`)
					.join("\n"),
			);
			continue;
		}
		let start = 0;
		const ownCaptions: string[] = [];
		while (start < rows.length && singleValue(view, rows[start] as number, band) !== undefined) {
			ownCaptions.push(singleValue(view, rows[start] as number, band) as string);
			start++;
		}
		const body = rows.slice(start);
		let captions = [...pendingCaptions, ...ownCaptions];
		pendingCaptions = [];
		const detected = detectHeader(view, band, body);
		let header = detected?.header;
		let dataRows = detected === undefined ? body : body.slice(detected.headerRows);
		if (header === undefined && previous !== undefined && continuesTable(view, band, body, previous)) {
			header = previous.header;
			dataRows = body;
			if (captions.length === 0) captions = [...previous.captions];
		}
		if (header !== undefined) previous = { band, header, captions };
		const lines = dataRows.flatMap((row) =>
			renderRow(grid.name, view, band, row, dataRows[0] as number, header, captions),
		);
		if (lines.length > 0) blocks.push(lines.join("\n"));
	}
	return [heading, ...blocks].join("\n\n");
}

function findBands(view: SheetView): Band[] {
	const bands: Band[] = [];
	let current: { top: number; bottom: number; left: number; right: number } | undefined;
	for (let row = 0; row < view.rowCount; row++) {
		let left = Number.POSITIVE_INFINITY;
		let right = Number.NEGATIVE_INFINITY;
		for (let col = 0; col < view.colCount; col++) {
			if (view.resolve(row, col) === undefined) continue;
			left = Math.min(left, col);
			right = Math.max(right, col);
		}
		if (right < left) {
			if (current !== undefined) bands.push(current);
			current = undefined;
			continue;
		}
		if (current === undefined) current = { top: row, bottom: row, left, right };
		else {
			current.bottom = row;
			current.left = Math.min(current.left, left);
			current.right = Math.max(current.right, right);
		}
	}
	if (current !== undefined) bands.push(current);
	return bands;
}

/** Distinct values shown in a row, keyed by the cell (or merged range) that owns them. */
function rowValues(view: SheetView, row: number, band: Band): Resolved[] {
	const values: Resolved[] = [];
	const seen = new Set<number>();
	for (let col = band.left; col <= band.right; col++) {
		const resolved = view.resolve(row, col);
		if (resolved === undefined) continue;
		const key = resolved.row * COLUMN_STRIDE + resolved.col;
		if (seen.has(key)) continue;
		seen.add(key);
		values.push(resolved);
	}
	return values;
}

/** The text of a row that shows exactly one value (a title, unit line or note), else undefined. */
function singleValue(view: SheetView, row: number, band: Band): string | undefined {
	const values = rowValues(view, row, band);
	return values.length === 1 ? clean((values[0] as Resolved).cell.text) : undefined;
}

function detectHeader(
	view: SheetView,
	band: Band,
	body: readonly number[],
): { readonly header: Header; readonly headerRows: number } | undefined {
	const [first, second] = body;
	if (first === undefined) return undefined;
	const firstValues = rowValues(view, first, band);
	if (firstValues.length < 2 || !firstValues.every((value) => headerLike(value.cell))) return undefined;

	const grouped = firstValues.some(
		(value) => value.row === first && value.merge !== undefined && value.merge.right > value.merge.left,
	);
	if (grouped && second !== undefined && body.length >= 3) {
		const childValues = rowValues(view, second, band).filter((value) => value.row === second);
		if (childValues.length >= 2 && childValues.every((value) => headerLike(value.cell))) {
			const labels = new Map<number, string>();
			for (let col = band.left; col <= band.right; col++) {
				const parent = labelAt(view, first, col);
				const child = labelAt(view, second, col);
				const label = child !== "" && child !== parent ? (parent === "" ? child : `${parent} > ${child}`) : parent;
				if (label !== "") labels.set(col, label);
			}
			const data = body.slice(2);
			return { header: { labels, typedColumns: typedColumns(view, band, data, MIN_TYPED_SHARE, 1) }, headerRows: 2 };
		}
	}

	const data = body.slice(1);
	if (data.length < MIN_HEADER_DATA_ROWS) return undefined;
	const typed = typedColumns(view, band, data, MIN_TYPED_SHARE, MIN_HEADER_DATA_ROWS);
	if (typed.length === 0) return undefined;
	const labels = new Map<number, string>();
	for (let col = band.left; col <= band.right; col++) {
		const label = labelAt(view, first, col);
		if (label !== "") labels.set(col, label);
	}
	return { header: { labels, typedColumns: typed }, headerRows: 1 };
}

function headerLike(cell: SheetCell): boolean {
	const text = clean(cell.text);
	if (cell.kind === "number") return YEAR_TEXT.test(text);
	return cell.kind === "text" && text.length <= MAX_HEADER_CELL_CHARS;
}

function labelAt(view: SheetView, row: number, col: number): string {
	const resolved = view.resolve(row, col);
	return resolved === undefined ? "" : clean(resolved.cell.text);
}

/** Columns whose non-empty values in `rows` are mostly typed data (not free text). */
function typedColumns(view: SheetView, band: Band, rows: readonly number[], share: number, minimum: number): number[] {
	const columns: number[] = [];
	for (let col = band.left; col <= band.right; col++) {
		let filled = 0;
		let typed = 0;
		for (const row of rows) {
			const resolved = view.resolve(row, col);
			if (resolved === undefined || resolved.col !== col) continue;
			filled++;
			if (resolved.cell.kind !== "text") typed++;
		}
		if (filled >= minimum && typed / filled >= share) columns.push(col);
	}
	return columns;
}

function continuesTable(view: SheetView, band: Band, body: readonly number[], previous: TableContext): boolean {
	if (band.left < previous.band.left || band.right > previous.band.right) return false;
	const columns = previous.header.typedColumns.filter((col) => col >= band.left && col <= band.right);
	if (columns.length === 0) return false;
	const typed = typedColumns(view, band, body, MIN_TYPED_SHARE, 1);
	return columns.every((col) => typed.includes(col));
}

function renderRow(
	sheet: string,
	view: SheetView,
	band: Band,
	row: number,
	firstDataRow: number,
	header: Header | undefined,
	captions: readonly string[],
): string[] {
	const items: Item[] = [];
	let spanning: number | undefined;
	for (let col = band.left; col <= band.right; col++) {
		const resolved = view.resolve(row, col);
		if (resolved === undefined || resolved.col !== col) continue;
		if (resolved.row !== row && (resolved.row < firstDataRow || !fillsDown(resolved.merge))) continue;
		const comment = resolved.row === row && resolved.cell.comment !== undefined ? clean(resolved.cell.comment) : "";
		const text = clean(resolved.cell.text);
		const value = comment === "" ? text : `${text} (comment: ${comment})`;
		if (resolved.merge !== undefined && resolved.merge.right > resolved.merge.left) spanning = col;
		items.push(
			header === undefined ? { col, value } : { col, label: header.labels.get(col) ?? columnName(col), value },
		);
	}
	if (items.length === 0) return [];
	// A lone value merged across columns is a note ("※ 부가세 포함"), not a value of its first column.
	const [only] = items;
	if (items.length === 1 && only !== undefined && only.col === spanning)
		items[0] = { col: only.col, value: only.value };
	const caption = captionText(captions);
	const prefix = (left: number, right: number) => `[${sheet}!${cellRange(left, right, row)}${caption}] `;
	const whole = prefix(band.left, band.right) + items.map(itemText).join(" | ");
	if (estimateLineTokens(whole) <= MAX_LINE_TOKENS) return [whole];
	return splitRow(items, prefix);
}

/** Splits an over-long row into lines that each start with the row key (its first value). */
function splitRow(items: readonly Item[], prefix: (left: number, right: number) => string): string[] {
	const [key, ...rest] = items as [Item, ...Item[]];
	const keyText = itemText(key);
	const lines: string[] = [];
	let piece: Item[] = [];
	const flush = () => {
		if (piece.length === 0) return;
		const head = lines.length === 0 ? "" : `${keyText} | `;
		const first = lines.length === 0 ? key : (piece[0] as Item);
		lines.push(prefix(first.col, (piece.at(-1) as Item).col) + head + piece.map(itemText).join(" | "));
		piece = [];
	};
	piece.push(key);
	for (const item of rest) {
		for (const part of fitItem(item, prefix, keyText)) {
			const candidate = [...piece, part];
			const head = lines.length === 0 ? "" : `${keyText} | `;
			const start = lines.length === 0 ? key : (candidate[0] as Item);
			const text = prefix(start.col, part.col) + head + candidate.map(itemText).join(" | ");
			if (piece.length > 0 && estimateLineTokens(text) > MAX_LINE_TOKENS) flush();
			piece.push(part);
		}
	}
	flush();
	return lines;
}

/** Cuts one value that cannot fit a line on its own into parts that each keep the column label. */
function fitItem(item: Item, prefix: (left: number, right: number) => string, keyText: string): Item[] {
	const overhead = estimateLineTokens(`${prefix(item.col, item.col)}${keyText} | ${item.label ?? ""}: `);
	const budget = Math.max(16, MAX_LINE_TOKENS - overhead);
	if (estimateLineTokens(item.value) <= budget) return [item];
	const parts: Item[] = [];
	let current = "";
	for (const char of item.value) {
		if (current !== "" && estimateLineTokens(current + char) > budget) {
			parts.push({ ...item, value: current });
			current = "";
		}
		current += char;
	}
	if (current !== "") parts.push({ ...item, value: current });
	return parts;
}

/**
 * A one-column vertical merge is a row key shared by the rows it spans (`구분: 직접비`), so every row
 * repeats it. A block merged across several columns is a note or picture box; it is shown once.
 */
function fillsDown(merge: SheetMerge | undefined): boolean {
	return merge !== undefined && merge.left === merge.right;
}

function itemText(item: Item): string {
	return item.label === undefined ? item.value : `${item.label}: ${item.value}`;
}

function captionText(captions: readonly string[]): string {
	if (captions.length === 0) return "";
	const joined = captions.join(" · ");
	return ` · ${joined.length > MAX_CAPTION_CHARS ? `${joined.slice(0, MAX_CAPTION_CHARS - 1)}…` : joined}`;
}

/** One line per value: line breaks become " / ", and a literal "|" is escaped so it never reads as a separator. */
function clean(text: string): string {
	return text
		.trim()
		.replace(/\s*(?:\r\n|\r|\n)\s*/g, " / ")
		.replace(/[\t\f\v]+/g, " ")
		.replace(/\|/g, "\\|");
}

function cellRange(left: number, right: number, row: number): string {
	const start = `${columnName(left)}${row + 1}`;
	return left === right ? start : `${start}:${columnName(right)}${row + 1}`;
}

/** Spreadsheet column letters: 0 → A, 25 → Z, 26 → AA. */
function columnName(col: number): string {
	let name = "";
	for (let rest = col + 1; rest > 0; rest = Math.floor((rest - 1) / 26)) {
		name = String.fromCharCode(65 + ((rest - 1) % 26)) + name;
	}
	return name;
}

function range(first: number, last: number): number[] {
	return Array.from({ length: last - first + 1 }, (_, index) => first + index);
}
