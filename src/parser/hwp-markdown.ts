export interface HwpExtractedDocument {
	readonly paragraphs: readonly HwpBodyParagraph[];
	readonly tables: readonly HwpTable[];
}

export interface HwpBodyParagraph {
	readonly sectionIndex: number;
	readonly paragraphIndex: number;
	readonly text: string;
}

export interface HwpTable {
	readonly sectionIndex: number;
	readonly parentParagraphIndex: number;
	readonly controlIndex: number;
	readonly rowCount: number;
	readonly columnCount: number;
	readonly cells: readonly HwpTableCell[];
}

export interface HwpTableCell {
	readonly row: number;
	readonly column: number;
	readonly rowSpan: number;
	readonly columnSpan: number;
	readonly paragraphs: readonly string[];
}

export function renderHwpMarkdown(document: HwpExtractedDocument): string {
	const tables = new Map<string, HwpTable[]>();
	for (const table of document.tables) {
		const key = `${table.sectionIndex}:${table.parentParagraphIndex}`;
		const group = tables.get(key) ?? [];
		group.push(table);
		tables.set(key, group);
	}
	for (const group of tables.values()) {
		group.sort((left, right) => left.controlIndex - right.controlIndex);
	}

	const blocks: string[] = [];
	for (const paragraph of document.paragraphs) {
		const text = cleanText(paragraph.text);
		if (hasRenderableText(text)) blocks.push(text);
		for (const table of tables.get(`${paragraph.sectionIndex}:${paragraph.paragraphIndex}`) ?? []) {
			const rendered = renderTable(table);
			if (rendered.length > 0) blocks.push(rendered);
		}
	}
	return blocks.join("\n\n");
}

function cleanText(text: string): string {
	return text.replace(/[\u0000\uFFFC]/gu, "");
}

function hasRenderableText(text: string): boolean {
	return text.trim().length > 0;
}

function renderTable(table: HwpTable): string {
	const cells = table.cells.map((cell) => ({ ...cell, text: renderCell(cell) }));
	if (!cells.some((cell) => hasRenderableText(cell.text))) return "";
	if (cells.some((cell) => cell.rowSpan > 1 || cell.columnSpan > 1)) return renderMergedTable(cells);

	const cellsByPosition = new Map(cells.map((cell) => [`${cell.row}:${cell.column}`, cell.text]));
	const header = Array.from({ length: table.columnCount }, (_, column) => `Column ${column + 1}`);
	const rows = Array.from({ length: table.rowCount }, (_, row) =>
		Array.from({ length: table.columnCount }, (_, column) => cellsByPosition.get(`${row}:${column}`) ?? ""),
	);

	return [
		`| ${header.join(" | ")} |`,
		`| ${header.map(() => "---").join(" | ")} |`,
		...rows.map((row) => `| ${row.join(" | ")} |`),
	].join("\n");
}

function renderMergedTable(cells: readonly (HwpTableCell & { readonly text: string })[]): string {
	const rows = new Map<number, (HwpTableCell & { readonly text: string })[]>();
	const coveredPositions = new Set<string>();
	for (const cell of [...cells].sort((left, right) => left.row - right.row || left.column - right.column)) {
		const position = `${cell.row}:${cell.column}`;
		if (coveredPositions.has(position)) continue;

		const row = rows.get(cell.row) ?? [];
		row.push(cell);
		rows.set(cell.row, row);

		for (let rowIndex = cell.row; rowIndex < cell.row + Math.max(1, cell.rowSpan); rowIndex += 1) {
			for (
				let columnIndex = cell.column;
				columnIndex < cell.column + Math.max(1, cell.columnSpan);
				columnIndex += 1
			) {
				if (rowIndex !== cell.row || columnIndex !== cell.column)
					coveredPositions.add(`${rowIndex}:${columnIndex}`);
			}
		}
	}

	const renderedRows = [...rows.entries()]
		.sort(([left], [right]) => left - right)
		.map(([row, cellsInRow]) => {
			const text = cellsInRow
				.sort((left, right) => left.column - right.column)
				.map((cell) => cell.text)
				.filter(hasRenderableText)
				.join(" | ");
			return hasRenderableText(text) ? `Row ${row + 1}: ${text}` : "";
		})
		.filter((row) => row.length > 0);

	return ["[Table]", ...renderedRows].join("\n");
}

function renderCell(cell: HwpTableCell): string {
	return cell.paragraphs
		.map(cleanText)
		.filter(hasRenderableText)
		.join("<br>")
		.replaceAll("\\", "\\\\")
		.replace(/\r\n|\r|\n/gu, "<br>")
		.replaceAll("|", "\\|");
}
