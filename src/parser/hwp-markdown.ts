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
		if (text.length > 0) blocks.push(text);
		for (const table of tables.get(`${paragraph.sectionIndex}:${paragraph.paragraphIndex}`) ?? []) {
			const rendered = renderTable(table);
			if (rendered.length > 0) blocks.push(rendered);
		}
	}
	return blocks.join("\n\n");
}

function cleanText(text: string): string {
	return text.replace(/[\u0000\uFFFC]/gu, "").trim();
}

function renderTable(table: HwpTable): string {
	const cells = table.cells.map((cell) => ({ ...cell, text: renderCell(cell) }));
	if (!cells.some((cell) => cell.text.length > 0)) return "";
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
	for (const cell of cells) {
		const row = rows.get(cell.row) ?? [];
		row.push(cell);
		rows.set(cell.row, row);
	}

	const renderedRows = [...rows.entries()]
		.sort(([left], [right]) => left - right)
		.map(([row, cellsInRow]) => {
			const text = cellsInRow
				.sort((left, right) => left.column - right.column)
				.map((cell) => cell.text)
				.filter((text) => text.length > 0)
				.join(" | ");
			return text.length > 0 ? `Row ${row + 1}: ${text}` : "";
		})
		.filter((row) => row.length > 0);

	return ["[Table]", ...renderedRows].join("\n");
}

function renderCell(cell: HwpTableCell): string {
	return cell.paragraphs.map(cleanText).filter(Boolean).join("<br>").replaceAll("|", "\\|");
}
