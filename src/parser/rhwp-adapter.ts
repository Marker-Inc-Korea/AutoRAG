import { readFile } from "node:fs/promises";
import initRhwp, { HwpDocument } from "@rhwp/core";
import type { HwpBodyParagraph, HwpExtractedDocument, HwpTable, HwpTableCell } from "./hwp-markdown.ts";

export interface HwpExtractionLimits {
	readonly maxSections?: number;
	readonly maxBodyParagraphs?: number;
	readonly maxTables?: number;
	readonly maxCells?: number;
	readonly maxCellParagraphs?: number;
	readonly maxCharacters?: number;
}

export type HwpExtractor = (bytes: Uint8Array) => Promise<HwpExtractedDocument>;

export interface RhwpDocumentApi {
	free(): void;
	getCellInfo(sectionIndex: number, parentParagraphIndex: number, controlIndex: number, cellIndex: number): string;
	getCellParagraphCount(
		sectionIndex: number,
		parentParagraphIndex: number,
		controlIndex: number,
		cellIndex: number,
	): number;
	getCellParagraphLength(
		sectionIndex: number,
		parentParagraphIndex: number,
		controlIndex: number,
		cellIndex: number,
		cellParagraphIndex: number,
	): number;
	getPageControlLayout(pageNumber: number): string;
	getParagraphCount(sectionIndex: number): number;
	getParagraphLength(sectionIndex: number, paragraphIndex: number): number;
	getSectionCount(): number;
	getTableDimensions(sectionIndex: number, parentParagraphIndex: number, controlIndex: number): string;
	getTextInCell(
		sectionIndex: number,
		parentParagraphIndex: number,
		controlIndex: number,
		cellIndex: number,
		cellParagraphIndex: number,
		characterOffset: number,
		count: number,
	): string;
	getTextRange(sectionIndex: number, paragraphIndex: number, characterOffset: number, count: number): string;
	pageCount(): number;
}

export interface RhwpRuntime {
	initialize(): Promise<void>;
	open(bytes: Uint8Array): RhwpDocumentApi;
}

type HwpExtractionLimitName = keyof Required<HwpExtractionLimits>;
type EffectiveHwpExtractionLimits = Required<HwpExtractionLimits>;

const DEFAULT_LIMITS: EffectiveHwpExtractionLimits = {
	maxSections: 256,
	maxBodyParagraphs: 200_000,
	maxTables: 20_000,
	maxCells: 500_000,
	maxCellParagraphs: 1_000_000,
	maxCharacters: 20_000_000,
};

class HwpExtractionBudgetError extends Error {
	readonly code = "HWP_EXTRACTION_BUDGET_EXCEEDED";
	readonly limit: HwpExtractionLimitName;
	readonly actual: number;
	readonly maximum: number;

	constructor(limit: HwpExtractionLimitName, actual: number, maximum: number) {
		super(`HWP extraction exceeded ${limit} budget of ${maximum} (received ${actual})`);
		this.name = "HwpExtractionBudgetError";
		this.limit = limit;
		this.actual = actual;
		this.maximum = maximum;
	}
}

const defaultRhwpRuntime: RhwpRuntime = {
	async initialize(): Promise<void> {
		const rhwpModuleUrl = import.meta.resolve("@rhwp/core");
		const wasmBytes = await readFile(new URL("./rhwp_bg.wasm", rhwpModuleUrl));
		await initRhwp({ module_or_path: wasmBytes });
	},
	open(bytes: Uint8Array): RhwpDocumentApi {
		return new HwpDocument(bytes);
	},
};

const initializeDefaultRhwpRuntime = createRuntimeInitializer(defaultRhwpRuntime);

export function createRhwpExtractor(
	runtime: RhwpRuntime = defaultRhwpRuntime,
	overrides: HwpExtractionLimits = {},
): HwpExtractor {
	const limits = resolveLimits(overrides);
	const initialize = runtime === defaultRhwpRuntime ? initializeDefaultRhwpRuntime : createRuntimeInitializer(runtime);

	return async (bytes) => {
		await initialize();

		const document = runtime.open(bytes);
		try {
			return extractDocument(document, limits);
		} finally {
			document.free();
		}
	};
}

export function extractHwpWithRhwp(bytes: Uint8Array, limits: HwpExtractionLimits = {}): Promise<HwpExtractedDocument> {
	return createRhwpExtractor(defaultRhwpRuntime, limits)(bytes);
}

function createRuntimeInitializer(runtime: RhwpRuntime): () => Promise<void> {
	let initialization: Promise<void> | undefined;
	return () => {
		if (initialization === undefined) {
			const attempt = runtime.initialize();
			initialization = attempt.catch((error: unknown) => {
				initialization = undefined;
				throw error;
			});
		}
		return initialization;
	};
}

function extractDocument(document: RhwpDocumentApi, limits: EffectiveHwpExtractionLimits): HwpExtractedDocument {
	const paragraphs: HwpBodyParagraph[] = [];
	const paragraphCounts: number[] = [];
	const counters = { characters: 0, tables: 0, cells: 0, cellParagraphs: 0 };
	const sectionCount = readCount(document.getSectionCount(), "section count");
	checkBudget("maxSections", sectionCount, limits);
	for (let sectionIndex = 0; sectionIndex < sectionCount; sectionIndex += 1) {
		const paragraphCount = readCount(
			document.getParagraphCount(sectionIndex),
			`paragraph count for section ${sectionIndex}`,
		);
		paragraphCounts.push(paragraphCount);
		checkBudget("maxBodyParagraphs", paragraphs.length + paragraphCount, limits);
		for (let paragraphIndex = 0; paragraphIndex < paragraphCount; paragraphIndex += 1) {
			const length = readCount(
				document.getParagraphLength(sectionIndex, paragraphIndex),
				`paragraph length at ${sectionIndex}:${paragraphIndex}`,
			);
			const text = readText(
				document.getTextRange(sectionIndex, paragraphIndex, 0, length),
				`paragraph text at ${sectionIndex}:${paragraphIndex}`,
			);
			counters.characters += [...text].length;
			checkBudget("maxCharacters", counters.characters, limits);
			paragraphs.push({
				sectionIndex,
				paragraphIndex,
				text,
			});
		}
	}
	return { paragraphs, tables: extractTables(document, limits, counters, paragraphCounts) };
}

interface ExtractionCounters {
	characters: number;
	tables: number;
	cells: number;
	cellParagraphs: number;
}

interface TableCoordinates {
	sectionIndex: number;
	parentParagraphIndex: number;
	controlIndex: number;
}

function extractTables(
	document: RhwpDocumentApi,
	limits: EffectiveHwpExtractionLimits,
	counters: ExtractionCounters,
	paragraphCounts: readonly number[],
): HwpTable[] {
	const tables: HwpTable[] = [];
	const seen = new Set<string>();
	const pageCount = readCount(document.pageCount(), "page count");
	for (let pageIndex = 0; pageIndex < pageCount; pageIndex += 1) {
		const layout = readJsonRecord(document.getPageControlLayout(pageIndex), `page ${pageIndex} control layout`);
		const controls = readArray(layout.controls, `controls for page ${pageIndex}`);
		for (let layoutIndex = 0; layoutIndex < controls.length; layoutIndex += 1) {
			const control = readRecord(controls[layoutIndex], `control ${layoutIndex} on page ${pageIndex}`);
			if (typeof control.type !== "string") {
				throw new TypeError(`rhwp returned an invalid control type at page ${pageIndex}:${layoutIndex}`);
			}
			if (control.type !== "table") continue;
			const label = `table at page ${pageIndex}:${layoutIndex}`;
			const coordinates = readTableCoordinates(control, label, paragraphCounts, limits);
			if (coordinates === undefined) continue;
			const key = `${coordinates.sectionIndex}:${coordinates.parentParagraphIndex}:${coordinates.controlIndex}`;
			if (seen.has(key)) continue;
			seen.add(key);
			counters.tables += 1;
			checkBudget("maxTables", counters.tables, limits);
			tables.push(extractTable(document, coordinates, limits, counters));
		}
	}
	return tables;
}

function readTableCoordinates(
	control: Record<string, unknown>,
	label: string,
	paragraphCounts: readonly number[],
	limits: EffectiveHwpExtractionLimits,
): TableCoordinates | undefined {
	const hasSectionIndex = Object.hasOwn(control, "secIdx");
	const hasParagraphIndex = Object.hasOwn(control, "paraIdx");
	const hasControlIndex = Object.hasOwn(control, "controlIdx");
	const coordinateCount = Number(hasSectionIndex) + Number(hasParagraphIndex) + Number(hasControlIndex);

	if (coordinateCount === 0) {
		validateDeferredNestedTable(control, label, limits);
		return undefined;
	}
	if (coordinateCount !== 3) {
		throw new TypeError(`rhwp returned a partial coordinate set for ${label}`);
	}
	if (Object.hasOwn(control, "cellPath") || Object.hasOwn(control, "parentParaIdx")) {
		throw new TypeError(`rhwp returned conflicting top-level and nested context for ${label}`);
	}

	const coordinates = {
		sectionIndex: readCount(control.secIdx, `secIdx for ${label}`),
		parentParagraphIndex: readCount(control.paraIdx, `paraIdx for ${label}`),
		controlIndex: readCount(control.controlIdx, `controlIdx for ${label}`),
	};
	const paragraphCount = paragraphCounts[coordinates.sectionIndex];
	if (paragraphCount === undefined) {
		throw new TypeError(`rhwp returned a section coordinate outside the announced document range for ${label}`);
	}
	if (coordinates.parentParagraphIndex >= paragraphCount) {
		throw new TypeError(
			`rhwp returned a parent paragraph coordinate outside the announced section range for ${label}`,
		);
	}
	return coordinates;
}

function validateDeferredNestedTable(
	control: Record<string, unknown>,
	label: string,
	limits: EffectiveHwpExtractionLimits,
): void {
	if (!Object.hasOwn(control, "cellPath") && !Object.hasOwn(control, "parentParaIdx")) {
		validateDeferredNestedLayout(control, label, limits);
		return;
	}
	readCount(control.parentParaIdx, `parentParaIdx for deferred nested ${label}`);
	const cellPath = readArray(control.cellPath, `cellPath for deferred nested ${label}`);
	if (cellPath.length === 0) {
		throw new TypeError(`rhwp returned an empty cellPath for deferred nested ${label}`);
	}
	checkBudget("maxCells", cellPath.length, limits);
	for (let pathIndex = 0; pathIndex < cellPath.length; pathIndex += 1) {
		const entry = readRecord(cellPath[pathIndex], `cellPath entry ${pathIndex} for deferred nested ${label}`);
		readCount(entry.controlIndex, `controlIndex in cellPath entry ${pathIndex} for deferred nested ${label}`);
		readCount(entry.cellIndex, `cellIndex in cellPath entry ${pathIndex} for deferred nested ${label}`);
		readCount(entry.cellParaIndex, `cellParaIndex in cellPath entry ${pathIndex} for deferred nested ${label}`);
	}
}

function validateDeferredNestedLayout(
	control: Record<string, unknown>,
	label: string,
	limits: EffectiveHwpExtractionLimits,
): void {
	readFiniteNumber(control.x, `x for deferred nested ${label}`);
	readFiniteNumber(control.y, `y for deferred nested ${label}`);
	readPositiveFiniteNumber(control.w, `w for deferred nested ${label}`);
	readPositiveFiniteNumber(control.h, `h for deferred nested ${label}`);
	const rowCount = readPositiveCount(control.rowCount, `rowCount for deferred nested ${label}`);
	const columnCount = readPositiveCount(control.colCount, `colCount for deferred nested ${label}`);
	const logicalCellCount = readLogicalCellCount(rowCount, columnCount, limits);
	readCount(control.plane, `plane for deferred nested ${label}`);
	readSafeInteger(control.zOrder, `zOrder for deferred nested ${label}`);
	readCount(control.stableIndex, `stableIndex for deferred nested ${label}`);
	const cells = readArray(control.cells, `cells for deferred nested ${label}`);
	if (cells.length === 0) {
		throw new TypeError(`rhwp returned empty cells for deferred nested ${label}`);
	}
	checkBudget("maxCells", cells.length, limits);
	if (cells.length > logicalCellCount) {
		throw new TypeError(`rhwp returned too many cells for deferred nested ${label}`);
	}

	const anchors = new Set<string>();
	const cellIndexes = new Set<number>();
	const coveredCoordinates = new Set<string>();
	for (let cellIndex = 0; cellIndex < cells.length; cellIndex += 1) {
		const cell = readRecord(cells[cellIndex], `cell ${cellIndex} for deferred nested ${label}`);
		readFiniteNumber(cell.x, `x for cell ${cellIndex} in deferred nested ${label}`);
		readFiniteNumber(cell.y, `y for cell ${cellIndex} in deferred nested ${label}`);
		readPositiveFiniteNumber(cell.w, `w for cell ${cellIndex} in deferred nested ${label}`);
		readPositiveFiniteNumber(cell.h, `h for cell ${cellIndex} in deferred nested ${label}`);
		const row = readCount(cell.row, `row for cell ${cellIndex} in deferred nested ${label}`);
		const column = readCount(cell.col, `col for cell ${cellIndex} in deferred nested ${label}`);
		const rowSpan = readPositiveCount(cell.rowSpan, `rowSpan for cell ${cellIndex} in deferred nested ${label}`);
		const columnSpan = readPositiveCount(cell.colSpan, `colSpan for cell ${cellIndex} in deferred nested ${label}`);
		const runtimeCellIndex = readCount(cell.cellIdx, `cellIdx for cell ${cellIndex} in deferred nested ${label}`);
		if (row >= rowCount || rowSpan > rowCount - row) {
			throw new TypeError(
				`rhwp returned an out-of-range row span for cell ${cellIndex} in deferred nested ${label}`,
			);
		}
		if (column >= columnCount || columnSpan > columnCount - column) {
			throw new TypeError(
				`rhwp returned an out-of-range col span for cell ${cellIndex} in deferred nested ${label}`,
			);
		}
		if (cellIndexes.has(runtimeCellIndex)) {
			throw new TypeError(`rhwp returned a duplicate cellIdx for deferred nested ${label}`);
		}
		cellIndexes.add(runtimeCellIndex);

		const anchor = `${row}:${column}`;
		if (anchors.has(anchor)) {
			throw new TypeError(`rhwp returned a duplicate cell anchor for deferred nested ${label}`);
		}
		anchors.add(anchor);
		for (let coveredRow = row; coveredRow < row + rowSpan; coveredRow += 1) {
			for (let coveredColumn = column; coveredColumn < column + columnSpan; coveredColumn += 1) {
				const coordinate = `${coveredRow}:${coveredColumn}`;
				if (coveredCoordinates.has(coordinate)) {
					throw new TypeError(`rhwp returned overlapping cell spans for deferred nested ${label}`);
				}
				coveredCoordinates.add(coordinate);
			}
		}
	}
}

function extractTable(
	document: RhwpDocumentApi,
	coordinates: TableCoordinates,
	limits: EffectiveHwpExtractionLimits,
	counters: ExtractionCounters,
): HwpTable {
	const { sectionIndex, parentParagraphIndex, controlIndex } = coordinates;
	const key = `${sectionIndex}:${parentParagraphIndex}:${controlIndex}`;
	const dimensions = readJsonRecord(
		document.getTableDimensions(sectionIndex, parentParagraphIndex, controlIndex),
		`table dimensions at ${key}`,
	);
	const rowCount = readPositiveCount(dimensions.rowCount, `rowCount for table ${key}`);
	const columnCount = readPositiveCount(dimensions.colCount, `colCount for table ${key}`);
	const cellCount = readCount(dimensions.cellCount, `cellCount for table ${key}`);
	const logicalCellCount = chargeLogicalCells(rowCount, columnCount, counters, limits);
	if (cellCount === 0 || cellCount > logicalCellCount) {
		throw new TypeError(
			`rhwp returned an invalid cellCount for table ${key}; expected 1..${logicalCellCount}, received ${cellCount}`,
		);
	}

	interface CellGeometry {
		readonly row: number;
		readonly column: number;
		readonly rowSpan: number;
		readonly columnSpan: number;
	}
	const cellGeometry: CellGeometry[] = [];
	const anchors = new Set<number>();
	const coveredCoordinates = new Set<number>();
	for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
		const cellInfo = readJsonRecord(
			document.getCellInfo(sectionIndex, parentParagraphIndex, controlIndex, cellIndex),
			`cell metadata at ${key}:${cellIndex}`,
		);
		const row = readCount(cellInfo.row, `row for cell ${key}:${cellIndex}`);
		const column = readCount(cellInfo.col, `col for cell ${key}:${cellIndex}`);
		const rowSpan = readPositiveCount(cellInfo.rowSpan, `rowSpan for cell ${key}:${cellIndex}`);
		const columnSpan = readPositiveCount(cellInfo.colSpan, `colSpan for cell ${key}:${cellIndex}`);
		if (row >= rowCount) {
			throw new TypeError(`rhwp returned a row outside the table range for cell ${key}:${cellIndex}`);
		}
		if (column >= columnCount) {
			throw new TypeError(`rhwp returned a col outside the table range for cell ${key}:${cellIndex}`);
		}
		if (rowSpan > rowCount - row) {
			throw new TypeError(`rhwp returned a rowSpan outside the table range for cell ${key}:${cellIndex}`);
		}
		if (columnSpan > columnCount - column) {
			throw new TypeError(`rhwp returned a colSpan outside the table range for cell ${key}:${cellIndex}`);
		}

		const anchor = row * columnCount + column;
		if (anchors.has(anchor)) {
			throw new TypeError(`rhwp returned a duplicate cell anchor for table ${key} at ${row}:${column}`);
		}
		anchors.add(anchor);
		for (let coveredRow = row; coveredRow < row + rowSpan; coveredRow += 1) {
			for (let coveredColumn = column; coveredColumn < column + columnSpan; coveredColumn += 1) {
				const coordinate = coveredRow * columnCount + coveredColumn;
				if (coveredCoordinates.has(coordinate)) {
					throw new TypeError(
						`rhwp returned overlapping cell spans for table ${key} at ${coveredRow}:${coveredColumn}`,
					);
				}
				coveredCoordinates.add(coordinate);
			}
		}
		cellGeometry.push({ row, column, rowSpan, columnSpan });
	}
	if (coveredCoordinates.size !== logicalCellCount) {
		throw new TypeError(
			`rhwp returned cellCount ${cellCount} that covers ${coveredCoordinates.size} of ${logicalCellCount} logical grid coordinates for table ${key}`,
		);
	}

	const cells: HwpTableCell[] = [];
	for (let cellIndex = 0; cellIndex < cellGeometry.length; cellIndex += 1) {
		const geometry = cellGeometry[cellIndex];
		if (geometry === undefined) throw new TypeError(`rhwp omitted cell geometry at ${key}:${cellIndex}`);
		const paragraphCount = readCount(
			document.getCellParagraphCount(sectionIndex, parentParagraphIndex, controlIndex, cellIndex),
			`cell paragraph count at ${key}:${cellIndex}`,
		);
		counters.cellParagraphs += paragraphCount;
		checkBudget("maxCellParagraphs", counters.cellParagraphs, limits);

		const paragraphs: string[] = [];
		for (let cellParagraphIndex = 0; cellParagraphIndex < paragraphCount; cellParagraphIndex += 1) {
			const length = readCount(
				document.getCellParagraphLength(
					sectionIndex,
					parentParagraphIndex,
					controlIndex,
					cellIndex,
					cellParagraphIndex,
				),
				`cell paragraph length at ${key}:${cellIndex}:${cellParagraphIndex}`,
			);
			const text = readText(
				document.getTextInCell(
					sectionIndex,
					parentParagraphIndex,
					controlIndex,
					cellIndex,
					cellParagraphIndex,
					0,
					length,
				),
				`cell paragraph text at ${key}:${cellIndex}:${cellParagraphIndex}`,
			);
			counters.characters += [...text].length;
			checkBudget("maxCharacters", counters.characters, limits);
			paragraphs.push(text);
		}

		cells.push({
			...geometry,
			paragraphs,
		});
	}

	return {
		sectionIndex,
		parentParagraphIndex,
		controlIndex,
		rowCount,
		columnCount,
		cells,
	};
}

function chargeLogicalCells(
	rowCount: number,
	columnCount: number,
	counters: ExtractionCounters,
	limits: EffectiveHwpExtractionLimits,
): number {
	const logicalCellCount = readLogicalCellCount(rowCount, columnCount, limits);
	if (logicalCellCount > limits.maxCells - counters.cells) {
		throw new HwpExtractionBudgetError("maxCells", counters.cells + logicalCellCount, limits.maxCells);
	}
	counters.cells += logicalCellCount;
	return logicalCellCount;
}

function readLogicalCellCount(rowCount: number, columnCount: number, limits: EffectiveHwpExtractionLimits): number {
	if (rowCount > Math.floor(limits.maxCells / columnCount)) {
		throw new HwpExtractionBudgetError("maxCells", rowCount * columnCount, limits.maxCells);
	}
	return rowCount * columnCount;
}

function resolveLimits(overrides: HwpExtractionLimits): EffectiveHwpExtractionLimits {
	const limits = { ...DEFAULT_LIMITS };
	for (const name of Object.keys(DEFAULT_LIMITS) as HwpExtractionLimitName[]) {
		const value = overrides[name];
		if (value === undefined) continue;
		if (!Number.isSafeInteger(value) || value <= 0) {
			throw new TypeError(`${name} must be a positive finite safe integer`);
		}
		limits[name] = value;
	}
	return limits;
}

function checkBudget(name: HwpExtractionLimitName, actual: number, limits: EffectiveHwpExtractionLimits): void {
	if (actual > limits[name]) throw new HwpExtractionBudgetError(name, actual, limits[name]);
}

function readCount(value: unknown, label: string): number {
	if (!Number.isSafeInteger(value) || (value as number) < 0) {
		throw new TypeError(`rhwp returned an invalid ${label}; expected a non-negative safe integer`);
	}
	return value as number;
}

function readPositiveCount(value: unknown, label: string): number {
	const count = readCount(value, label);
	if (count === 0) {
		throw new TypeError(`rhwp returned an invalid ${label}; expected a positive safe integer`);
	}
	return count;
}

function readSafeInteger(value: unknown, label: string): number {
	if (!Number.isSafeInteger(value)) {
		throw new TypeError(`rhwp returned an invalid ${label}; expected a safe integer`);
	}
	return value as number;
}

function readFiniteNumber(value: unknown, label: string): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		throw new TypeError(`rhwp returned an invalid ${label}; expected a finite number`);
	}
	return value;
}

function readPositiveFiniteNumber(value: unknown, label: string): number {
	const number = readFiniteNumber(value, label);
	if (number <= 0) {
		throw new TypeError(`rhwp returned an invalid ${label}; expected a positive finite number`);
	}
	return number;
}

function readText(value: unknown, label: string): string {
	if (typeof value !== "string") throw new TypeError(`rhwp returned invalid ${label}; expected a string`);
	return value;
}

function readJsonRecord(value: unknown, label: string): Record<string, unknown> {
	const text = readText(value, label);
	let parsed: unknown;
	try {
		parsed = JSON.parse(text) as unknown;
	} catch (cause) {
		throw new TypeError(`rhwp returned malformed JSON for ${label}`, { cause });
	}
	return readRecord(parsed, label);
}

function readRecord(value: unknown, label: string): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new TypeError(`rhwp returned invalid ${label}; expected an object`);
	}
	return value as Record<string, unknown>;
}

function readArray(value: unknown, label: string): readonly unknown[] {
	if (!Array.isArray(value)) throw new TypeError(`rhwp returned invalid ${label}; expected an array`);
	return value;
}
