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
	const counters = { characters: 0, tables: 0, cells: 0, cellParagraphs: 0 };
	const sectionCount = readCount(document.getSectionCount(), "section count");
	checkBudget("maxSections", sectionCount, limits);
	for (let sectionIndex = 0; sectionIndex < sectionCount; sectionIndex += 1) {
		const paragraphCount = readCount(
			document.getParagraphCount(sectionIndex),
			`paragraph count for section ${sectionIndex}`,
		);
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
	return { paragraphs, tables: extractTables(document, limits, counters) };
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
			if (Object.hasOwn(control, "cellPath")) {
				throw new TypeError(
					`rhwp returned nested table metadata at page ${pageIndex}:${layoutIndex}; only top-level tables are supported`,
				);
			}

			const coordinates: TableCoordinates = {
				sectionIndex: readCount(control.secIdx, `table secIdx at page ${pageIndex}:${layoutIndex}`),
				parentParagraphIndex: readCount(control.paraIdx, `table paraIdx at page ${pageIndex}:${layoutIndex}`),
				controlIndex: readCount(control.controlIdx, `table controlIdx at page ${pageIndex}:${layoutIndex}`),
			};
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
	const rowCount = readCount(dimensions.rowCount, `rowCount for table ${key}`);
	const columnCount = readCount(dimensions.colCount, `colCount for table ${key}`);
	const cellCount = readCount(dimensions.cellCount, `cellCount for table ${key}`);
	counters.cells += cellCount;
	checkBudget("maxCells", counters.cells, limits);

	const cells: HwpTableCell[] = [];
	for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
		const cellInfo = readJsonRecord(
			document.getCellInfo(sectionIndex, parentParagraphIndex, controlIndex, cellIndex),
			`cell metadata at ${key}:${cellIndex}`,
		);
		const row = readCount(cellInfo.row, `row for cell ${key}:${cellIndex}`);
		const column = readCount(cellInfo.col, `col for cell ${key}:${cellIndex}`);
		const rowSpan = readCount(cellInfo.rowSpan, `rowSpan for cell ${key}:${cellIndex}`);
		const columnSpan = readCount(cellInfo.colSpan, `colSpan for cell ${key}:${cellIndex}`);
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
			row,
			column,
			rowSpan,
			columnSpan,
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
