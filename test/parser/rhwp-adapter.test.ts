import { describe, expect, it, vi } from "vitest";

const defaultRuntimeMocks = vi.hoisted(() => ({
	initialize: vi.fn<() => Promise<void>>(),
	open: vi.fn(),
}));

vi.mock("node:fs/promises", async (importOriginal) => ({
	...(await importOriginal<typeof import("node:fs/promises")>()),
	readFile: vi.fn(async () => new Uint8Array([0])),
}));

vi.mock("@rhwp/core", () => ({
	default: defaultRuntimeMocks.initialize,
	HwpDocument: function MockHwpDocument(bytes: Uint8Array) {
		return defaultRuntimeMocks.open(bytes);
	},
}));

import {
	createRhwpExtractor,
	extractHwpWithRhwp,
	type RhwpDocumentApi,
	type RhwpRuntime,
} from "../../src/parser/rhwp-adapter.ts";

function createFakeDocument(overrides: Partial<RhwpDocumentApi> = {}): RhwpDocumentApi {
	return {
		free: vi.fn(),
		getCellInfo: vi.fn(() => '{"row":0,"col":0,"rowSpan":1,"colSpan":1}'),
		getCellParagraphCount: vi.fn(() => 0),
		getCellParagraphLength: vi.fn(() => 0),
		getPageControlLayout: vi.fn(() => '{"controls":[]}'),
		getParagraphCount: vi.fn(() => 1),
		getParagraphLength: vi.fn(() => 4),
		getSectionCount: vi.fn(() => 1),
		getTableDimensions: vi.fn(() => '{"rowCount":0,"colCount":0,"cellCount":0}'),
		getTextInCell: vi.fn(() => ""),
		getTextRange: vi.fn(() => "body"),
		pageCount: vi.fn(() => 1),
		...overrides,
	};
}

function createRuntime(document: RhwpDocumentApi): RhwpRuntime {
	return {
		initialize: vi.fn(async () => undefined),
		open: vi.fn(() => document),
	};
}

describe("createRhwpExtractor cleanup", () => {
	it("frees an opened document after successful extraction", async () => {
		const document = createFakeDocument();

		await createRhwpExtractor(createRuntime(document))(new Uint8Array([1]));

		expect(document.free).toHaveBeenCalledTimes(1);
	});

	it("frees an opened document when extraction throws", async () => {
		const document = createFakeDocument({
			getTextRange: vi.fn(() => {
				throw new Error("text failure");
			}),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow("text failure");

		expect(document.free).toHaveBeenCalledTimes(1);
	});
});

describe("createRhwpExtractor initialization", () => {
	it("shares fulfilled default initialization across convenience calls and retries rejection", async () => {
		const documents = [createFakeDocument(), createFakeDocument()];
		defaultRuntimeMocks.initialize
			.mockRejectedValueOnce(new Error("default initialization failure"))
			.mockResolvedValue(undefined);
		defaultRuntimeMocks.open.mockImplementation(() => {
			const document = documents.shift();
			if (document === undefined) throw new Error("unexpected default runtime open");
			return document;
		});

		await expect(extractHwpWithRhwp(new Uint8Array([1]))).rejects.toThrow("default initialization failure");
		await expect(extractHwpWithRhwp(new Uint8Array([2]))).resolves.toMatchObject({
			paragraphs: [{ text: "body" }],
		});
		await expect(extractHwpWithRhwp(new Uint8Array([3]), { maxCharacters: 3 })).rejects.toMatchObject({
			code: "HWP_EXTRACTION_BUDGET_EXCEEDED",
			limit: "maxCharacters",
		});

		expect(defaultRuntimeMocks.initialize).toHaveBeenCalledTimes(2);
		expect(defaultRuntimeMocks.open).toHaveBeenCalledTimes(2);
		expect(documents).toHaveLength(0);
	});

	it("initializes a runtime only once after initialization succeeds", async () => {
		const document = createFakeDocument();
		const runtime = createRuntime(document);
		const extractor = createRhwpExtractor(runtime);

		await extractor(new Uint8Array([1]));
		await extractor(new Uint8Array([2]));

		expect(runtime.initialize).toHaveBeenCalledTimes(1);
	});

	it("keeps initialization caches isolated between injected-runtime extractors", async () => {
		const runtime = createRuntime(createFakeDocument());

		await createRhwpExtractor(runtime)(new Uint8Array([1]));
		await createRhwpExtractor(runtime)(new Uint8Array([2]));

		expect(runtime.initialize).toHaveBeenCalledTimes(2);
	});

	it("clears a rejected initialization so the next extraction can retry", async () => {
		const document = createFakeDocument();
		let attempts = 0;
		const runtime: RhwpRuntime = {
			initialize: vi.fn(async () => {
				attempts += 1;
				if (attempts === 1) throw new Error("initialization failure");
			}),
			open: vi.fn(() => document),
		};
		const extractor = createRhwpExtractor(runtime);

		await expect(extractor(new Uint8Array([1]))).rejects.toThrow("initialization failure");
		await expect(extractor(new Uint8Array([2]))).resolves.toMatchObject({ paragraphs: [{ text: "body" }] });

		expect(runtime.initialize).toHaveBeenCalledTimes(2);
		expect(runtime.open).toHaveBeenCalledTimes(1);
	});
});

describe("createRhwpExtractor body traversal and limits", () => {
	it("extracts body paragraphs from every section in traversal order", async () => {
		const paragraphs = [["first", "second"], ["third"]];
		const document = createFakeDocument({
			getParagraphCount: vi.fn((sectionIndex) => paragraphs[sectionIndex]?.length ?? 0),
			getParagraphLength: vi.fn(
				(sectionIndex, paragraphIndex) => [...(paragraphs[sectionIndex]?.[paragraphIndex] ?? "")].length,
			),
			getSectionCount: vi.fn(() => paragraphs.length),
			getTextRange: vi.fn((sectionIndex, paragraphIndex) => paragraphs[sectionIndex]?.[paragraphIndex] ?? ""),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).resolves.toEqual({
			paragraphs: [
				{ sectionIndex: 0, paragraphIndex: 0, text: "first" },
				{ sectionIndex: 0, paragraphIndex: 1, text: "second" },
				{ sectionIndex: 1, paragraphIndex: 0, text: "third" },
			],
			tables: [],
		});
	});

	it.each([0, 1.5, Number.POSITIVE_INFINITY])(
		"rejects invalid positive finite safe-integer overrides (%s)",
		(maxSections) => {
			expect(() => createRhwpExtractor(createRuntime(createFakeDocument()), { maxSections })).toThrow(/maxSections/);
		},
	);

	it("rejects an announced body paragraph count beyond its budget with a category", async () => {
		const document = createFakeDocument({ getParagraphCount: vi.fn(() => 2) });

		const result = createRhwpExtractor(createRuntime(document), { maxBodyParagraphs: 1 })(new Uint8Array([1]));

		await expect(result).rejects.toMatchObject({
			code: "HWP_EXTRACTION_BUDGET_EXCEEDED",
			limit: "maxBodyParagraphs",
		});
		expect(document.getParagraphLength).not.toHaveBeenCalled();
	});

	it("counts extracted characters as Unicode code points", async () => {
		const document = createFakeDocument({
			getParagraphLength: vi.fn(() => 2),
			getTextRange: vi.fn(() => "😀"),
		});

		await expect(
			createRhwpExtractor(createRuntime(document), { maxCharacters: 1 })(new Uint8Array([1])),
		).resolves.toMatchObject({ paragraphs: [{ text: "😀" }] });
	});
});

const topLevelTableLayout = JSON.stringify({
	controls: [{ type: "table", secIdx: 0, paraIdx: 0, controlIdx: 3 }],
});

describe("createRhwpExtractor table traversal and validation", () => {
	it("de-duplicates a top-level table repeated in multiple page layouts", async () => {
		const document = createFakeDocument({
			getCellParagraphCount: vi.fn(() => 2),
			getCellParagraphLength: vi.fn((_section, _paragraph, _control, _cell, cellParagraph) =>
				cellParagraph === 0 ? 5 : 6,
			),
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":1}'),
			getTextInCell: vi.fn((_section, _paragraph, _control, _cell, cellParagraph) =>
				cellParagraph === 0 ? "first" : "second",
			),
			pageCount: vi.fn(() => 2),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).resolves.toEqual({
			paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "body" }],
			tables: [
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 3,
					rowCount: 1,
					columnCount: 1,
					cells: [{ row: 0, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: ["first", "second"] }],
				},
			],
		});
		expect(document.getTableDimensions).toHaveBeenCalledTimes(1);
	});

	it("skips a deferred nested table while preserving body and top-level table extraction", async () => {
		const layout = JSON.stringify({
			controls: [
				{ type: "table", secIdx: 0, paraIdx: 0, controlIdx: 3 },
				{
					type: "table",
					x: 120,
					y: 240,
					w: 300,
					h: 80,
					rowCount: 1,
					colCount: 1,
					parentParaIdx: 0,
					cellPath: [{ controlIndex: 3, cellIndex: 0, cellParaIndex: 0 }],
					plane: 2,
					zOrder: 0,
					stableIndex: 12,
					cells: [{ row: 0, col: 0, rowSpan: 1, colSpan: 1, cellIdx: 0 }],
				},
			],
		});
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => layout),
			getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":1}'),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).resolves.toEqual({
			paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "body" }],
			tables: [
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 3,
					rowCount: 1,
					columnCount: 1,
					cells: [{ row: 0, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: [] }],
				},
			],
		});
		expect(document.getTableDimensions).toHaveBeenCalledTimes(1);
	});

	it("skips the pinned runtime's deferred nested-table layout context", async () => {
		const layout = JSON.stringify({
			controls: [
				{ type: "table", secIdx: 0, paraIdx: 0, controlIdx: 3 },
				{
					type: "table",
					x: 120,
					y: 240,
					w: 300,
					h: 80,
					rowCount: 2,
					colCount: 2,
					plane: 2,
					zOrder: 0,
					stableIndex: 12,
					cells: [
						{ x: 120, y: 240, w: 300, h: 40, row: 0, col: 0, rowSpan: 1, colSpan: 2, cellIdx: 0 },
						{ x: 120, y: 280, w: 150, h: 40, row: 1, col: 0, rowSpan: 1, colSpan: 1, cellIdx: 1 },
						{ x: 270, y: 280, w: 150, h: 40, row: 1, col: 1, rowSpan: 1, colSpan: 1, cellIdx: 2 },
					],
				},
			],
		});
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => layout),
			getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":1}'),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).resolves.toMatchObject({
			paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "body" }],
			tables: [{ sectionIndex: 0, parentParagraphIndex: 0, controlIndex: 3 }],
		});
		expect(document.getTableDimensions).toHaveBeenCalledTimes(1);
	});

	it("charges a deferred nested table's logical grid before expanding cell spans", async () => {
		const layout = JSON.stringify({
			controls: [
				{
					type: "table",
					x: 0,
					y: 0,
					w: 10,
					h: 10,
					rowCount: 10,
					colCount: 1,
					plane: 2,
					zOrder: 0,
					stableIndex: 0,
					cells: [{ x: 0, y: 0, w: 10, h: 10, row: 0, col: 0, rowSpan: 10, colSpan: 1, cellIdx: 0 }],
				},
			],
		});
		const document = createFakeDocument({ getPageControlLayout: vi.fn(() => layout) });

		await expect(
			createRhwpExtractor(createRuntime(document), { maxCells: 1 })(new Uint8Array([1])),
		).rejects.toMatchObject({
			code: "HWP_EXTRACTION_BUDGET_EXCEEDED",
			limit: "maxCells",
			actual: 10,
			maximum: 1,
		});
	});

	it("rejects an overflow-unsafe deferred nested logical grid before span traversal", async () => {
		const layout = JSON.stringify({
			controls: [
				{
					type: "table",
					x: 0,
					y: 0,
					w: 10,
					h: 10,
					rowCount: Number.MAX_SAFE_INTEGER,
					colCount: 2,
					plane: 2,
					zOrder: 0,
					stableIndex: 0,
					cells: [{ x: 0, y: 0, w: 10, h: 10, row: 0, col: 0, rowSpan: 1, colSpan: 1, cellIdx: 0 }],
				},
			],
		});
		const document = createFakeDocument({ getPageControlLayout: vi.fn(() => layout) });

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toMatchObject({
			code: "HWP_EXTRACTION_BUDGET_EXCEEDED",
			limit: "maxCells",
		});
	});

	it("rejects malformed layout JSON", async () => {
		const document = createFakeDocument({ getPageControlLayout: vi.fn(() => "{not json") });

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow();
	});

	it.each([
		["a table with no coordinates or nested context", { type: "table" }],
		["a partial coordinate set", { type: "table", paraIdx: 0, controlIdx: 0 }],
		[
			"top-level coordinates mixed with nested context",
			{
				type: "table",
				secIdx: 0,
				paraIdx: 0,
				controlIdx: 0,
				parentParaIdx: 0,
				cellPath: [{ controlIndex: 0, cellIndex: 0, cellParaIndex: 0 }],
			},
		],
		["an empty nested path", { type: "table", parentParaIdx: 0, cellPath: [] }],
		[
			"an incomplete nested path entry",
			{ type: "table", parentParaIdx: 0, cellPath: [{ controlIndex: 0, cellIndex: 0 }] },
		],
		[
			"nested context without parentParaIdx",
			{ type: "table", cellPath: [{ controlIndex: 0, cellIndex: 0, cellParaIndex: 0 }] },
		],
		[
			"an empty pinned-runtime cell context",
			{
				type: "table",
				x: 0,
				y: 0,
				w: 1,
				h: 1,
				rowCount: 1,
				colCount: 1,
				plane: 2,
				zOrder: 0,
				stableIndex: 0,
				cells: [],
			},
		],
		[
			"a malformed pinned-runtime cell span",
			{
				type: "table",
				x: 0,
				y: 0,
				w: 1,
				h: 1,
				rowCount: 1,
				colCount: 1,
				plane: 2,
				zOrder: 0,
				stableIndex: 0,
				cells: [{ x: 0, y: 0, w: 1, h: 1, row: 0, col: 0, rowSpan: 0, colSpan: 1, cellIdx: 0 }],
			},
		],
	])("rejects %s instead of broadly skipping it", async (_label, control) => {
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => JSON.stringify({ controls: [control] })),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow();
	});

	it("rejects a negative table cell count", async () => {
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":-1}'),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(/cellCount/);
	});

	it("rejects JSON numeric overflow in table dimensions before cell traversal", async () => {
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => '{"rowCount":1e309,"colCount":1,"cellCount":1}'),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(/rowCount/);
		expect(document.getCellInfo).not.toHaveBeenCalled();
	});

	it.each([
		["zero rowCount", { rowCount: 0, colCount: 1, cellCount: 0 }, /rowCount/],
		["zero colCount", { rowCount: 1, colCount: 0, cellCount: 0 }, /colCount/],
		["zero cellCount for a nonempty grid", { rowCount: 1, colCount: 1, cellCount: 0 }, /cellCount/],
		["more cells than logical coordinates", { rowCount: 1, colCount: 1, cellCount: 2 }, /cellCount/],
	] as const)("rejects inconsistent table geometry with %s", async (_label, dimensions, expected) => {
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => JSON.stringify(dimensions)),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(expected);
	});

	it.each([
		["section", { type: "table", secIdx: 1, paraIdx: 0, controlIdx: 0 }],
		["parent paragraph", { type: "table", secIdx: 0, paraIdx: 1, controlIdx: 0 }],
	] as const)("rejects an out-of-range table %s coordinate", async (label, control) => {
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => JSON.stringify({ controls: [control] })),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(label);
		expect(document.getTableDimensions).not.toHaveBeenCalled();
	});

	it("charges the logical grid against maxCells before cell iteration", async () => {
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => '{"rowCount":10,"colCount":1,"cellCount":1}'),
		});

		await expect(
			createRhwpExtractor(createRuntime(document), { maxCells: 1 })(new Uint8Array([1])),
		).rejects.toMatchObject({
			code: "HWP_EXTRACTION_BUDGET_EXCEEDED",
			limit: "maxCells",
			actual: 10,
			maximum: 1,
		});
		expect(document.getCellInfo).not.toHaveBeenCalled();
	});

	it("rejects an overflow-unsafe logical grid before cell iteration", async () => {
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() =>
				JSON.stringify({ rowCount: Number.MAX_SAFE_INTEGER, colCount: 2, cellCount: 1 }),
			),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toMatchObject({
			code: "HWP_EXTRACTION_BUDGET_EXCEEDED",
			limit: "maxCells",
		});
		expect(document.getCellInfo).not.toHaveBeenCalled();
	});

	it("charges logical grids cumulatively across tables", async () => {
		const layout = JSON.stringify({
			controls: [
				{ type: "table", secIdx: 0, paraIdx: 0, controlIdx: 3 },
				{ type: "table", secIdx: 0, paraIdx: 0, controlIdx: 4 },
			],
		});
		const document = createFakeDocument({
			getCellInfo: vi.fn(() => '{"row":0,"col":0,"rowSpan":1,"colSpan":2}'),
			getPageControlLayout: vi.fn(() => layout),
			getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":2,"cellCount":1}'),
		});

		await expect(
			createRhwpExtractor(createRuntime(document), { maxCells: 3 })(new Uint8Array([1])),
		).rejects.toMatchObject({
			code: "HWP_EXTRACTION_BUDGET_EXCEEDED",
			limit: "maxCells",
			actual: 4,
			maximum: 3,
		});
		expect(document.getCellInfo).toHaveBeenCalledTimes(1);
	});

	it.each([
		["row outside the table", { row: 2, col: 0, rowSpan: 1, colSpan: 1 }, /row.*range/],
		["column outside the table", { row: 0, col: 2, rowSpan: 1, colSpan: 1 }, /col.*range/],
		["zero rowSpan", { row: 0, col: 0, rowSpan: 0, colSpan: 1 }, /rowSpan/],
		["zero colSpan", { row: 0, col: 0, rowSpan: 1, colSpan: 0 }, /colSpan/],
		["rowSpan past the table", { row: 1, col: 0, rowSpan: 2, colSpan: 1 }, /rowSpan.*range/],
		["colSpan past the table", { row: 0, col: 1, rowSpan: 1, colSpan: 2 }, /colSpan.*range/],
	] as const)("rejects cell geometry with %s before paragraph traversal", async (_label, cellInfo, expected) => {
		const document = createFakeDocument({
			getCellInfo: vi.fn(() => JSON.stringify(cellInfo)),
			getCellParagraphCount: vi.fn(() => {
				throw new Error("must not traverse invalid cell geometry");
			}),
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => '{"rowCount":2,"colCount":2,"cellCount":1}'),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(expected);
		expect(document.getCellParagraphCount).not.toHaveBeenCalled();
	});

	it.each([
		[
			"duplicate anchors",
			{ rowCount: 1, colCount: 2, cellCount: 2 },
			[
				{ row: 0, col: 0, rowSpan: 1, colSpan: 1 },
				{ row: 0, col: 0, rowSpan: 1, colSpan: 1 },
			],
			/duplicate cell anchor/,
		],
		[
			"overlapping spans",
			{ rowCount: 1, colCount: 3, cellCount: 2 },
			[
				{ row: 0, col: 0, rowSpan: 1, colSpan: 2 },
				{ row: 0, col: 1, rowSpan: 1, colSpan: 2 },
			],
			/overlapping cell spans/,
		],
		[
			"an uncovered logical coordinate",
			{ rowCount: 2, colCount: 2, cellCount: 2 },
			[
				{ row: 0, col: 0, rowSpan: 1, colSpan: 1 },
				{ row: 1, col: 1, rowSpan: 1, colSpan: 1 },
			],
			/logical grid/,
		],
	] as const)("rejects %s before paragraph traversal", async (_label, dimensions, cellInfos, expected) => {
		const document = createFakeDocument({
			getCellInfo: vi.fn((_section, _paragraph, _control, cellIndex) => JSON.stringify(cellInfos[cellIndex])),
			getCellParagraphCount: vi.fn(() => {
				throw new Error("must validate the complete grid before paragraph traversal");
			}),
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => JSON.stringify(dimensions)),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(expected);
		expect(document.getCellParagraphCount).not.toHaveBeenCalled();
	});

	it("rejects a non-finite numeric count", async () => {
		const document = createFakeDocument({ pageCount: vi.fn(() => Number.POSITIVE_INFINITY) });

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(/page count/);
	});

	it("rejects cell metadata with missing coordinates", async () => {
		const document = createFakeDocument({
			getCellInfo: vi.fn(() => '{"row":0,"rowSpan":1,"colSpan":1}'),
			getCellParagraphCount: vi.fn(() => {
				throw new Error("must not traverse invalid cell metadata");
			}),
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":1}'),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(/col/);
		expect(document.getCellParagraphCount).not.toHaveBeenCalled();
	});

	it.each([
		[
			"maxTables",
			{ maxTables: 1 },
			JSON.stringify({
				controls: [
					{ type: "table", secIdx: 0, paraIdx: 0, controlIdx: 0 },
					{ type: "table", secIdx: 0, paraIdx: 0, controlIdx: 1 },
				],
			}),
			{ getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":1}') },
		],
		[
			"maxCells",
			{ maxCells: 1 },
			topLevelTableLayout,
			{ getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":2,"cellCount":2}') },
		],
		[
			"maxCellParagraphs",
			{ maxCellParagraphs: 1 },
			topLevelTableLayout,
			{
				getCellParagraphCount: vi.fn(() => 2),
				getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":1}'),
			},
		],
	] as const)("enforces the %s announced-size budget before traversal", async (limit, limits, layout, overrides) => {
		const document = createFakeDocument({ getPageControlLayout: vi.fn(() => layout), ...overrides });

		await expect(createRhwpExtractor(createRuntime(document), limits)(new Uint8Array([1]))).rejects.toMatchObject({
			code: "HWP_EXTRACTION_BUDGET_EXCEEDED",
			limit,
		});
	});

	it("enforces the shared character budget after every extracted cell string", async () => {
		const document = createFakeDocument({
			getCellParagraphCount: vi.fn(() => 1),
			getCellParagraphLength: vi.fn(() => 2),
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getParagraphLength: vi.fn(() => 1),
			getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":1}'),
			getTextInCell: vi.fn(() => "😀😀"),
			getTextRange: vi.fn(() => "a"),
		});

		await expect(
			createRhwpExtractor(createRuntime(document), { maxCharacters: 2 })(new Uint8Array([1])),
		).rejects.toMatchObject({ limit: "maxCharacters", actual: 3 });
	});
});
