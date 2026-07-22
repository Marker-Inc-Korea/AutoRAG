import { describe, expect, it, vi } from "vitest";
import { createRhwpExtractor, type RhwpDocumentApi, type RhwpRuntime } from "../../src/parser/rhwp-adapter.ts";

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
	it("initializes a runtime only once after initialization succeeds", async () => {
		const document = createFakeDocument();
		const runtime = createRuntime(document);
		const extractor = createRhwpExtractor(runtime);

		await extractor(new Uint8Array([1]));
		await extractor(new Uint8Array([2]));

		expect(runtime.initialize).toHaveBeenCalledTimes(1);
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

	it.each([
		["malformed layout JSON", "{not json"],
		["a table without secIdx", '{"controls":[{"type":"table","paraIdx":0,"controlIdx":0}]}'],
		[
			"nested table metadata",
			'{"controls":[{"type":"table","secIdx":0,"paraIdx":0,"controlIdx":0,"cellPath":[{"controlIndex":0,"cellIndex":0,"cellParaIndex":0}]}]}',
		],
	])("rejects %s instead of skipping it", async (_label, layout) => {
		const document = createFakeDocument({ getPageControlLayout: vi.fn(() => layout) });

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow();
	});

	it("rejects a negative table cell count", async () => {
		const document = createFakeDocument({
			getPageControlLayout: vi.fn(() => topLevelTableLayout),
			getTableDimensions: vi.fn(() => '{"rowCount":1,"colCount":1,"cellCount":-1}'),
		});

		await expect(createRhwpExtractor(createRuntime(document))(new Uint8Array([1]))).rejects.toThrow(/cellCount/);
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
			{},
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
