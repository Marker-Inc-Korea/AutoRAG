import { describe, expect, it } from "vitest";
import { type HwpExtractedDocument, renderHwpMarkdown } from "../../src/parser/hwp-markdown.ts";

describe("renderHwpMarkdown", () => {
	it("preserves paragraphs and inserts a regular table after its parent", () => {
		const document: HwpExtractedDocument = {
			paragraphs: [
				{ sectionIndex: 0, paragraphIndex: 0, text: "본문 시작\u0000" },
				{ sectionIndex: 0, paragraphIndex: 1, text: "본문 끝\uFFFC" },
			],
			tables: [
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 0,
					rowCount: 2,
					columnCount: 2,
					cells: [
						{ row: 0, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: ["A|1"] },
						{ row: 0, column: 1, rowSpan: 1, columnSpan: 1, paragraphs: ["B"] },
						{ row: 1, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: ["C", "C2"] },
						{ row: 1, column: 1, rowSpan: 1, columnSpan: 1, paragraphs: ["D"] },
					],
				},
			],
		};

		expect(renderHwpMarkdown(document)).toBe(
			[
				"본문 시작",
				"",
				"| Column 1 | Column 2 |",
				"| --- | --- |",
				"| A\\|1 | B |",
				"| C<br>C2 | D |",
				"",
				"본문 끝",
			].join("\n"),
		);
	});

	it("uses readable row labels for tables with merged cells", () => {
		const document: HwpExtractedDocument = {
			paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "Merged table" }],
			tables: [
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 0,
					rowCount: 2,
					columnCount: 2,
					cells: [
						{ row: 0, column: 0, rowSpan: 2, columnSpan: 1, paragraphs: ["merged"] },
						{ row: 0, column: 1, rowSpan: 1, columnSpan: 1, paragraphs: ["right"] },
						{ row: 1, column: 1, rowSpan: 1, columnSpan: 1, paragraphs: ["lower-right"] },
					],
				},
			],
		};

		expect(renderHwpMarkdown(document)).toBe(
			["Merged table", "", "[Table]", "Row 1: merged | right", "Row 2: lower-right"].join("\n"),
		);
	});

	it("skips supplied coordinates covered by an earlier merged-cell anchor", () => {
		const document: HwpExtractedDocument = {
			paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "Merged table" }],
			tables: [
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 0,
					rowCount: 2,
					columnCount: 2,
					cells: [
						{ row: 0, column: 0, rowSpan: 2, columnSpan: 1, paragraphs: ["merged"] },
						{ row: 0, column: 1, rowSpan: 1, columnSpan: 1, paragraphs: ["right"] },
						{ row: 1, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: ["merged"] },
						{ row: 1, column: 1, rowSpan: 1, columnSpan: 1, paragraphs: ["lower-right"] },
					],
				},
			],
		};

		expect(renderHwpMarkdown(document)).toBe(
			["Merged table", "", "[Table]", "Row 1: merged | right", "Row 2: lower-right"].join("\n"),
		);
	});

	it("renders table-cell backslashes, pipes, and line endings safely", () => {
		const document: HwpExtractedDocument = {
			paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "Table" }],
			tables: [
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 0,
					rowCount: 1,
					columnCount: 1,
					cells: [{ row: 0, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: ["a\\b|c\r\nd\ne"] }],
				},
			],
		};

		expect(renderHwpMarkdown(document)).toBe(
			["Table", "", "| Column 1 |", "| --- |", "| a\\\\b\\|c<br>d<br>e |"].join("\n"),
		);
	});

	it("preserves leading and trailing ordinary whitespace in retained text", () => {
		const document: HwpExtractedDocument = {
			paragraphs: [
				{ sectionIndex: 0, paragraphIndex: 0, text: " \t " },
				{ sectionIndex: 0, paragraphIndex: 1, text: "  retained  \u0000\uFFFC" },
			],
			tables: [],
		};

		expect(renderHwpMarkdown(document)).toBe("  retained  ");
	});

	it("omits all-empty tables", () => {
		const document: HwpExtractedDocument = {
			paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "Only paragraph" }],
			tables: [
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 0,
					rowCount: 1,
					columnCount: 1,
					cells: [{ row: 0, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: ["\u0000"] }],
				},
			],
		};

		expect(renderHwpMarkdown(document)).toBe("Only paragraph");
	});

	it("orders multiple tables at one parent by control index", () => {
		const document: HwpExtractedDocument = {
			paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "Tables" }],
			tables: [
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 2,
					rowCount: 1,
					columnCount: 1,
					cells: [{ row: 0, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: ["second"] }],
				},
				{
					sectionIndex: 0,
					parentParagraphIndex: 0,
					controlIndex: 1,
					rowCount: 1,
					columnCount: 1,
					cells: [{ row: 0, column: 0, rowSpan: 1, columnSpan: 1, paragraphs: ["first"] }],
				},
			],
		};

		expect(renderHwpMarkdown(document)).toBe(
			["Tables", "", "| Column 1 |", "| --- |", "| first |", "", "| Column 1 |", "| --- |", "| second |"].join("\n"),
		);
	});
});
