import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { createDefaultParserRegistry } from "../../src/parser/defaults.ts";
import { ParseError } from "../../src/parser/errors.ts";
import { KordocParser } from "../../src/parser/kordoc.ts";
import {
	createDocxFixture,
	createMultiSheetXlsxFixture,
	createNestedTableHwpxFixture,
} from "../fixtures/document-formats.ts";

async function parseWith(virtualPath: string, bytes: Uint8Array) {
	const registry = createDefaultParserRegistry();
	const parser = registry.getForVirtualPath(virtualPath);
	expect(parser).toBeInstanceOf(KordocParser);
	const parsed = await parser?.parse({ virtualPath, bytes });
	if (!parsed) throw new Error(`no parser for ${virtualPath}`);
	return parsed;
}

describe("KordocParser routing", () => {
	it("owns every document extension kordoc supports", () => {
		const registry = createDefaultParserRegistry();
		for (const path of [
			"/docs/a.hwp",
			"/docs/a.hwpx",
			"/docs/a.hml",
			"/docs/a.hwpml",
			"/docs/a.pdf",
			"/docs/a.docx",
			"/docs/a.xlsx",
			"/docs/a.xls",
		]) {
			expect(registry.getForVirtualPath(path)?.name, path).toBe("kordoc");
		}
	});

	it("leaves formats kordoc cannot parse on their existing parsers", () => {
		const registry = createDefaultParserRegistry();
		expect(registry.getForVirtualPath("/docs/a.pptx")?.name).toBe("pptx");
		expect(registry.getForVirtualPath("/docs/a.eml")?.name).toBe("eml");
		expect(registry.getForVirtualPath("/docs/a.md")?.name).toBe("plain-text");
	});
});

describe("KordocParser structure fidelity", () => {
	it("parses a real HWP5 body into tables instead of flattened row lines", async () => {
		const bytes = await readFile(new URL("../fixtures/hwp5/minimal-body-table.hwp", import.meta.url));
		const parsed = await parseWith("/docs/minimal-body-table.hwp", bytes);

		expect(parsed.markdown).toContain("편집 탭 – 표");
		expect(parsed.markdown).toContain("제목");
		expect(parsed.markdown).toContain("담당자");
		expect(parsed.markdown).toContain("세부 내용");
		// kordoc renders real table markup; the retired rhwp path emitted "Row 1: a | b".
		expect(parsed.markdown).toMatch(/<table>|\| --- \|/);
		expect(parsed.markdown).not.toContain("Row 1:");
		expect(parsed.metadata).toMatchObject({ parser: "kordoc", format: "hwp" });
	});

	it("keeps a table nested inside a table cell and never leaks header.xml", async () => {
		const parsed = await parseWith("/docs/nested.hwpx", await createNestedTableHwpxFixture());

		expect(parsed.markdown).toContain("outerCellMarker");
		expect(parsed.markdown).toContain("innerA1Marker");
		expect(parsed.markdown).toContain("innerB1Marker");
		expect(parsed.markdown).toContain("innerA2Marker");
		expect(parsed.markdown).toContain("innerB2Marker");
		// The header part must stay out of the indexed body.
		expect(parsed.markdown).not.toContain("hwpxHeaderLeakToken");
		// Nesting is preserved: an inner table inside the outer table's markup.
		expect((parsed.markdown.match(/<table>/g) ?? []).length).toBe(2);
		// Body paragraphs keep document order around the table.
		expect(parsed.markdown.indexOf("bodyBeforeMarker")).toBeLessThan(parsed.markdown.indexOf("outerCellMarker"));
		expect(parsed.markdown.indexOf("innerB2Marker")).toBeLessThan(parsed.markdown.indexOf("bodyAfterMarker"));
		expect(parsed.metadata).toMatchObject({ parser: "kordoc", format: "hwpx" });
	});

	it("keeps XLSX sheet boundaries and rows instead of flattening cell values", async () => {
		const parsed = await parseWith("/docs/book.xlsx", createMultiSheetXlsxFixture());

		expect(parsed.markdown).toContain("## RevenueSheet");
		expect(parsed.markdown).toContain("## RiskSheet");
		expect(parsed.markdown).toContain("| Quarter | Revenue | Owner |");
		expect(parsed.markdown).toContain("| Q3 | 742000 | finance team |");
		expect(parsed.markdown).toContain("| supply chain | open |");
		expect(parsed.metadata).toMatchObject({ parser: "kordoc", format: "xlsx" });
	});

	it("parses DOCX text", async () => {
		const parsed = await parseWith("/docs/note.docx", await createDocxFixture("docx body marker"));
		expect(parsed.markdown).toContain("docx body marker");
		expect(parsed.metadata).toMatchObject({ parser: "kordoc", format: "docx" });
	});
});

describe("KordocParser failure transparency", () => {
	it("wraps a kordoc failure in ParseError carrying the verbatim code and message", async () => {
		const registry = createDefaultParserRegistry();
		const parser = registry.getForVirtualPath("/docs/broken.hwp");

		const error = await parser
			?.parse({ virtualPath: "/docs/broken.hwp", bytes: Buffer.from("not a document at all") })
			.then(
				() => undefined,
				(caught: unknown) => caught,
			);

		expect(error).toBeInstanceOf(ParseError);
		expect((error as ParseError).message).toContain("UNSUPPORTED_FORMAT");
		expect((error as ParseError).message).toContain("지원하지 않는 파일 형식입니다.");
		expect((error as ParseError).parserName).toBe("kordoc");
	});

	it("propagates an injected failure code and text without classifying it", async () => {
		const parser = new KordocParser({
			parse: async () => ({
				success: false as const,
				fileType: "hwp" as const,
				error: "문서가 암호로 보호되어 있습니다.",
				code: "ENCRYPTED" as const,
			}),
		});

		await expect(parser.parse({ virtualPath: "/docs/locked.hwp", bytes: new Uint8Array([1]) })).rejects.toThrow(
			/ENCRYPTED: 문서가 암호로 보호되어 있습니다\./,
		);
	});

	it("maps kordoc warnings to parser-warning diagnostics verbatim", async () => {
		const parser = new KordocParser({
			parse: async () => ({
				success: true as const,
				fileType: "pdf" as const,
				markdown: "body",
				blocks: [],
				warnings: [
					{ code: "NEEDS_OCR", message: "텍스트 레이어가 없어 OCR이 필요합니다." },
					{ code: "BROKEN_ZIP_RECOVERY", message: "손상된 zip을 복구했습니다." },
				],
			}),
		});

		const parsed = await parser.parse({ virtualPath: "/docs/scan.pdf", bytes: new Uint8Array([1]) });

		expect(parsed.diagnostics).toEqual([
			{
				code: "parser-warning",
				severity: "warning",
				message: "NEEDS_OCR: 텍스트 레이어가 없어 OCR이 필요합니다.",
			},
			{
				code: "parser-warning",
				severity: "warning",
				message: "BROKEN_ZIP_RECOVERY: 손상된 zip을 복구했습니다.",
			},
		]);
	});
});
