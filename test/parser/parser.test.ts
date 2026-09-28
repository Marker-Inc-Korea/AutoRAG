import { describe, expect, it } from "vitest";
import {
	createDefaultParserRegistry,
	ParseError,
	Parser,
	ParserRegistry,
	PlainTextParser,
} from "../../src/parser/index.ts";
import {
	createDocxFixture,
	createEmlFixture,
	createEucKrEmlFixture,
	createHwpxFixture,
	createPptxFixture,
	createRichXlsFixture,
	createXlsFixture,
	createXlsxFixture,
} from "../fixtures/document-formats.ts";
import { createMinimalPdfBuffer } from "../fixtures/minimal-pdf.ts";

const pdfMarker = "AutoRAG PDF marker refund policy alpha";

class UppercaseParser extends Parser {
	readonly name = "uppercase";
	readonly extensions = [".up"];

	async parse(input: { readonly bytes: Uint8Array }): Promise<{ readonly markdown: string }> {
		return { markdown: Buffer.from(input.bytes).toString("utf8").toUpperCase() };
	}
}

describe("ParserRegistry", () => {
	it("routes by lowercased extension through Parser subclasses", async () => {
		const registry = new ParserRegistry([new UppercaseParser()]);
		const parser = registry.getForVirtualPath("/docs/NOTE.UP");

		expect(parser).toBeInstanceOf(UppercaseParser);
		await expect(parser?.parse({ virtualPath: "/docs/NOTE.UP", bytes: Buffer.from("alpha") })).resolves.toEqual({
			markdown: "ALPHA",
		});
	});

	it("rejects duplicate extension ownership", () => {
		const first = new PlainTextParser();
		const second = new PlainTextParser();

		expect(() => new ParserRegistry([first, second])).toThrow('Parser extension ".txt" is already registered');
	});

	it("default registry supports text, markdown, and PDF but skips unsupported binary files", async () => {
		// Given: a default parser registry and a minimal PDF with searchable marker text.
		const registry = createDefaultParserRegistry();

		// When: parser lookup routes common document extensions.
		const pdfParser = registry.getForVirtualPath("/docs/report.pdf");

		// Then: PDF files are parsed through the default registry without importing a concrete parser class.
		expect(registry.getForVirtualPath("/docs/a.txt")).toBeInstanceOf(PlainTextParser);
		expect(registry.getForVirtualPath("/docs/a.md")).toBeInstanceOf(PlainTextParser);
		expect(pdfParser).toBeDefined();
		await expect(
			pdfParser?.parse({ virtualPath: "/docs/report.pdf", bytes: createMinimalPdfBuffer(pdfMarker) }),
		).resolves.toMatchObject({
			markdown: expect.stringContaining(pdfMarker),
		});
		expect(registry.getForVirtualPath("/docs/a.bin")).toBeUndefined();
	});

	it("default PDF parser rejects malformed PDFs with a typed ParseError", async () => {
		// Given: a default registry PDF parser and bytes that are not a valid PDF.
		const registry = createDefaultParserRegistry();
		const pdfParser = registry.getForVirtualPath("/docs/broken.pdf");

		// When/Then: parser failures are typed at the AutoRAG parser boundary.
		await expect(
			pdfParser?.parse({ virtualPath: "/docs/broken.pdf", bytes: Buffer.from("not a pdf") }),
		).rejects.toBeInstanceOf(ParseError);
	});

	it("default registry parses office, HWPX, and email formats into searchable markdown", async () => {
		// Given: representative document bytes for the newly supported document formats.
		const registry = createDefaultParserRegistry();
		const cases = [
			{
				path: "/docs/contract.docx",
				marker: "DOCX refund policy marker",
				bytes: await createDocxFixture("DOCX refund policy marker"),
			},
			{
				path: "/docs/deck.pptx",
				marker: "PPTX roadmap marker",
				bytes: await createPptxFixture("PPTX roadmap marker"),
			},
			{
				path: "/docs/budget.xlsx",
				marker: "XLSX budget marker",
				bytes: await createXlsxFixture("XLSX budget marker"),
			},
			{
				path: "/docs/form.hwpx",
				marker: "HWPX Korean corpus marker",
				bytes: await createHwpxFixture("HWPX Korean corpus marker"),
			},
			{ path: "/docs/thread.eml", marker: "EML decision marker", bytes: createEmlFixture("EML decision marker") },
			{ path: "/docs/korean.eml", marker: "한글 메일 marker", bytes: createEucKrEmlFixture("한글 메일 marker") },
		] as const;

		for (const testCase of cases) {
			// When: each extension is routed through the default registry.
			const parser = registry.getForVirtualPath(testCase.path);

			// Then: marker text becomes searchable markdown without callers importing parser classes.
			expect(parser, testCase.path).toBeDefined();
			const parsed = await parser?.parse({ virtualPath: testCase.path, bytes: testCase.bytes });
			expect(parsed?.markdown).toContain(testCase.marker);
			expect(parsed?.metadata).toMatchObject({ parser: parser?.name });
		}
	});

	it("decodes legacy Korean text and normalizes parsed markdown to NFC", async () => {
		// Given: CP949 bytes and decomposed Hangul text entering the parser boundary.
		const registry = createDefaultParserRegistry();
		const textParser = registry.getForVirtualPath("/docs/korean.txt");
		const decomposed = "한글";
		const cp949Bytes = Buffer.from([0xc7, 0xd1, 0xb1, 0xdb]);

		// When/Then: text bytes decode correctly and every parsed text output is NFC-normalized.
		await expect(textParser?.parse({ virtualPath: "/docs/korean.txt", bytes: cp949Bytes })).resolves.toMatchObject({
			markdown: "한글",
		});
		await expect(
			textParser?.parse({ virtualPath: "/docs/decomposed.txt", bytes: Buffer.from(decomposed, "utf8") }),
		).resolves.toMatchObject({ markdown: "한글" });
	});

	it("forwards kordoc options through the default parser registry", async () => {
		const bytes = Buffer.from("registry HWP bytes");
		let receivedBytes: Uint8Array | undefined;
		const registry = createDefaultParserRegistry({
			kordoc: {
				parse: async (inputBytes) => {
					receivedBytes = inputBytes;
					return { success: true, fileType: "hwp", markdown: "Registry HWP marker" };
				},
			},
		});
		const parser = registry.getForVirtualPath("/docs/registry.hwp");

		await expect(parser?.parse({ virtualPath: "/docs/registry.hwp", bytes })).resolves.toMatchObject({
			markdown: "Registry HWP marker",
			metadata: { parser: "kordoc", format: "hwp" },
		});
		expect(receivedBytes).toBe(bytes);
	});

	it("rejects malformed legacy HWP bytes with a typed parser error", async () => {
		const registry = createDefaultParserRegistry();
		const hwpParser = registry.getForVirtualPath("/docs/legacy.hwp");

		expect(hwpParser).toBeDefined();
		await expect(
			hwpParser?.parse({ virtualPath: "/docs/legacy.hwp", bytes: Buffer.from("not hwp5") }),
		).rejects.toBeInstanceOf(ParseError);
	});

	it("parses legacy XLS worksheets through the default registry", async () => {
		const registry = createDefaultParserRegistry();
		const xlsParser = registry.getForVirtualPath("/docs/legacy.xls");

		expect(xlsParser).toBeDefined();
		await expect(
			xlsParser?.parse({ virtualPath: "/docs/legacy.xls", bytes: createXlsFixture("Legacy XLS marker") }),
		).resolves.toMatchObject({
			markdown: expect.stringContaining("| Topic | Legacy XLS marker |"),
			metadata: { parser: "kordoc", format: "xls" },
		});
	});

	it("rejects malformed legacy XLS bytes with a typed parser error", async () => {
		const registry = createDefaultParserRegistry();
		const xlsParser = registry.getForVirtualPath("/docs/legacy.xls");

		expect(xlsParser).toBeDefined();
		await expect(
			xlsParser?.parse({ virtualPath: "/docs/legacy.xls", bytes: Buffer.from("not xls") }),
		).rejects.toBeInstanceOf(ParseError);
	});

	it("wraps corrupt OLE-prefixed legacy XLS input in a typed parser error", async () => {
		const registry = createDefaultParserRegistry();
		const xlsParser = registry.getForVirtualPath("/docs/corrupt.xls");
		const bytes = Buffer.from([0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1, 0, 0, 0, 0]);

		await expect(xlsParser?.parse({ virtualPath: "/docs/corrupt.xls", bytes })).rejects.toBeInstanceOf(ParseError);
	});

	it("preserves legacy XLS cell boundaries and values in escaped Markdown", async () => {
		const registry = createDefaultParserRegistry();
		const xlsParser = registry.getForVirtualPath("/docs/rich.xls");

		const parsed = await xlsParser?.parse({
			virtualPath: "/docs/rich.xls",
			bytes: createRichXlsFixture(),
		});

		// kordoc trims surrounding cell whitespace; the value itself stays intact.
		expect(parsed?.markdown).toContain("| Whitespace | preserve me |");
		expect(parsed?.markdown).toContain("left \\| right");
		expect(parsed?.markdown).toContain("first line<br>second line");
		expect(parsed?.markdown).toContain("C:\\docs\\file.xls");
		expect(parsed?.markdown).toContain("## Details");
		expect(parsed?.markdown).toContain("Unicode | 한글 marker");
		expect(parsed?.markdown).toContain("Number | 42");
		expect(parsed?.markdown).toContain("Boolean | TRUE");
	});
});
