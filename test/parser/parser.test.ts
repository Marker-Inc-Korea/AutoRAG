import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
	createDefaultParserRegistry,
	HwpParser,
	OpendataloaderPdfParser,
	ParseError,
	Parser,
	ParserRegistry,
	type PdfConverter,
	PlainTextParser,
} from "../../src/parser/index.ts";
import {
	createDocxFixture,
	createEmlFixture,
	createEucKrEmlFixture,
	createHwpxFixture,
	createPptxFixture,
	createXlsxFixture,
} from "../fixtures/document-formats.ts";
import { createMinimalPdfBuffer } from "../fixtures/minimal-pdf.ts";

const pdfMarker = "OpenDataLoader AutoRAG PDF marker refund policy alpha";

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

	it("passes opt-in scanned-PDF OCR fallback options to the OpenDataLoader convert API", async () => {
		// Given: a PDF parser configured for hybrid OCR fallback with an injected converter.
		const calls: Array<{ readonly hybrid?: string; readonly hybridMode?: string; readonly hybridTimeout?: string }> =
			[];
		const converter: PdfConverter = async (inputPath, options) => {
			calls.push({
				hybrid: options.hybrid,
				hybridMode: options.hybridMode,
				hybridTimeout: options.hybridTimeout,
			});
			const outputDir = options.outputDir;
			if (outputDir === undefined) throw new Error("expected parser to provide outputDir");
			await import("node:fs/promises").then((fs) =>
				fs.writeFile(
					`${outputDir}/${
						inputPath
							.split("/")
							.pop()
							?.replace(/\.pdf$/i, ".md") ?? "scanned.md"
					}`,
					"Hybrid OCR marker",
				),
			);
			return "ok";
		};
		const parser = new OpendataloaderPdfParser({
			converter,
			ocr: { enabled: true, hybrid: "docling-fast", hybridMode: "full", timeoutMs: 4_000, maxBytes: 4_096 },
		});

		// When: the PDF parser runs.
		const parsed = await parser.parse({ virtualPath: "/docs/scanned.pdf", bytes: createMinimalPdfBuffer("scanned") });

		// Then: the OpenDataLoader convert API receives the configured hybrid OCR controls.
		expect(parsed.markdown).toBe("Hybrid OCR marker");
		expect(calls).toEqual([{ hybrid: "docling-fast", hybridMode: "full", hybridTimeout: "4000" }]);
	});

	it("parses legacy binary HWP through an injected extractor", async () => {
		const parser = new HwpParser({
			extractor: async () => ({
				paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "한글 HWP marker" }],
				tables: [],
			}),
		});

		await expect(
			parser.parse({ virtualPath: "/docs/legacy.hwp", bytes: Buffer.from("injected HWP bytes") }),
		).resolves.toEqual({
			markdown: "한글 HWP marker",
			metadata: { parser: "hwp", format: "hwp5" },
		});
	});

	it("forwards HWP options through the default parser registry", async () => {
		const bytes = Buffer.from("registry HWP bytes");
		let receivedBytes: Uint8Array | undefined;
		const registry = createDefaultParserRegistry({
			hwp: {
				extractor: async (inputBytes) => {
					receivedBytes = inputBytes;
					return {
						paragraphs: [{ sectionIndex: 0, paragraphIndex: 0, text: "Registry HWP marker" }],
						tables: [],
					};
				},
			},
		});
		const parser = registry.getForVirtualPath("/docs/registry.hwp");

		await expect(parser?.parse({ virtualPath: "/docs/registry.hwp", bytes })).resolves.toMatchObject({
			markdown: expect.stringContaining("Registry HWP marker"),
		});
		expect(receivedBytes).toBe(bytes);
	});

	it("parses a real HWP5 body and table", async () => {
		const bytes = await readFile(new URL("../fixtures/hwp5/minimal-body-table.hwp", import.meta.url));
		const registry = createDefaultParserRegistry();
		const parser = registry.getForVirtualPath("/docs/minimal-body-table.hwp");

		expect(parser).toBeInstanceOf(HwpParser);
		const parsed = await parser?.parse({ virtualPath: "/docs/minimal-body-table.hwp", bytes });

		expect(parsed?.markdown).toContain("편집 탭 – 표");
		expect(parsed?.markdown).toContain("Row 1: 제목 | 담당자 | 세부 내용");
		expect(parsed?.markdown).toContain("제목");
		expect(parsed?.markdown).toContain("담당자");
		expect(parsed?.markdown).toContain("세부 내용");
		expect(parsed?.metadata).toMatchObject({ format: "hwp5" });
	});

	it("rejects malformed legacy HWP bytes with a typed parser error", async () => {
		const registry = createDefaultParserRegistry();
		const hwpParser = registry.getForVirtualPath("/docs/legacy.hwp");

		expect(hwpParser).toBeDefined();
		await expect(
			hwpParser?.parse({ virtualPath: "/docs/legacy.hwp", bytes: Buffer.from("not hwp5") }),
		).rejects.toBeInstanceOf(ParseError);
	});

	it("routes legacy XLS files but rejects them with typed parser errors", async () => {
		const registry = createDefaultParserRegistry();
		const xlsParser = registry.getForVirtualPath("/docs/legacy.xls");

		expect(xlsParser).toBeDefined();
		await expect(
			xlsParser?.parse({ virtualPath: "/docs/legacy.xls", bytes: Buffer.from("not xls") }),
		).rejects.toBeInstanceOf(ParseError);
	});
});
