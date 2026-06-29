import JSZip from "jszip";
import { describe, expect, it, vi } from "vitest";
import {
	createDefaultParserRegistry,
	ImageOcrParser,
	OpendataloaderPdfParser,
	ParseError,
	Parser,
	ParserRegistry,
	type PdfConverter,
	PlainTextParser,
} from "../../src/parser/index.ts";
import { readZipXmlText } from "../../src/parser/xml-text.ts";
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

const tesseractMock = vi.hoisted(() => ({
	createWorker: vi.fn(),
}));

vi.mock("tesseract.js", () => tesseractMock);

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

	it("OCR image parser is opt-in and enforces timeout budgets", async () => {
		// Given: the default registry and explicitly enabled OCR registries with budget controls.
		let abortObserved = false;
		const disabled = createDefaultParserRegistry();
		const timed = createDefaultParserRegistry({
			ocr: {
				enabled: true,
				timeoutMs: 1,
				engine: (input) =>
					new Promise<string>(() => {
						input.signal.addEventListener("abort", () => {
							abortObserved = true;
						});
					}),
			},
		});
		const budgeted = createDefaultParserRegistry({
			ocr: {
				enabled: true,
				maxBytes: 3,
				engine: async () => "OCR marker",
			},
		});

		// When/Then: images are invisible by default but become routed when OCR is explicitly enabled.
		expect(disabled.getForVirtualPath("/docs/scan.png")).toBeUndefined();
		const imageParser = timed.getForVirtualPath("/docs/scan.png");
		expect(imageParser).toBeDefined();
		await expect(
			imageParser?.parse({ virtualPath: "/docs/scan.png", bytes: Buffer.from([0x89, 0x50, 0x4e, 0x47]) }),
		).rejects.toThrow(/timed out/i);
		expect(abortObserved).toBe(true);
		await expect(
			budgeted.getForVirtualPath("/docs/large.png")?.parse({
				virtualPath: "/docs/large.png",
				bytes: Buffer.from([0x89, 0x50, 0x4e, 0x47]),
			}),
		).rejects.toBeInstanceOf(ParseError);
	});

	it("OCR timeout waits for engine cleanup before returning", async () => {
		// Given: an OCR engine that starts cleanup only after the abort signal.
		let cleanupCompleted = false;
		const parser = new ImageOcrParser({
			enabled: true,
			timeoutMs: 1,
			engine: (input) =>
				new Promise<string>(() => {
					input.signal.addEventListener("abort", () => {
						cleanupCompleted = true;
					});
				}),
		});

		// When: parsing times out.
		await expect(
			parser.parse({ virtualPath: "/docs/scan.png", bytes: Buffer.from([0x89, 0x50]) }),
		).rejects.toBeInstanceOf(ParseError);

		// Then: cleanup has completed before parse() resolves/rejects.
		expect(cleanupCompleted).toBe(true);
	});

	it("OCR timeout waits for Tesseract worker termination before returning", async () => {
		// Given: the real OCR adapter observes a timeout while Tesseract termination is still pending.
		let rejectRecognition: (error: Error) => void = () => undefined;
		let finishTermination: () => void = () => undefined;
		tesseractMock.createWorker.mockResolvedValueOnce({
			recognize: () =>
				new Promise<never>((_, reject) => {
					rejectRecognition = reject;
				}),
			terminate: () =>
				new Promise<void>((resolve) => {
					finishTermination = resolve;
				}),
		});
		const parser = new ImageOcrParser({ enabled: true, timeoutMs: 1 });
		const result = parser.parse({ virtualPath: "/docs/scan.png", bytes: Buffer.from([0x89, 0x50]) });
		await new Promise((resolve) => setTimeout(resolve, 10));

		// When: recognition rejects because termination started, but termination has not completed.
		rejectRecognition(new Error("terminated"));
		await Promise.resolve();

		// Then: parse() must still wait for terminate() to finish before rejecting.
		let settled = false;
		result.then(
			() => {
				settled = true;
			},
			() => {
				settled = true;
			},
		);
		await Promise.resolve();
		expect(settled).toBe(false);
		finishTermination();
		await expect(result).rejects.toBeInstanceOf(ParseError);
		expect(settled).toBe(true);
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

	it("legacy binary HWP and XLS files are routed but rejected with typed parser errors", async () => {
		// Given: legacy HWP5 and XLS bytes, which require separate binary parsers from HWPX/XLSX.
		const registry = createDefaultParserRegistry();
		const hwpParser = registry.getForVirtualPath("/docs/legacy.hwp");
		const xlsParser = registry.getForVirtualPath("/docs/legacy.xls");

		// When/Then: the default pipeline handles the extensions safely instead of crashing or emitting garbage.
		expect(hwpParser).toBeDefined();
		await expect(
			hwpParser?.parse({ virtualPath: "/docs/legacy.hwp", bytes: Buffer.from("not hwp5") }),
		).rejects.toBeInstanceOf(ParseError);
		expect(xlsParser).toBeDefined();
		await expect(
			xlsParser?.parse({ virtualPath: "/docs/legacy.xls", bytes: Buffer.from("not xls") }),
		).rejects.toBeInstanceOf(ParseError);
	});

	it("rejects oversized zipped XML documents before extraction", async () => {
		// Given: a DOCX-like ZIP with more XML files than the parser budget allows.
		const zip = new JSZip();
		for (let index = 0; index < 65; index += 1) {
			zip.file(`word/header${index}.xml`, `<w:t>oversized ${index}</w:t>`);
		}
		const bytes = Buffer.from(await zip.generateAsync({ type: "uint8array" }));
		const parser = createDefaultParserRegistry().getForVirtualPath("/docs/oversized.docx");

		// When/Then: the archive is rejected through the typed parser boundary.
		await expect(parser?.parse({ virtualPath: "/docs/oversized.docx", bytes })).rejects.toBeInstanceOf(ParseError);
	});

	it("rejects ZIP XML members with oversized uncompressed metadata before streaming", async () => {
		// Given: a highly-compressible XML member whose uncompressed central-directory size exceeds the budget.
		const zip = new JSZip();
		zip.file("word/document.xml", `<w:t>${"x".repeat(5_000_001)}</w:t>`);
		const bytes = Buffer.from(await zip.generateAsync({ type: "uint8array", compression: "DEFLATE" }));
		const loaded = await JSZip.loadAsync(bytes);
		const sample = loaded.file("word/document.xml");
		if (sample === null) throw new Error("fixture missing document.xml");
		const prototype = Object.getPrototypeOf(sample) as { nodeStream: JSZip.JSZipObject["nodeStream"] };
		const originalNodeStream = prototype.nodeStream;
		prototype.nodeStream = () => {
			throw new Error("nodeStream should not run for oversized metadata");
		};
		try {
			// When/Then: metadata rejects the member before content streaming starts.
			await expect(readZipXmlText(bytes, /^word\/document\.xml$/)).rejects.toThrow(/exceeds limit/);
		} finally {
			prototype.nodeStream = originalNodeStream;
		}
	});
});
