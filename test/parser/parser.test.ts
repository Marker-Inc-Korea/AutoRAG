import { describe, expect, it } from "vitest";
import {
	createDefaultParserRegistry,
	ParseError,
	Parser,
	ParserRegistry,
	PlainTextParser,
} from "../../src/parser/index.ts";
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
});
