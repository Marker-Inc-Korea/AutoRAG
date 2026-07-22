import { Readable } from "node:stream";
import JSZip from "jszip";
import { describe, expect, it, vi } from "vitest";
import { createDefaultParserRegistry, ParseError } from "../../src/parser/index.ts";
import { readZipXmlText } from "../../src/parser/xml-text.ts";

describe("ZIP XML parser budgets", () => {
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

	it("rejects ZIP XML members that exceed the streaming byte budget without metadata", async () => {
		// Given: a ZIP member whose declared uncompressed size is unavailable.
		const originalLoadAsync = JSZip.loadAsync.bind(JSZip);
		const zip = new JSZip();
		zip.file("word/document.xml", "<w:t>small</w:t>");
		const bytes = Buffer.from(await zip.generateAsync({ type: "uint8array", compression: "DEFLATE" }));
		const largeChunk = Buffer.alloc(5_000_001, "x");
		vi.spyOn(JSZip, "loadAsync").mockImplementationOnce(async (input) => {
			const loaded = await originalLoadAsync(input);
			const file = loaded.file("word/document.xml");
			if (file === null) throw new Error("fixture missing document.xml");
			const data = Reflect.get(file, "_data");
			if (typeof data === "object" && data !== null) Reflect.set(data, "uncompressedSize", undefined);
			file.nodeStream = () => Readable.from([largeChunk]);
			return loaded;
		});

		// When/Then: streamed bytes are counted and rejected even without trusted metadata.
		await expect(readZipXmlText(bytes, /^word\/document\.xml$/)).rejects.toThrow(/exceeds limit/);
	});

	it("rejects ZIP XML documents that expand beyond the text chunk budget", async () => {
		// Given: one XML file under the byte budget but over the text-node expansion budget.
		const zip = new JSZip();
		const xml = `<root>${Array.from({ length: 20_001 }, (_, index) => `<w:t>${index}</w:t>`).join("")}</root>`;
		zip.file("word/document.xml", xml);
		const bytes = Buffer.from(await zip.generateAsync({ type: "uint8array", compression: "DEFLATE" }));

		// When/Then: extracted text chunk expansion is bounded independently from bytes.
		await expect(readZipXmlText(bytes, /^word\/document\.xml$/)).rejects.toThrow(/text chunk count/);
	});
});
