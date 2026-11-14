import { describe, expect, it } from "vitest";
import { alignPages } from "../../src/parser/pages.ts";

const DOC = ["# Title", "", "alpha one", "alpha two", "", "bravo one", "", "charlie one", "charlie two", ""].join("\n");

describe("alignPages", () => {
	it("maps each page's markdown to the 1-based line range it occupies", () => {
		const pages = alignPages(
			DOC,
			[
				{ page: 1, markdown: "# Title\n\nalpha one\nalpha two" },
				{ page: 2, markdown: "bravo one" },
				{ page: 3, markdown: "charlie one\ncharlie two" },
			],
			"layout",
			3,
		);
		expect(pages).toEqual({
			mode: "layout",
			pageCount: 3,
			spans: [
				{ page: 1, lineStart: 1, lineEnd: 4 },
				{ page: 2, lineStart: 6, lineEnd: 6 },
				{ page: 3, lineStart: 8, lineEnd: 9 },
			],
		});
	});

	it("skips a page with empty markdown (a table continued across pages) without moving its neighbours", () => {
		const pages = alignPages(
			DOC,
			[
				{ page: 1, markdown: "# Title\n\nalpha one\nalpha two" },
				{ page: 2, markdown: "" },
				{ page: 3, markdown: "charlie one\ncharlie two" },
			],
			"layout",
			3,
		);
		expect(pages?.spans).toEqual([
			{ page: 1, lineStart: 1, lineEnd: 4 },
			{ page: 3, lineStart: 8, lineEnd: 9 },
		]);
	});

	it("skips a page whose markdown is not in the document instead of guessing", () => {
		const pages = alignPages(
			DOC,
			[
				{ page: 1, markdown: "alpha one\nalpha two" },
				{ page: 2, markdown: "text kordoc rendered differently" },
				{ page: 3, markdown: "charlie one" },
			],
			"layout",
		);
		expect(pages?.spans.map((span) => span.page)).toEqual([1, 3]);
	});

	it("matches repeated page text in order, never twice at the same place", () => {
		const doc = "same\n\nsame\n\nsame\n";
		const pages = alignPages(
			doc,
			[
				{ page: 1, markdown: "same" },
				{ page: 2, markdown: "same" },
				{ page: 3, markdown: "same" },
			],
			"layout",
		);
		expect(pages?.spans).toEqual([
			{ page: 1, lineStart: 1, lineEnd: 1 },
			{ page: 2, lineStart: 3, lineEnd: 3 },
			{ page: 3, lineStart: 5, lineEnd: 5 },
		]);
	});

	it("lets two pages share the line kordoc joined across the page break", () => {
		const doc = "head\n\nend of page one start of page two\n\ntail\n";
		const pages = alignPages(
			doc,
			[
				{ page: 1, markdown: "head\n\nend of page one" },
				{ page: 2, markdown: "start of page two\n\ntail" },
			],
			"layout",
		);
		expect(pages?.spans).toEqual([
			{ page: 1, lineStart: 1, lineEnd: 3 },
			{ page: 2, lineStart: 3, lineEnd: 5 },
		]);
	});

	it("ignores whitespace differences, including a line break kordoc turned into a space", () => {
		const pages = alignPages(
			"one two\nthree\n\nfour\n",
			[
				{ page: 1, markdown: "one\ntwo three" },
				{ page: 2, markdown: "four" },
			],
			"layout",
		);
		expect(pages?.spans).toEqual([
			{ page: 1, lineStart: 1, lineEnd: 2 },
			{ page: 2, lineStart: 4, lineEnd: 4 },
		]);
	});

	it("takes a page that does not start where the previous one ended only when its text is unique", () => {
		const doc = "intro\n\nrepeat\n\nmiddle\n\nrepeat\n";
		expect(
			alignPages(
				doc,
				[
					{ page: 1, markdown: "intro" },
					{ page: 2, markdown: "gap text not in doc" },
					{ page: 3, markdown: "repeat" },
				],
				"layout",
			)?.spans.map((span) => span.page),
		).toEqual([1]);
		expect(
			alignPages(
				doc,
				[
					{ page: 1, markdown: "intro" },
					{ page: 2, markdown: "middle" },
				],
				"layout",
			)?.spans,
		).toEqual([
			{ page: 1, lineStart: 1, lineEnd: 1 },
			{ page: 2, lineStart: 5, lineEnd: 5 },
		]);
	});

	it("keeps the mode so a section approximation is never mistaken for a layout page", () => {
		expect(alignPages("a\n\nb\n", [{ page: 1, markdown: "a" }], "section")?.mode).toBe("section");
	});

	it("reports nothing when no page aligns", () => {
		expect(alignPages("a\n", [], "layout")).toBeUndefined();
		expect(alignPages("a\n", [{ page: 1, markdown: "" }], "layout")).toBeUndefined();
	});

	it("ignores pages that go backwards", () => {
		const pages = alignPages(
			"a\n\nb\n",
			[
				{ page: 2, markdown: "b" },
				{ page: 1, markdown: "a" },
			],
			"layout",
		);
		expect(pages?.spans).toEqual([{ page: 2, lineStart: 3, lineEnd: 3 }]);
	});
});

import { readFile } from "node:fs/promises";
import { KordocParser } from "../../src/parser/kordoc.ts";
import { PptxParser } from "../../src/parser/office.ts";
import { locateChunk } from "../../src/retrieval/chunk-location.ts";
import { createDeckPptxFixture, createDocxFixture, createPptxFixture } from "../fixtures/document-formats.ts";
import { createMultiPagePdfBuffer } from "../fixtures/minimal-pdf.ts";

describe("KordocParser page index", () => {
	it("places every page of a multi-page PDF at the lines kordoc rendered it on", async () => {
		const bytes = createMultiPagePdfBuffer([
			["Alpha first page", "alpha second line"],
			["Bravo second page unique", "bravo more"],
			["Charlie third page", "charlie end"],
		]);
		const parsed = await new KordocParser().parse({ virtualPath: "/docs/a.pdf", bytes });
		expect(parsed.pages).toEqual({
			mode: "layout",
			pageCount: 3,
			spans: [
				{ page: 1, lineStart: 1, lineEnd: 3 },
				{ page: 2, lineStart: 5, lineEnd: 7 },
				{ page: 3, lineStart: 9, lineEnd: 11 },
			],
		});
		const lines = parsed.markdown.split("\n");
		expect(lines[4]).toBe("Bravo second page unique");
	});

	it("places pages whose boundary kordoc joined into one markdown line, and reports both pages for a chunk on it", async () => {
		const bytes = createMultiPagePdfBuffer([
			["Quarterly overview", "Zephyrine intro paragraph on page one"],
			["Kumquat budget table", "Only page two mentions the kumquat ledger"],
			["Appendix glossary", "Vermilion closing remarks on page three"],
		]);
		const parsed = await new KordocParser().parse({ virtualPath: "/docs/a.pdf", bytes });
		expect(parsed.markdown.split("\n")[2]).toBe("Zephyrine intro paragraph on page one Kumquat budget table");
		expect(parsed.pages?.spans).toEqual([
			{ page: 1, lineStart: 1, lineEnd: 3 },
			{ page: 2, lineStart: 3, lineEnd: 5 },
			{ page: 3, lineStart: 5, lineEnd: 7 },
		]);
		const only = locateChunk(parsed.markdown, "Vermilion closing remarks on page three", parsed.pages);
		expect(only).toMatchObject({ lineStart: 7, pageStart: 3, pageEnd: 3 });
		const shared = locateChunk(
			parsed.markdown,
			"Zephyrine intro paragraph on page one Kumquat budget table",
			parsed.pages,
		);
		expect(shared).toMatchObject({ lineStart: 3, pageStart: 1, pageEnd: 2 });
	});

	it("tolerates kordoc returning fewer pages than pageCount (real HWP5 fixture) and keeps the placed pages exact", async () => {
		const bytes = await readFile(new URL("../fixtures/hwp5/minimal-body-table.hwp", import.meta.url));
		const parsed = await new KordocParser().parse({ virtualPath: "/docs/a.hwp", bytes });
		expect(parsed.pages?.mode).toBe("layout");
		expect(parsed.pages?.pageCount).toBe(3);
		expect(parsed.pages?.spans.map((span) => span.page)).toEqual([1, 2]);
		const [first, second] = parsed.pages?.spans ?? [];
		expect(first?.lineStart).toBe(1);
		expect(second?.lineStart).toBeGreaterThan(first?.lineEnd ?? 0);
	});

	it("keeps pageMode section so the approximation can be told from a real page", async () => {
		const parser = new KordocParser({
			parse: async () => ({
				success: true,
				fileType: "hwpx",
				markdown: "one\n\ntwo\n",
				pages: [
					{ pageNumber: 1, markdown: "one" },
					{ pageNumber: 2, markdown: "two" },
				],
				metadata: { pageCount: 2, pageMode: "section" },
			}),
		});
		const parsed = await parser.parse({ virtualPath: "/docs/a.hwpx", bytes: new Uint8Array() });
		expect(parsed.pages?.mode).toBe("section");
	});

	it("reports no pages for a format kordoc gives none for, and for pages without a pageMode", async () => {
		const docx = await new KordocParser().parse({
			virtualPath: "/docs/a.docx",
			bytes: await createDocxFixture("plain words"),
		});
		expect(docx.pages).toBeUndefined();
		const unmoded = new KordocParser({
			parse: async () => ({
				success: true,
				markdown: "one\n",
				pages: [{ pageNumber: 1, markdown: "one" }],
			}),
		});
		expect((await unmoded.parse({ virtualPath: "/docs/a.pdf", bytes: new Uint8Array() })).pages).toBeUndefined();
	});

	it("aligns against the NFC text the mirror stores", async () => {
		const decomposed = "한글".normalize("NFD");
		const parser = new KordocParser({
			parse: async () => ({
				success: true,
				markdown: `${decomposed}\n\nnext\n`,
				pages: [
					{ pageNumber: 1, markdown: decomposed },
					{ pageNumber: 2, markdown: "next" },
				],
				metadata: { pageCount: 2, pageMode: "layout" },
			}),
		});
		const parsed = await parser.parse({ virtualPath: "/docs/a.pdf", bytes: new Uint8Array() });
		expect(parsed.pages?.spans).toHaveLength(2);
	});

	it("changes the parser version so existing mirrors re-parse and gain pages", () => {
		expect(new KordocParser().version).toMatch(/\+pages1$/u);
		expect(new PptxParser().version).not.toBe("1");
	});
});

describe("PptxParser page index", () => {
	it("numbers slides in presentation order, not file-name order, and keeps a slide's note on its slide", async () => {
		const bytes = await createDeckPptxFixture({
			slides: { 1: "Intro marker", 2: "Moved last marker", 3: "Middle marker" },
			order: [1, 3, 2],
			notes: { 3: "Note for middle" },
		});
		const parsed = await new PptxParser().parse({ virtualPath: "/docs/deck.pptx", bytes });
		expect(parsed.markdown).toBe("Intro marker\n\nMiddle marker\n\nNote for middle\n\nMoved last marker");
		expect(parsed.pages).toEqual({
			mode: "layout",
			pageCount: 3,
			spans: [
				{ page: 1, lineStart: 1, lineEnd: 1 },
				{ page: 2, lineStart: 3, lineEnd: 5 },
				{ page: 3, lineStart: 7, lineEnd: 7 },
			],
		});
	});

	it("keeps a blank slide's number so the next slide is not renumbered", async () => {
		const bytes = await createDeckPptxFixture({
			slides: { 1: "First", 2: "   ", 3: "Third" },
			order: [1, 2, 3],
		});
		const parsed = await new PptxParser().parse({ virtualPath: "/docs/deck.pptx", bytes });
		expect(parsed.pages?.spans.map((span) => span.page)).toEqual([1, 3]);
		expect(parsed.pages?.pageCount).toBe(3);
	});

	it("reports no pages when the deck has no presentation order, and keeps its text", async () => {
		const parsed = await new PptxParser().parse({
			virtualPath: "/docs/deck.pptx",
			bytes: await createPptxFixture("Orderless marker"),
		});
		expect(parsed.markdown).toBe("Orderless marker");
		expect(parsed.pages).toBeUndefined();
	});
});
