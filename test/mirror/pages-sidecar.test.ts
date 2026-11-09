import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { pagesSidecarPath, readPagesSidecar, writePagesSidecar } from "../../src/mirror/pages-sidecar.ts";
import type { ParsedPages } from "../../src/parser/pages.ts";

let dir: string;
let mirror: string;
const PAGES: ParsedPages = { mode: "layout", pageCount: 2, spans: [{ page: 1, lineStart: 1, lineEnd: 2 }] };

beforeEach(() => {
	dir = mkdtempSync(join(tmpdir(), "autorag-pages-sidecar-"));
	mkdirSync(join(dir, "files"));
	mkdirSync(join(dir, "pages"));
	mirror = join(dir, "files", "abc.md");
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

describe("pages sidecar", () => {
	it("round-trips the page index for the exact markdown it was written for", () => {
		writePagesSidecar(mirror, "one\ntwo\n", PAGES);
		expect(readPagesSidecar(mirror, "one\ntwo\n")).toEqual(PAGES);
	});

	it("returns nothing for markdown that changed after the sidecar was written", () => {
		writePagesSidecar(mirror, "one\ntwo\n", PAGES);
		expect(readPagesSidecar(mirror, "one\ntwo\nthree\n")).toBeUndefined();
	});

	it("keeps the sidecar out of the stage output directory", () => {
		writePagesSidecar(mirror, "one\n", PAGES);
		expect(pagesSidecarPath(mirror)).toBe(join(dir, "pages", "abc.json"));
	});

	it("removes the sidecar when the re-parse has no pages, so a stale index never outlives its format", () => {
		writePagesSidecar(mirror, "one\ntwo\n", PAGES);
		writePagesSidecar(mirror, "one\ntwo\n", undefined);
		expect(existsSync(pagesSidecarPath(mirror))).toBe(false);
		expect(readPagesSidecar(mirror, "one\ntwo\n")).toBeUndefined();
	});

	it("ignores a corrupt, foreign-version or malformed sidecar", () => {
		const sidecar = pagesSidecarPath(mirror);
		for (const content of [
			"not json",
			JSON.stringify({ version: 9 }),
			JSON.stringify({
				version: 1,
				markdownSha256: "x",
				pages: { mode: "layout", spans: [{ page: 0, lineStart: 1, lineEnd: 1 }] },
			}),
		]) {
			mkdirSync(dirname(sidecar), { recursive: true });
			writeFileSync(sidecar, content);
			expect(readPagesSidecar(mirror, "one\n")).toBeUndefined();
		}
	});

	it("rejects spans that overlap or run backwards", () => {
		writePagesSidecar(mirror, "a\n", {
			mode: "layout",
			spans: [
				{ page: 1, lineStart: 3, lineEnd: 5 },
				{ page: 2, lineStart: 4, lineEnd: 6 },
			],
		});
		expect(readPagesSidecar(mirror, "a\n")).toBeUndefined();
	});
});

import { mkdirSync as mkdir, mkdtempSync as mkTemp, rmSync as rm, unlinkSync, writeFileSync as write } from "node:fs";
import { loadMirrorIndex, syncParsedMirrors } from "../../src/mirror/index.ts";
import { type ParseInput, type ParseOutput, ParserRegistry, PlainTextParser } from "../../src/parser/index.ts";

class PagedParser extends PlainTextParser {
	pages: ParsedPages | undefined = PAGES;
	async parse(input: ParseInput): Promise<ParseOutput> {
		return {
			markdown: `${new TextDecoder().decode(input.bytes).trim()}\n`,
			...(this.pages === undefined ? {} : { pages: this.pages }),
		};
	}
}

describe("mirror sync keeps the page sidecar in step with the mirror entry", () => {
	let root: string;
	let docs: string;
	beforeEach(() => {
		root = mkTemp(join(tmpdir(), "autorag-pages-sync-"));
		docs = join(root, "docs");
		mkdir(docs);
		write(join(docs, "a.txt"), "Alpha\nBeta\n");
	});
	afterEach(() => rm(root, { recursive: true, force: true }));

	function outputPath(): string {
		const entry = loadMirrorIndex(root).entries["/docs/a.txt"];
		if (!entry) throw new Error("missing entry");
		return entry.outputPath;
	}
	const refresh = (parser: PlainTextParser, force = false) =>
		syncParsedMirrors({ root, searchPaths: [docs], registry: new ParserRegistry([parser]), force });

	it("writes the sidecar with the parse, replaces it on re-parse, and drops it when pages disappear", async () => {
		const parser = new PagedParser();
		await refresh(parser);
		const mirrorFile = outputPath();
		expect(readPagesSidecar(mirrorFile, "Alpha\nBeta\n")).toEqual(PAGES);

		parser.pages = { mode: "layout", spans: [{ page: 4, lineStart: 1, lineEnd: 2 }] };
		await refresh(parser, true);
		expect(readPagesSidecar(mirrorFile, "Alpha\nBeta\n")?.spans[0]?.page).toBe(4);

		parser.pages = undefined;
		await refresh(parser, true);
		expect(existsSync(pagesSidecarPath(mirrorFile))).toBe(false);
	});

	it("removes the sidecar when the source is deleted", async () => {
		await refresh(new PagedParser());
		const mirrorFile = outputPath();
		expect(existsSync(pagesSidecarPath(mirrorFile))).toBe(true);

		unlinkSync(join(docs, "a.txt"));
		write(join(docs, "keep.txt"), "keep\n");
		await refresh(new PagedParser());
		expect(existsSync(pagesSidecarPath(mirrorFile))).toBe(false);
	});
});
