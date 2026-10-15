import {
	copyFileSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	renameSync,
	rmSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	detectMirrorStaleness,
	loadMirrorIndex,
	type ParsedMirrorIndex,
	parsedMirrorIndexPath,
	parsedMirrorRoot,
	saveMirrorIndex,
	syncParsedMirrors,
} from "../../src/mirror/index.ts";
import {
	createDefaultParserRegistry,
	type ParseInput,
	type ParseOutput,
	ParserRegistry,
	PlainTextParser,
} from "../../src/parser/index.ts";
import { createDocxFixture, createXlsFixture } from "../fixtures/document-formats.ts";

let root: string;
let source: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-mirror-test-"));
	source = join(root, "docs");
	mkdirSync(source, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function outputPaths(index: ParsedMirrorIndex): string[] {
	return Object.values(index.entries).map((entry) => entry.outputPath);
}

function requireValue<T>(value: T | undefined, label: string): T {
	if (value === undefined) throw new Error(`missing ${label}`);
	return value;
}

/** Unlinks the next scanned file while the first text file is parsed. */
class DeleteNextFileParser extends PlainTextParser {
	async parse(input: ParseInput): Promise<ParseOutput> {
		if (input.virtualPath === "/docs/a.txt") unlinkSync(join(source, "b.txt"));
		return super.parse(input);
	}
}

describe("syncParsedMirrors", () => {
	it("writes legacy XLS worksheet text to the parsed mirror", async () => {
		writeFileSync(join(source, "legacy.xls"), createXlsFixture("Mirror legacy XLS marker"));

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const outputPath = requireValue(
			loadMirrorIndex(root).entries["/docs/legacy.xls"]?.outputPath,
			"legacy XLS output path",
		);
		const markdown = readFileSync(outputPath, "utf8");

		expect(result).toMatchObject({ scanned: 1, written: 1, skipped: 0 });
		expect(markdown).toContain("Mirror legacy XLS marker");
	});

	it("writes legacy HWP5 body and table text to the parsed mirror", async () => {
		copyFileSync(
			new URL("../fixtures/hwp5/minimal-body-table.hwp", import.meta.url),
			join(source, "minimal-body-table.hwp"),
		);

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const outputPath = requireValue(
			loadMirrorIndex(root).entries["/docs/minimal-body-table.hwp"]?.outputPath,
			"HWP5 output path",
		);
		const markdown = readFileSync(outputPath, "utf8");

		expect(result).toMatchObject({ scanned: 1, written: 1, skipped: 0 });
		expect(markdown).toContain("편집 탭 – 표");
		// kordoc writes real table markup instead of the retired "Row 1: a | b" flattening.
		expect(markdown).toContain("<td>셀 편집</td>");
		expect(markdown).toContain("| 제목 | 담당자 | 세부 내용 |");
	});

	it("creates real markdown files and an index for supported virtual files", async () => {
		writeFileSync(join(source, "note.txt"), "Alpha\n");
		writeFileSync(join(source, "skip.bin"), Buffer.from([0, 1]));

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const index = loadMirrorIndex(root);
		const paths = outputPaths(index);

		expect(result).toMatchObject({ scanned: 1, written: 1, deleted: 0 });
		expect(paths).toHaveLength(1);
		const outputPath = requireValue(paths[0], "output path");
		expect(outputPath).toContain(parsedMirrorRoot(root));
		expect(readFileSync(outputPath, "utf8")).toBe("Alpha\n");
		expect(index.entries["/docs/note.txt"]?.sourcePath).toBe(join(source, "note.txt"));
		expect(index.entries["/docs/skip.bin"]).toBeUndefined();
	});

	it("updates changed content in place", async () => {
		const file = join(source, "note.md");
		writeFileSync(file, "Alpha\n");
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const first = loadMirrorIndex(root).entries["/docs/note.md"]?.outputPath;

		writeFileSync(file, "Beta\n");
		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const second = loadMirrorIndex(root).entries["/docs/note.md"]?.outputPath;

		expect(result.written).toBe(1);
		expect(second).toBe(first);
		expect(readFileSync(requireValue(second, "updated output path"), "utf8")).toBe("Beta\n");
	});

	it("indexes directories outside the workspace root with stable source-root prefixes", async () => {
		const externalRoot = mkdtempSync(join(tmpdir(), "autorag-mirror-external-"));
		try {
			const externalDocs = join(externalRoot, "docs");
			mkdirSync(join(externalDocs, "sub"), { recursive: true });
			writeFileSync(join(externalDocs, "root.txt"), "External root\n");
			writeFileSync(join(externalDocs, "sub", "note.md"), "External nested\n");

			const result = await syncParsedMirrors({
				root,
				searchPaths: [externalDocs],
				registry: createDefaultParserRegistry(),
			});
			const index = loadMirrorIndex(root);

			expect(result.written).toBe(2);
			expect(Object.keys(index.entries).sort()).toEqual(["/docs/root.txt", "/docs/sub/note.md"]);
		} finally {
			rmSync(externalRoot, { recursive: true, force: true });
		}
	});

	it("removes stale parsed files and relinks moved files without stale entries", async () => {
		const keep = join(source, "keep.txt");
		const move = join(source, "move.txt");
		writeFileSync(keep, "Keep\n");
		writeFileSync(move, "Move\n");
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const before = loadMirrorIndex(root);
		const deletedOutput = before.entries["/docs/keep.txt"]?.outputPath;
		const oldMoveOutput = before.entries["/docs/move.txt"]?.outputPath;

		unlinkSync(keep);
		renameSync(move, join(source, "moved.txt"));
		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const after = loadMirrorIndex(root);

		expect(result.deleted).toBe(2);
		expect(result.written).toBe(1);
		expect(after.entries["/docs/keep.txt"]).toBeUndefined();
		expect(after.entries["/docs/move.txt"]).toBeUndefined();
		expect(after.entries["/docs/moved.txt"]?.outputPath).toBeDefined();
		expect(existsSync(requireValue(deletedOutput, "deleted output path"))).toBe(false);
		expect(existsSync(requireValue(oldMoveOutput, "old move output path"))).toBe(false);
		const movedOutput = requireValue(after.entries["/docs/moved.txt"]?.outputPath, "moved output path");
		expect(readFileSync(movedOutput, "utf8")).toBe("Move\n");
	});

	it("ignores poisoned index output paths when updating changed files", async () => {
		const file = join(source, "note.txt");
		const outside = join(root, "outside.md");
		writeFileSync(file, "Alpha\n");
		writeFileSync(outside, "do not touch\n");
		saveMirrorIndex(root, {
			version: 1,
			entries: {
				"/docs/note.txt": {
					virtualPath: "/docs/note.txt",
					sourcePath: file,
					outputPath: outside,
					parserName: "plain-text",
					sourceMtimeNs: 0,
					sourceSizeBytes: 0,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
			},
		});

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const safeOutput = requireValue(loadMirrorIndex(root).entries["/docs/note.txt"]?.outputPath, "safe output path");

		expect(result.written).toBe(1);
		expect(readFileSync(outside, "utf8")).toBe("do not touch\n");
		expect(safeOutput).not.toBe(outside);
		expect(safeOutput).toContain(parsedMirrorRoot(root));
		expect(readFileSync(safeOutput, "utf8")).toBe("Alpha\n");
	});

	it("ignores poisoned stale index output paths when deleting removed files", async () => {
		const outside = join(root, "outside-stale.md");
		writeFileSync(outside, "do not delete\n");
		saveMirrorIndex(root, {
			version: 1,
			entries: {
				"/docs/missing.txt": {
					virtualPath: "/docs/missing.txt",
					sourcePath: join(source, "missing.txt"),
					outputPath: outside,
					parserName: "plain-text",
					sourceMtimeNs: 0,
					sourceSizeBytes: 0,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
			},
		});

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const index = loadMirrorIndex(root);

		expect(result.deleted).toBe(1);
		expect(readFileSync(outside, "utf8")).toBe("do not delete\n");
		expect(index.entries["/docs/missing.txt"]).toBeUndefined();
	});

	it("drops stale unsupported index entries without following poisoned output paths", async () => {
		// Unsupported extensions are not re-scanned; residual index rows are swept on refresh.
		const unsupported = join(source, "skip.bin");
		const outside = join(root, "outside-unsupported.md");
		writeFileSync(unsupported, Buffer.from([0, 1]));
		writeFileSync(outside, "do not delete unsupported\n");
		saveMirrorIndex(root, {
			version: 1,
			entries: {
				"/docs/skip.bin": {
					virtualPath: "/docs/skip.bin",
					sourcePath: unsupported,
					outputPath: outside,
					parserName: "plain-text",
					sourceMtimeNs: 0,
					sourceSizeBytes: 0,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
			},
		});

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const index = loadMirrorIndex(root);

		expect(result.deleted).toBe(1);
		expect(result.skipped).toBe(0);
		expect(readFileSync(outside, "utf8")).toBe("do not delete unsupported\n");
		expect(index.entries["/docs/skip.bin"]).toBeUndefined();
	});
	it("ignores unknown extensions instead of emitting unsupported-file diagnostics", async () => {
		// Scans only registered parser extensions, so bare binaries never enter the worklist.
		writeFileSync(join(source, "note.txt"), "Alpha\n");
		writeFileSync(join(source, "skip.bin"), Buffer.from([0, 1]));

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const diag = result.diagnostics.find((d) => d.code === "unsupported-file");

		expect(result.scanned).toBe(1);
		expect(result.written).toBe(1);
		expect(diag).toBeUndefined();
		expect(JSON.stringify(result.diagnostics)).not.toContain(root);
	});

	it("returns a parser-failed diagnostic when a routed parser throws", async () => {
		// .hwp is routed by extension but fails with a typed ParseError (legacy binary).
		writeFileSync(join(source, "legacy.hwp"), Buffer.from([1, 2, 3, 4]));

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const diag = result.diagnostics.find((d) => d.code === "parser-failed");

		expect(diag?.source).toBe("/docs/legacy.hwp");
		expect(JSON.stringify(result.diagnostics)).not.toContain(root);
	});

	it("returns deleted-mirror diagnostics when previously indexed files disappear", async () => {
		const file = join(source, "gone.txt");
		writeFileSync(file, "Bye\n");
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });

		unlinkSync(file);
		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const diag = result.diagnostics.find((d) => d.code === "deleted-mirror");

		expect(diag?.source).toBe("/docs/gone.txt");
	});

	it("returns an empty diagnostics array when everything parses cleanly", async () => {
		writeFileSync(join(source, "ok.txt"), "Fine\n");
		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		expect(result.diagnostics).toEqual([]);
	});
	it("skips a standalone image below 3 pixels and still indexes a sibling text file", async () => {
		// PNG signature + IHDR width 2, height 36.
		writeFileSync(
			join(source, "tiny.png"),
			Buffer.from([
				0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00,
				0x00, 0x02, 0x00, 0x00, 0x00, 0x24,
			]),
		);
		writeFileSync(join(source, "note.txt"), "Alpha\n");

		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry({ ocr: { enabled: true } }),
		});
		const diagnostic = result.diagnostics.find((entry) => entry.code === "parser-skipped");
		const index = loadMirrorIndex(root);

		expect(result).toMatchObject({ written: 1, skipped: 1 });
		expect(diagnostic?.source).toBe("/docs/tiny.png");
		expect(diagnostic?.message).toContain("2");
		expect(diagnostic?.message).toContain("36");
		expect(JSON.stringify(result.diagnostics)).not.toContain(root);
		expect(index.entries["/docs/note.txt"]).toBeDefined();
		expect(index.entries["/docs/tiny.png"]).toBeUndefined();
		expect(index.skipped?.["/docs/tiny.png"]?.reason).toBe("parser-skipped");
	});

	it("skips sources larger than maxSourceBytes without writing a mirror", async () => {
		const file = join(source, "huge.txt");
		writeFileSync(file, "x".repeat(1024));
		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			maxSourceBytes: 10,
		});
		const index = loadMirrorIndex(root);
		expect(result.written).toBe(0);
		expect(result.skipped).toBe(1);
		expect(index.entries["/docs/huge.txt"]).toBeUndefined();
		expect(result.diagnostics.some((d) => d.code === "parser-skipped")).toBe(true);
	});

	it("removes a user-excluded source and records the exclusion reason", async () => {
		const file = join(source, "private.txt");
		writeFileSync(file, "Private\n");
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });

		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			userExcludedSourcePaths: new Set([file]),
		});
		const index = loadMirrorIndex(root);

		expect(result.skipped).toBe(1);
		expect(index.entries["/docs/private.txt"]).toBeUndefined();
		expect(index.skipped?.["/docs/private.txt"]?.reason).toBe("user-excluded");
	});

	it("excludes every source under an excluded directory", async () => {
		const dir = join(source, "private");
		mkdirSync(dir);
		writeFileSync(join(dir, "a.txt"), "A\n");
		writeFileSync(join(dir, "b.txt"), "B\n");
		writeFileSync(join(source, "keep.txt"), "Keep\n");
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });

		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			userExcludedSourcePaths: new Set([dir]),
		});
		const index = loadMirrorIndex(root);

		expect(result.skipped).toBe(2);
		expect(index.entries["/docs/private/a.txt"]).toBeUndefined();
		expect(index.entries["/docs/private/b.txt"]).toBeUndefined();
		expect(index.skipped?.["/docs/private/a.txt"]?.reason).toBe("user-excluded");
		expect(index.entries["/docs/keep.txt"]).toBeDefined();
	});

	it("continues when a scanned source disappears while an earlier file is parsed", async () => {
		writeFileSync(join(source, "a.txt"), "Alpha\n");
		writeFileSync(join(source, "b.txt"), "Beta\n");
		writeFileSync(join(source, "c.txt"), "Gamma\n");

		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: new ParserRegistry([new DeleteNextFileParser()]),
		});
		const index = loadMirrorIndex(root);
		const diag = result.diagnostics.find((d) => d.code === "source-vanished");

		expect(result.scanned).toBe(3);
		expect(result.written).toBe(2);
		expect(result.skipped).toBe(1);
		expect(diag?.severity).toBe("info");
		expect(diag?.source).toBe("/docs/b.txt");
		expect(index.entries["/docs/b.txt"]).toBeUndefined();
		expect(index.entries["/docs/c.txt"]).toBeDefined();
	});
});

/**
 * The staleness scan must not re-derive what refresh already decided. A file that
 * refresh deliberately skipped (excluded duplicate, oversize, unparseable) used to
 * be reported as a brand-new source forever, which made `lite retrieve` and
 * `autorag status` permanently report a corpus that never changed as stale.
 */
describe("mirror staleness decisions", () => {
	async function refreshWithDeliberateSkips(): Promise<void> {
		writeFileSync(join(source, "ok.txt"), "Fine\n");
		writeFileSync(join(source, "dup.txt"), "same content\n");
		writeFileSync(join(source, "dup-copy.txt"), "same content\n");
		writeFileSync(join(source, "huge.txt"), "x".repeat(1024));
		writeFileSync(join(source, "broken.hwp"), Buffer.from([1, 2, 3, 4]));
		writeFileSync(join(source, ".jikji_agent_map.md"), "# Jikji Agent Map\n");

		await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			maxSourceBytes: 100,
			excludeSourcePaths: new Set([join(source, "dup-copy.txt")]),
		});
	}

	/** Read the persisted index artifact itself, so the test pins what lands on disk. */
	function persistedSkipReasons(): Record<string, string> {
		const raw: unknown = JSON.parse(readFileSync(parsedMirrorIndexPath(root), "utf8"));
		const skipped = (raw as { skipped?: Record<string, { reason?: unknown }> }).skipped ?? {};
		return Object.fromEntries(
			Object.entries(skipped).map(([virtualPath, entry]) => [virtualPath, String(entry.reason)]),
		);
	}

	it("treats a refresh's deliberate skips as current instead of as new sources", async () => {
		await refreshWithDeliberateSkips();

		const diagnostics = await detectMirrorStaleness({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
		});

		expect(diagnostics).toEqual([]);
	});

	it("never treats Jikji's product artifact as a corpus source", async () => {
		writeFileSync(join(source, "ok.txt"), "Fine\n");
		writeFileSync(join(source, ".jikji_agent_map.md"), "# Jikji Agent Map\n");

		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
		});
		const index = loadMirrorIndex(root);

		expect(result.scanned).toBe(1);
		expect(index.entries["/docs/.jikji_agent_map.md"]).toBeUndefined();
		const diagnostics = await detectMirrorStaleness({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
		});
		expect(diagnostics).toEqual([]);
	});

	it("records each deliberate skip with its reason in the mirror index", async () => {
		await refreshWithDeliberateSkips();

		expect(persistedSkipReasons()).toEqual({
			"/docs/dup-copy.txt": "duplicate-excluded",
			"/docs/huge.txt": "parser-skipped",
			"/docs/broken.hwp": "parser-failed",
		});
	});

	it("reports a previously skipped source as stale again once it changes", async () => {
		await refreshWithDeliberateSkips();
		writeFileSync(join(source, "broken.hwp"), Buffer.from([9, 9, 9, 9, 9, 9]));

		const diagnostics = await detectMirrorStaleness({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
		});

		expect(diagnostics).toHaveLength(1);
		expect(diagnostics[0]).toMatchObject({
			code: "stale-index",
			source: "/docs/broken.hwp",
			reason: "mtime-and-size-changed",
		});
	});

	it("stops reporting a skipped source once a later refresh indexes it", async () => {
		const file = join(source, "big.txt");
		writeFileSync(file, "x".repeat(1024));
		await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			maxSourceBytes: 100,
		});
		expect(persistedSkipReasons()["/docs/big.txt"]).toBe("parser-skipped");

		writeFileSync(file, "fits now\n");
		await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			maxSourceBytes: 100,
		});

		expect(persistedSkipReasons()["/docs/big.txt"]).toBeUndefined();
		expect(loadMirrorIndex(root).entries["/docs/big.txt"]).toBeDefined();
	});

	it("ignores changes to a still-excluded source", async () => {
		const file = join(source, "private.txt");
		writeFileSync(file, "Private\n");
		await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			userExcludedSourcePaths: new Set([file]),
		});
		expect(persistedSkipReasons()["/docs/private.txt"]).toBe("user-excluded");

		writeFileSync(file, "Private v2 — longer content\n");
		const diagnostics = await detectMirrorStaleness({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			userExcludedSourcePaths: new Set([file]),
		});

		expect(diagnostics).toEqual([]);
	});

	it("reports a re-included source as newly added even when its skip record still matches", async () => {
		const file = join(source, "private.txt");
		writeFileSync(file, "Private\n");
		await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			userExcludedSourcePaths: new Set([file]),
		});
		expect(persistedSkipReasons()["/docs/private.txt"]).toBe("user-excluded");

		// Exclusion removed, file untouched: the old skip record matches mtime/size
		// but must not mask the source — nothing indexes it until a refresh runs.
		const diagnostics = await detectMirrorStaleness({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
		});

		expect(diagnostics).toHaveLength(1);
		expect(diagnostics[0]).toMatchObject({
			code: "stale-index",
			source: "/docs/private.txt",
			reason: "source-added",
		});
	});
});

describe("syncParsedMirrors: isolation, abort and progress", () => {
	it("turns a parse past fileTimeoutMs into one parser-failed skip and finishes the rest of the corpus", async () => {
		// A kordoc stub cannot cross the worker boundary, so this exercises the real default registry
		// with a worker entry that spins on `spin.bin`-style names is covered in test/parser; here a
		// real deadline of 1ms makes any worker parse of a real document time out deterministically.
		writeFileSync(join(source, "slow.docx"), await createDocxFixture("slow"));
		writeFileSync(join(source, "plain.txt"), "Plain survives\n");

		const result = await syncParsedMirrors({ root, searchPaths: [source], parserOptions: { fileTimeoutMs: 1 } });
		const index = loadMirrorIndex(root);

		expect(result).toMatchObject({ scanned: 2, written: 1, skipped: 1 });
		expect(result.diagnostics.filter((d) => d.code === "parser-failed")).toEqual([
			expect.objectContaining({ source: "/docs/slow.docx", severity: "warning" }),
		]);
		expect(result.diagnostics.find((d) => d.code === "parser-failed")?.message).toContain("1ms");
		expect(index.skipped?.["/docs/slow.docx"]?.reason).toBe("parser-failed");
		expect(index.entries["/docs/plain.txt"]).toBeDefined();
	});

	it("parses documents through the isolated worker by default", async () => {
		writeFileSync(join(source, "doc.docx"), await createDocxFixture("Worker default marker"));

		const result = await syncParsedMirrors({ root, searchPaths: [source] });
		const outputPath = requireValue(loadMirrorIndex(root).entries["/docs/doc.docx"]?.outputPath, "docx output");

		expect(result).toMatchObject({ scanned: 1, written: 1, skipped: 0 });
		expect(readFileSync(outputPath, "utf8")).toContain("Worker default marker");
	});

	it("rejects a non-positive fileTimeoutMs or maxHeapMb", async () => {
		await expect(
			syncParsedMirrors({ root, searchPaths: [source], parserOptions: { fileTimeoutMs: 0 } }),
		).rejects.toThrow(/fileTimeoutMs/);
		await expect(
			syncParsedMirrors({ root, searchPaths: [source], parserOptions: { maxHeapMb: -1 } }),
		).rejects.toThrow(/maxHeapMb/);
	});

	it("reports advancing processed/total progress during the parse phase", async () => {
		for (const name of ["a", "b", "c"]) writeFileSync(join(source, `${name}.txt`), `${name}\n`);
		const events: { processed: number; total: number; currentVirtualPath?: string }[] = [];

		await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			onProgress: (progress) => events.push({ ...progress }),
		});

		expect(events.every((event) => event.total === 3)).toBe(true);
		expect(events.map((event) => event.processed)).toEqual([0, 1, 2, 3]);
		expect(events.slice(0, 3).map((event) => event.currentVirtualPath)).toEqual([
			"/docs/a.txt",
			"/docs/b.txt",
			"/docs/c.txt",
		]);
		expect(events[3]?.currentVirtualPath).toBeUndefined();
	});

	it("counts skipped, unchanged and excluded files as processed", async () => {
		writeFileSync(join(source, "a.txt"), "a\n");
		writeFileSync(join(source, "b.txt"), "b\n");
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const events: number[] = [];

		await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			userExcludedSourcePaths: new Set([join(source, "a.txt")]),
			onProgress: (progress) => events.push(progress.processed),
		});

		expect(events).toEqual([0, 1, 2]);
	});

	it("stops between files when the signal aborts, keeps finished work checkpointed, and rethrows the reason", async () => {
		for (const name of ["a", "b", "c"]) writeFileSync(join(source, `${name}.txt`), `${name}\n`);
		const controller = new AbortController();
		const reason = new Error("host cancelled");

		const run = syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			signal: controller.signal,
			onProgress: (progress) => {
				if (progress.processed === 1) controller.abort(reason);
			},
		});

		await expect(run).rejects.toBe(reason);
		const index = loadMirrorIndex(root);
		expect(Object.keys(index.entries)).toEqual(["/docs/a.txt"]);
		expect(existsSync(requireValue(index.entries["/docs/a.txt"]?.outputPath, "a output"))).toBe(true);
	});

	it("rejects without touching the mirror when the signal is already aborted", async () => {
		writeFileSync(join(source, "a.txt"), "a\n");
		await expect(
			syncParsedMirrors({
				root,
				searchPaths: [source],
				registry: createDefaultParserRegistry(),
				signal: AbortSignal.abort(),
			}),
		).rejects.toMatchObject({ name: "AbortError" });
		expect(existsSync(parsedMirrorIndexPath(root))).toBe(false);
	});

	it("kills an in-flight isolated parse when the signal aborts", async () => {
		writeFileSync(join(source, "doc.docx"), await createDocxFixture("never finishes"));
		const controller = new AbortController();
		// Abort as soon as the file is announced, i.e. while its parse starts.
		const run = syncParsedMirrors({
			root,
			searchPaths: [source],
			signal: controller.signal,
			onProgress: (progress) => {
				if (progress.currentVirtualPath === "/docs/doc.docx") queueMicrotask(() => controller.abort());
			},
		});
		await expect(run).rejects.toMatchObject({ name: "AbortError" });
		expect(loadMirrorIndex(root).entries["/docs/doc.docx"]).toBeUndefined();
	});
});
