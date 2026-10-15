import {
	chmodSync,
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
	type ParsedMirrorSyncOptions,
	type ParsedMirrorSyncResult,
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
		writeFileSync(join(source, "present.txt"), "Still here\n");
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
		writeFileSync(join(source, "present.txt"), "Still here\n");
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

const canChmod = process.platform !== "win32" && process.getuid?.() !== 0;

function writeFiles(directory: string, count: number): void {
	mkdirSync(directory, { recursive: true });
	for (let index = 0; index < count; index += 1) {
		writeFileSync(join(directory, `f${String(index).padStart(3, "0")}.txt`), `File ${index}\n`);
	}
}

function diagnosticsWith(result: ParsedMirrorSyncResult, code: string): ParsedMirrorSyncResult["diagnostics"] {
	return result.diagnostics.filter((diagnostic) => diagnostic.code === code);
}

describe("syncParsedMirrors search-root availability", () => {
	async function indexOnce(searchPaths: string[] = [source]) {
		return syncParsedMirrors({ root, searchPaths, registry: createDefaultParserRegistry() });
	}

	it("keeps every mirror when a previously populated root is emptied", async () => {
		writeFileSync(join(source, "a.txt"), "Alpha\n");
		writeFileSync(join(source, "b.txt"), "Beta\n");
		await indexOnce();
		const before = loadMirrorIndex(root);

		unlinkSync(join(source, "a.txt"));
		unlinkSync(join(source, "b.txt"));
		const result = await indexOnce();

		expect(result.deleted).toBe(0);
		expect(diagnosticsWith(result, "deleted-mirror")).toEqual([]);
		const unavailable = diagnosticsWith(result, "search-root-unavailable");
		expect(unavailable).toHaveLength(1);
		expect(unavailable[0]).toMatchObject({ severity: "warning", source: "/docs", reason: "empty" });
		expect(JSON.stringify(unavailable)).not.toContain(root);
		const after = loadMirrorIndex(root);
		expect(Object.keys(after.entries).sort()).toEqual(["/docs/a.txt", "/docs/b.txt"]);
		for (const entry of Object.values(before.entries)) expect(existsSync(entry.outputPath)).toBe(true);
	});

	it("keeps every mirror when the root directory disappeared", async () => {
		writeFileSync(join(source, "a.txt"), "Alpha\n");
		await indexOnce();

		rmSync(source, { recursive: true, force: true });
		const result = await indexOnce();

		expect(result.deleted).toBe(0);
		expect(diagnosticsWith(result, "search-root-unavailable")[0]).toMatchObject({
			severity: "warning",
			source: "/docs",
			reason: "gone",
		});
		expect(Object.keys(loadMirrorIndex(root).entries)).toEqual(["/docs/a.txt"]);
	});

	it("keeps every mirror when the root path is no longer a directory", async () => {
		writeFileSync(join(source, "a.txt"), "Alpha\n");
		await indexOnce();

		rmSync(source, { recursive: true, force: true });
		writeFileSync(source, "now a file\n");
		const result = await indexOnce();

		expect(result.deleted).toBe(0);
		expect(diagnosticsWith(result, "search-root-unavailable")[0]).toMatchObject({ reason: "not-a-directory" });
		expect(Object.keys(loadMirrorIndex(root).entries)).toEqual(["/docs/a.txt"]);
	});

	it.skipIf(!canChmod)("keeps every mirror when the root cannot be listed (chmod 000 and 111)", async () => {
		writeFileSync(join(source, "a.txt"), "Alpha\n");
		await indexOnce();

		try {
			for (const mode of [0o000, 0o111]) {
				chmodSync(source, mode);
				const result = await indexOnce();
				expect(result.deleted).toBe(0);
				expect(diagnosticsWith(result, "deleted-mirror")).toEqual([]);
				expect(diagnosticsWith(result, "search-root-unavailable")[0]).toMatchObject({
					severity: "warning",
					source: "/docs",
					reason: "unreadable",
				});
				expect(Object.keys(loadMirrorIndex(root).entries)).toEqual(["/docs/a.txt"]);
			}
		} finally {
			chmodSync(source, 0o755);
		}
	});

	it("still removes genuinely stale mirrors in healthy roots while an unavailable root keeps its own", async () => {
		const other = join(root, "other");
		mkdirSync(other, { recursive: true });
		writeFileSync(join(source, "a.txt"), "Alpha\n");
		writeFileSync(join(other, "x.txt"), "Ex\n");
		writeFileSync(join(other, "y.txt"), "Why\n");
		await indexOnce([source, other]);

		unlinkSync(join(source, "a.txt"));
		unlinkSync(join(other, "x.txt"));
		unlinkSync(join(other, "y.txt"));
		writeFileSync(join(source, "b.txt"), "Beta\n");
		const result = await indexOnce([source, other]);

		expect(result.deleted).toBe(1);
		expect(diagnosticsWith(result, "deleted-mirror").map((d) => d.source)).toEqual(["/docs/a.txt"]);
		expect(diagnosticsWith(result, "search-root-unavailable").map((d) => d.source)).toEqual(["/other"]);
		expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual([
			"/docs/b.txt",
			"/other/x.txt",
			"/other/y.txt",
		]);
	});

	it("does not warn about a root that was never populated", async () => {
		const result = await indexOnce();

		expect(result.diagnostics).toEqual([]);
	});

	it("indexes the root again after it comes back and deletes nothing it still has", async () => {
		writeFileSync(join(source, "a.txt"), "Alpha\n");
		await indexOnce();
		unlinkSync(join(source, "a.txt"));
		await indexOnce();

		writeFileSync(join(source, "a.txt"), "Alpha\n");
		const result = await indexOnce();

		expect(result.deleted).toBe(0);
		expect(diagnosticsWith(result, "search-root-unavailable")).toEqual([]);
		expect(Object.keys(loadMirrorIndex(root).entries)).toEqual(["/docs/a.txt"]);
	});
});

describe("syncParsedMirrors deletion-ratio safety valve", () => {
	async function sync(extra: Partial<ParsedMirrorSyncOptions> = {}) {
		return syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry(), ...extra });
	}

	function removeFirst(count: number): void {
		for (let index = 0; index < count; index += 1) {
			unlinkSync(join(source, `f${String(index).padStart(3, "0")}.txt`));
		}
	}

	it("refuses to delete more than maxDeleteRatio above the 100-file floor", async () => {
		writeFiles(source, 150);
		await sync();

		removeFirst(120);
		const result = await sync();

		expect(result.deleted).toBe(0);
		expect(diagnosticsWith(result, "deleted-mirror")).toEqual([]);
		const suppressed = diagnosticsWith(result, "mass-delete-suppressed");
		expect(suppressed).toHaveLength(1);
		expect(suppressed[0]?.severity).toBe("warning");
		expect(suppressed[0]?.message).toContain("120");
		expect(Object.keys(loadMirrorIndex(root).entries)).toHaveLength(150);
		for (const entry of Object.values(loadMirrorIndex(root).entries)) expect(existsSync(entry.outputPath)).toBe(true);
	});

	it("keeps suppressing on the next refresh until force is set, then deletes", async () => {
		writeFiles(source, 150);
		await sync();
		removeFirst(120);
		await sync();

		const forced = await sync({ force: true });

		expect(forced.deleted).toBe(120);
		expect(diagnosticsWith(forced, "mass-delete-suppressed")).toEqual([]);
		expect(Object.keys(loadMirrorIndex(root).entries)).toHaveLength(30);
	});

	it("deletes at or below the ratio", async () => {
		writeFiles(source, 150);
		await sync();

		removeFirst(45);
		const result = await sync();

		expect(result.deleted).toBe(45);
		expect(diagnosticsWith(result, "mass-delete-suppressed")).toEqual([]);
		expect(Object.keys(loadMirrorIndex(root).entries)).toHaveLength(105);
	});

	it("honors a custom maxDeleteRatio", async () => {
		writeFiles(source, 150);
		await sync();

		removeFirst(120);
		const result = await sync({ maxDeleteRatio: 0.9 });

		expect(result.deleted).toBe(120);
		expect(diagnosticsWith(result, "mass-delete-suppressed")).toEqual([]);
	});

	it("never suppresses deletions at or below the 100-file floor", async () => {
		writeFiles(source, 100);
		await sync();

		removeFirst(99);
		const result = await sync();

		expect(result.deleted).toBe(99);
		expect(diagnosticsWith(result, "mass-delete-suppressed")).toEqual([]);
	});
});

describe("syncParsedMirrors walk hardening", () => {
	it("reports walk-truncated and keeps unseen mirrors when maxScannedEntries is hit", async () => {
		writeFiles(source, 20);
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });

		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			maxScannedEntries: 5,
		});

		expect(result.scanned).toBeLessThanOrEqual(5);
		expect(result.deleted).toBe(0);
		const truncated = diagnosticsWith(result, "walk-truncated");
		expect(truncated).toHaveLength(1);
		expect(truncated[0]).toMatchObject({ severity: "warning", source: "/docs", reason: "max-entries" });
		expect(JSON.stringify(truncated)).not.toContain(root);
		expect(Object.keys(loadMirrorIndex(root).entries)).toHaveLength(20);
	});

	it("reports walk-truncated and keeps deeper mirrors when maxDepth is hit", async () => {
		const deep = join(source, "a", "b", "c");
		mkdirSync(deep, { recursive: true });
		writeFileSync(join(source, "top.txt"), "Top\n");
		writeFileSync(join(source, "a", "mid.txt"), "Mid\n");
		writeFileSync(join(deep, "deep.txt"), "Deep\n");
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });

		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			maxDepth: 2,
		});

		expect(result.deleted).toBe(0);
		expect(diagnosticsWith(result, "walk-truncated")[0]).toMatchObject({
			severity: "warning",
			source: "/docs",
			reason: "max-depth",
		});
		expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual([
			"/docs/a/b/c/deep.txt",
			"/docs/a/mid.txt",
			"/docs/top.txt",
		]);
	});

	it("does not report truncation for a tree within the limits", async () => {
		writeFiles(source, 3);
		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: createDefaultParserRegistry(),
			maxScannedEntries: 3,
			maxDepth: 1,
		});

		expect(result.diagnostics).toEqual([]);
		expect(result.scanned).toBe(3);
	});

	it.skipIf(!canChmod)("reports directory-unreadable for one subdirectory while siblings still index", async () => {
		const locked = join(source, "locked");
		const open = join(source, "open");
		mkdirSync(locked, { recursive: true });
		mkdirSync(open, { recursive: true });
		writeFileSync(join(locked, "secret.txt"), "Secret\n");
		writeFileSync(join(open, "one.txt"), "One\n");
		await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });

		writeFileSync(join(open, "two.txt"), "Two\n");
		try {
			chmodSync(locked, 0o000);
			const result = await syncParsedMirrors({
				root,
				searchPaths: [source],
				registry: createDefaultParserRegistry(),
			});

			expect(result.deleted).toBe(0);
			expect(diagnosticsWith(result, "deleted-mirror")).toEqual([]);
			const unreadable = diagnosticsWith(result, "directory-unreadable");
			expect(unreadable).toHaveLength(1);
			expect(unreadable[0]).toMatchObject({ severity: "warning", source: "/docs/locked" });
			expect(JSON.stringify(unreadable)).not.toContain(root);
			expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual([
				"/docs/locked/secret.txt",
				"/docs/open/one.txt",
				"/docs/open/two.txt",
			]);
		} finally {
			chmodSync(locked, 0o755);
		}
	});
});
