import {
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
	loadMirrorIndex,
	type ParsedMirrorIndex,
	parsedMirrorRoot,
	saveMirrorIndex,
	syncParsedMirrors,
} from "../../src/mirror/index.ts";
import { createDefaultParserRegistry } from "../../src/parser/index.ts";

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

describe("syncParsedMirrors", () => {
	it("creates real markdown files and an index for supported virtual files", async () => {
		writeFileSync(join(source, "note.txt"), "Alpha\n");
		writeFileSync(join(source, "skip.bin"), Buffer.from([0, 1]));

		const result = await syncParsedMirrors({ root, searchPaths: [source], registry: createDefaultParserRegistry() });
		const index = loadMirrorIndex(root);
		const paths = outputPaths(index);

		expect(result).toMatchObject({ scanned: 2, written: 1, deleted: 0 });
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

	it("ignores poisoned index output paths when dropping unsupported current files", async () => {
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
		expect(result.skipped).toBe(1);
		expect(readFileSync(outside, "utf8")).toBe("do not delete unsupported\n");
		expect(index.entries["/docs/skip.bin"]).toBeUndefined();
	});
});
