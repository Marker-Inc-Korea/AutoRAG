import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	isFilesystemAbsolutePath,
	normalizeSource,
	normalizeVirtualPath,
	planSourceRoots,
} from "../../src/filesystem/source-paths.ts";

describe("isFilesystemAbsolutePath", () => {
	it("accepts POSIX absolute paths", () => {
		expect(isFilesystemAbsolutePath("/Shared Docs/guide.md")).toBe(true);
		expect(isFilesystemAbsolutePath("/")).toBe(true);
	});

	it("accepts Windows drive absolute paths regardless of host OS", () => {
		expect(isFilesystemAbsolutePath("C:\\docs\\notes.md")).toBe(true);
		expect(isFilesystemAbsolutePath("D:/shared/file.txt")).toBe(true);
	});

	it("accepts Windows UNC absolute paths", () => {
		expect(isFilesystemAbsolutePath("\\\\server\\share\\file.txt")).toBe(true);
	});

	it("rejects relative and drive-relative paths", () => {
		expect(isFilesystemAbsolutePath("docs/notes.md")).toBe(false);
		expect(isFilesystemAbsolutePath("docs\\notes.md")).toBe(false);
		expect(isFilesystemAbsolutePath("C:partial")).toBe(false);
		expect(isFilesystemAbsolutePath("")).toBe(false);
	});
});

describe("normalizeSource", () => {
	let root: string | undefined;

	afterEach(() => {
		if (root) rmSync(root, { recursive: true, force: true });
		root = undefined;
	});

	function makeRoots() {
		root = mkdtempSync(join(tmpdir(), "autorag-normalize-source-"));
		const docsDir = join(root, "docs");
		const sharedDir = join(root, "shared");
		return { roots: planSourceRoots([docsDir, sharedDir]), docsDir, sharedDir };
	}

	it("converts an absolute real path under a source root to the canonical virtual id", () => {
		const { roots, docsDir } = makeRoots();
		expect(normalizeSource(join(docsDir, "policy", "refund.md"), roots)).toBe("/docs/policy/refund.md");
	});

	it("converts the root path itself to the root prefix", () => {
		const { roots, sharedDir } = makeRoots();
		expect(normalizeSource(sharedDir, roots)).toBe("/shared");
	});

	it("prefers the longest containing root for nested roots", () => {
		root = mkdtempSync(join(tmpdir(), "autorag-normalize-source-"));
		const outer = join(root, "docs");
		const inner = join(root, "docs", "team");
		const roots = planSourceRoots([outer, inner]);
		expect(normalizeSource(join(inner, "note.md"), roots)).toBe("/team/note.md");
	});

	it("passes an already-virtual id through validation unchanged", () => {
		const { roots } = makeRoots();
		expect(normalizeSource("/docs/policy/refund.md", roots)).toBe("/docs/policy/refund.md");
	});

	it("passes a datasource slash identity through validation unchanged", () => {
		const { roots } = makeRoots();
		expect(normalizeSource("/kakao/default/chunks/chunk-1", roots)).toBe("/kakao/default/chunks/chunk-1");
	});

	it("collapses duplicate separators in virtual and datasource ids", () => {
		const { roots } = makeRoots();
		expect(normalizeSource("/kakao//default//chunks/chunk-1", roots)).toBe("/kakao/default/chunks/chunk-1");
	});

	it("keeps file names that contain '#' or '?' as canonical virtual ids", () => {
		const { roots, docsDir } = makeRoots();
		// Real file names are data, not URL syntax: a receipt named
		// `... #2832-1476.txt` must keep its canonical id so parsed mirrors stay
		// indexable end to end (issue #1617).
		expect(normalizeVirtualPath("/docs/receipt #2832-1476.txt")).toBe("/docs/receipt #2832-1476.txt");
		expect(normalizeVirtualPath("/docs/what ? query.txt")).toBe("/docs/what ? query.txt");
		expect(normalizeSource(join(docsDir, "receipt #2832-1476.txt"), roots)).toBe("/docs/receipt #2832-1476.txt");
		expect(normalizeSource(join(docsDir, "what ? query.txt"), roots)).toBe("/docs/what ? query.txt");
		expect(normalizeSource("/docs/receipt #2832-1476.txt", roots)).toBe("/docs/receipt #2832-1476.txt");
		// URL-shaped sources are still rejected: the ban targeted syntax, not names.
		expect(normalizeVirtualPath("file:///docs/a.md")).toBeUndefined();
	});

	it("passes an absolute filesystem path outside every source root through as its own canonical form", () => {
		const { roots } = makeRoots();
		const outside = join(tmpdir(), "outside-file.md");
		// Canonical form uses forward slashes on every host.
		expect(normalizeSource(outside, roots)).toBe(outside.replaceAll("\\", "/"));
	});

	it("canonicalizes Windows drive and UNC absolute paths to forward-slash form", () => {
		const { roots } = makeRoots();
		expect(normalizeSource("C:\\outside\\file.md", roots)).toBe("C:/outside/file.md");
		expect(normalizeSource("D:/shared/file.txt", roots)).toBe("D:/shared/file.txt");
		expect(normalizeSource("\\\\server\\share\\file.txt", roots)).toBe("/server/share/file.txt");
	});

	it("fails closed for traversal, scheme, backslash, and empty sources", () => {
		const { roots } = makeRoots();
		expect(normalizeSource("/docs/../secret.md", roots)).toBeUndefined();
		expect(normalizeSource("kakao:chat/sender/chunk-1", roots)).toBeUndefined();
		expect(normalizeSource("file:///docs/a.md", roots)).toBeUndefined();
		expect(normalizeSource("docs\\a.md", roots)).toBeUndefined();
		expect(normalizeSource("", roots)).toBeUndefined();
	});
});
