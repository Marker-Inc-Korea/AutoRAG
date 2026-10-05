import {
	chmodSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type FileNameSearchResult, searchFileNames } from "../../src/filesystem/name-search.ts";

const resultPaths = (result: FileNameSearchResult): string[] => result.results.map((match) => match.path);

describe("searchFileNames", () => {
	let base: string;
	let rootA: string;
	let rootB: string;
	let rootP: string;
	let rootOrder: string;
	let rootLock: string;
	let rootRead: string;
	let docs: string;
	let docsOther: string;

	function writeFile(parent: string, name: string, content = name): string {
		const target = join(parent, name);
		writeFileSync(target, content);
		return target;
	}

	beforeEach(() => {
		base = mkdtempSync(join(realpathSync(tmpdir()), "name-search-"));

		rootA = join(base, "rootA");
		mkdirSync(join(rootA, "nested", "deep"), { recursive: true });
		mkdirSync(join(rootA, ".git"), { recursive: true });
		mkdirSync(join(rootA, ".autorag"), { recursive: true });
		mkdirSync(join(rootA, ".jikji"), { recursive: true });
		mkdirSync(join(rootA, "node_modules", "pkg"), { recursive: true });
		writeFile(rootA, "alpha.txt", "alpha content");
		writeFile(rootA, "Beta.md", "beta content");
		writeFile(rootA, "axb.txt", "literal dot content");
		writeFile(join(rootA, "nested"), "gamma-alpha.txt");
		writeFile(join(rootA, "nested", "deep"), "delta-alpha.bin");
		writeFile(join(rootA, ".git"), "config.txt");
		writeFile(join(rootA, ".autorag"), "cache.txt");
		writeFile(join(rootA, ".jikji"), "map.txt");
		writeFile(join(rootA, "node_modules", "pkg"), "alpha.js");

		rootB = join(base, "rootB");
		mkdirSync(rootB, { recursive: true });
		writeFile(rootB, "beta-root.txt");
		writeFile(rootB, "omega.txt");

		rootP = join(base, "rootP");
		mkdirSync(rootP, { recursive: true });
		for (let index = 0; index < 5; index += 1) writeFile(rootP, `page-${index}.txt`);

		rootOrder = join(base, "rootOrder");
		mkdirSync(rootOrder, { recursive: true });
		writeFile(rootOrder, "c.txt");
		writeFile(rootOrder, "a.txt");
		writeFile(rootOrder, "b.txt");

		rootLock = join(base, "rootLock");
		mkdirSync(join(rootLock, "locked"), { recursive: true });
		writeFile(rootLock, "visible.txt");

		rootRead = join(base, "rootRead");
		mkdirSync(rootRead, { recursive: true });
		const unreadable = writeFile(rootRead, "unreadable.txt", "SENTINEL\n");
		chmodSync(unreadable, 0o000);

		const outside = join(base, "outside");
		mkdirSync(outside, { recursive: true });
		writeFile(outside, "escape-secret.txt", "outside content");
		symlinkSync(outside, join(rootA, "link-out"));
		symlinkSync(join(outside, "escape-secret.txt"), join(rootA, "link-file"));

		docs = join(base, "docs");
		docsOther = join(base, "docs-other");
		mkdirSync(docs, { recursive: true });
		mkdirSync(docsOther, { recursive: true });
		writeFile(docs, "in-docs.txt");
		writeFile(docsOther, "in-other.txt");
	});

	afterEach(() => {
		// Restore permissions so recursive removal works on every platform.
		try {
			chmodSync(join(rootLock, "locked"), 0o755);
		} catch {
			/* already restored or absent */
		}
		try {
			chmodSync(join(rootRead, "unreadable.txt"), 0o644);
		} catch {
			/* already restored or absent */
		}
		rmSync(base, { recursive: true, force: true });
	});

	it("matches literal file names across every configured root without mutating sources", async () => {
		const alphaBefore = readFileSync(join(rootA, "alpha.txt"), "utf8");
		const result = await searchFileNames([rootA, rootB], { query: "alpha" });

		expect(result.ok).toBe(true);
		expect(result.backend).toBe("filesystem");
		expect(result.truncated).toBe(false);
		expect(result.diagnostics).toEqual([]);
		const paths = resultPaths(result);
		expect(paths).toContain(join(rootA, "alpha.txt"));
		expect(paths).toContain(join(rootA, "nested", "gamma-alpha.txt"));
		expect(paths).toContain(join(rootA, "nested", "deep", "delta-alpha.bin"));
		expect(paths).not.toContain(join(rootA, "axb.txt"));
		expect(result.results.every((match) => match.type === "file")).toBe(true);
		expect(readFileSync(join(rootA, "alpha.txt"), "utf8")).toBe(alphaBefore);
	});

	it("treats the query as a literal substring rather than a regular expression", async () => {
		const dotted = await searchFileNames([rootA], { query: "a.b" });
		expect(resultPaths(dotted)).not.toContain(join(rootA, "axb.txt"));
		const anyChar = await searchFileNames([rootA], { query: "al.ha" });
		expect(resultPaths(anyChar)).toEqual([]);
		const literal = await searchFileNames([rootA], { query: "alpha.txt" });
		expect(resultPaths(literal)).toContain(join(rootA, "alpha.txt"));
	});

	it("is case-insensitive by default and case-sensitive when requested", async () => {
		const insensitive = await searchFileNames([rootA, rootB], { query: "beta" });
		const insensitivePaths = resultPaths(insensitive);
		expect(insensitivePaths).toContain(join(rootA, "Beta.md"));
		expect(insensitivePaths).toContain(join(rootB, "beta-root.txt"));

		const sensitive = await searchFileNames([rootA, rootB], { query: "beta", matchCase: true });
		const sensitivePaths = resultPaths(sensitive);
		expect(sensitivePaths).not.toContain(join(rootA, "Beta.md"));
		expect(sensitivePaths).toContain(join(rootB, "beta-root.txt"));
	});

	it("filters results by kind", async () => {
		const folders = await searchFileNames([rootA], { query: "nested", kind: "folders" });
		expect(folders.results).toEqual([{ path: join(rootA, "nested"), type: "folder" }]);

		const files = await searchFileNames([rootA], { query: "nested", kind: "files" });
		expect(files.results).toEqual([]);
	});

	it("matches the root-relative path only when matchPath is set", async () => {
		const scopedFiles = await searchFileNames([rootA], { query: "nested/deep", matchPath: true, kind: "files" });
		expect(resultPaths(scopedFiles)).toEqual([join(rootA, "nested", "deep", "delta-alpha.bin")]);

		const scopedFolders = await searchFileNames([rootA], { query: "nested/deep", matchPath: true, kind: "folders" });
		expect(scopedFolders.results).toEqual([{ path: join(rootA, "nested", "deep"), type: "folder" }]);

		const nameOnly = await searchFileNames([rootA], { query: "nested/deep" });
		expect(resultPaths(nameOnly)).toEqual([]);
	});

	it("limits discovery to a configured subdirectory via request.root", async () => {
		const result = await searchFileNames([rootA], { query: "alpha", root: "nested" });
		expect(resultPaths(result).sort()).toEqual(
			[join(rootA, "nested", "deep", "delta-alpha.bin"), join(rootA, "nested", "gamma-alpha.txt")].sort(),
		);
	});

	it("rejects a request.root outside every configured root", async () => {
		const outsideResult = await searchFileNames([rootA], { query: "escape", root: join(base, "outside") });
		expect(outsideResult.results).toEqual([]);
		expect(outsideResult.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(["root-out-of-scope"]);

		const escapingLink = await searchFileNames([rootA], { query: "alpha", root: join(rootA, "link-out") });
		expect(escapingLink.results).toEqual([]);
		expect(escapingLink.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(["root-out-of-scope"]);

		const prefixSibling = await searchFileNames([docs], { query: "in-other", root: docsOther });
		expect(prefixSibling.results).toEqual([]);
		expect(prefixSibling.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(["root-out-of-scope"]);

		const traversal = await searchFileNames([docs], { query: "in-other", root: "../docs-other" });
		expect(traversal.results).toEqual([]);
		expect(traversal.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(["root-out-of-scope"]);
	});

	it("never follows child symlinks out of a root", async () => {
		const byName = await searchFileNames([rootA], { query: "escape" });
		expect(byName.results).toEqual([]);
		const byContentName = await searchFileNames([rootA], { query: "secret" });
		expect(byContentName.results).toEqual([]);
		const linkEntries = await searchFileNames([rootA], { query: "link-" });
		expect(linkEntries.results).toEqual([]);
	});

	it("skips tool product directories", async () => {
		const result = await searchFileNames([rootA], { query: "" });
		const paths = resultPaths(result);
		for (const hidden of [".git", ".autorag", ".jikji", "node_modules"]) {
			expect(paths.some((path) => path.includes(`${hidden}/`))).toBe(false);
		}
		expect(resultPaths(await searchFileNames([rootA], { query: "config" }))).toEqual([]);
		expect(resultPaths(await searchFileNames([rootA], { query: "alpha.js" }))).toEqual([]);
		expect(resultPaths(await searchFileNames([rootA], { query: "cache" }))).toEqual([]);
		expect(resultPaths(await searchFileNames([rootA], { query: "map" }))).toEqual([]);
	});

	it("honors excluded file and directory paths", async () => {
		const excludedDir = await searchFileNames([rootA], { query: "alpha" }, [join(rootA, "nested")]);
		expect(resultPaths(excludedDir)).toEqual([join(rootA, "alpha.txt")]);

		const excludedFile = await searchFileNames([rootA], { query: "alpha" }, [join(rootA, "alpha.txt")]);
		const excludedPaths = resultPaths(excludedFile);
		expect(excludedPaths).not.toContain(join(rootA, "alpha.txt"));
		expect(excludedPaths).toContain(join(rootA, "nested", "gamma-alpha.txt"));
	});

	it("paginates with an accurate truncated flag", async () => {
		const first = await searchFileNames([rootP], { query: "page", maxResults: 2, offset: 0 });
		expect(resultPaths(first)).toEqual([join(rootP, "page-0.txt"), join(rootP, "page-1.txt")]);
		expect(first.truncated).toBe(true);

		const second = await searchFileNames([rootP], { query: "page", maxResults: 2, offset: 2 });
		expect(resultPaths(second)).toEqual([join(rootP, "page-2.txt"), join(rootP, "page-3.txt")]);
		expect(second.truncated).toBe(true);

		const last = await searchFileNames([rootP], { query: "page", maxResults: 2, offset: 4 });
		expect(resultPaths(last)).toEqual([join(rootP, "page-4.txt")]);
		expect(last.truncated).toBe(false);

		const past = await searchFileNames([rootP], { query: "page", maxResults: 2, offset: 6 });
		expect(past.results).toEqual([]);
		expect(past.truncated).toBe(false);

		const exact = await searchFileNames([rootP], { query: "page", maxResults: 5, offset: 0 });
		expect(exact.results).toHaveLength(5);
		expect(exact.truncated).toBe(false);
	});

	it("walks in a deterministic order", async () => {
		const first = await searchFileNames([rootOrder], { query: "txt" });
		const second = await searchFileNames([rootOrder], { query: "txt" });
		expect(resultPaths(first)).toEqual([
			join(rootOrder, "a.txt"),
			join(rootOrder, "b.txt"),
			join(rootOrder, "c.txt"),
		]);
		expect(resultPaths(second)).toEqual(resultPaths(first));
	});

	it("does not read source file contents", async () => {
		const result = await searchFileNames([rootRead], { query: "unreadable" });
		// The file is chmod 0o000, so discovering it proves no content read happened.
		expect(resultPaths(result)).toEqual([join(rootRead, "unreadable.txt")]);
		expect(result.diagnostics).toEqual([]);
		chmodSync(join(rootRead, "unreadable.txt"), 0o644);
		expect(readFileSync(join(rootRead, "unreadable.txt"), "utf8")).toBe("SENTINEL\n");
	});

	it("surfaces an unreadable directory as a diagnostic", async () => {
		// Windows ignores POSIX mode bits, so chmod 0o000 cannot revoke read access.
		if (process.platform === "win32") return;
		if (typeof process.getuid === "function" && process.getuid() === 0) return;
		chmodSync(join(rootLock, "locked"), 0o000);
		const result = await searchFileNames([rootLock], { query: "visible" });
		expect(resultPaths(result)).toContain(join(rootLock, "visible.txt"));
		expect(resultPaths(result).some((path) => path.includes("locked"))).toBe(false);
		const diagnostic = result.diagnostics.find((entry) => entry.source === join(rootLock, "locked"));
		expect(diagnostic?.code).toBe("permission-denied");
		expect(diagnostic?.severity).toBe("warning");
	});

	it("reports a missing configured root instead of throwing", async () => {
		const result = await searchFileNames([join(base, "does-not-exist")], { query: "anything" });
		expect(result.ok).toBe(true);
		expect(result.results).toEqual([]);
		expect(result.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(["root-unavailable"]);
	});

	it("returns an empty page when cancelled before it starts", async () => {
		const controller = new AbortController();
		controller.abort();
		const result = await searchFileNames([rootA], { query: "alpha", signal: controller.signal });
		expect(result.results).toEqual([]);
		expect(result.diagnostics.map((diagnostic) => diagnostic.code)).toContain("search-cancelled");
	});
});
