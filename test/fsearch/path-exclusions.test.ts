import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createPathExclusionMatcher } from "../../src/filesystem/path-exclusions.ts";
import { FSearchClient } from "../../src/fsearch/client.ts";

describe("path exclusion matching", () => {
	it("matches POSIX files and subtrees without excluding sibling prefixes or changing case", () => {
		const isExcluded = createPathExclusionMatcher(["/docs/private", "/docs/secret.txt"], "linux");
		for (const path of ["/docs/private", "/docs/private/nested/note.txt", "/docs/secret.txt"]) {
			expect(isExcluded(path)).toBe(true);
		}
		for (const path of ["/docs/private-copy/note.txt", "/docs/secret.txt.bak", "/docs/Private/note.txt"]) {
			expect(isExcluded(path)).toBe(false);
		}
	});

	it("normalizes POSIX dot segments and trailing separators", () => {
		const isExcluded = createPathExclusionMatcher(["/docs/unused/../private/"], "darwin");
		expect(isExcluded("/docs/private")).toBe(true);
		expect(isExcluded("/docs/./private//note.txt")).toBe(true);
		expect(isExcluded("/docs/private-copy")).toBe(false);
	});

	it("matches a POSIX filesystem root", () => {
		const isExcluded = createPathExclusionMatcher(["/"], "linux");
		expect(isExcluded("/")).toBe(true);
		expect(isExcluded("/docs/note.txt")).toBe(true);
	});

	it("matches Windows paths case-insensitively across separator spellings", () => {
		const isExcluded = createPathExclusionMatcher(["C:\\Docs\\Private\\", "C:\\Docs\\secret.txt"], "win32");
		for (const path of ["c:/docs/private", "c:/DOCS/private/note.txt", "C:\\docs\\SECRET.TXT"]) {
			expect(isExcluded(path)).toBe(true);
		}
		for (const path of ["C:/docs/private-copy/note.txt", "C:/docs/secret.txt.bak", "D:/docs/private/note.txt"]) {
			expect(isExcluded(path)).toBe(false);
		}
	});

	it("handles Windows drive and UNC share roots without crossing volumes", () => {
		const drive = createPathExclusionMatcher(["C:\\"], "win32");
		expect(drive("c:/")).toBe(true);
		expect(drive("c:/docs/note.txt")).toBe(true);
		expect(drive("d:/docs/note.txt")).toBe(false);
		const share = createPathExclusionMatcher(["\\\\server\\Private\\"], "win32");
		expect(share("//SERVER/private")).toBe(true);
		expect(share("//server/private/docs/note.txt")).toBe(true);
		expect(share("//server/private-copy/docs/note.txt")).toBe(false);
	});

	it("leaves paths searchable when no exclusions are configured", () => {
		expect(createPathExclusionMatcher([], "linux")("/docs/note.txt")).toBe(false);
		expect(createPathExclusionMatcher([], "win32")("C:\\docs\\note.txt")).toBe(false);
	});
});

// The native runner is injected; these boundary checks run on Windows too.
describe("FSearchClient path exclusions", () => {
	let root: string;
	let docs: string;

	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-fsearch-boundary-"));
		docs = join(root, "docs");
		const dbPath = join(root, ".autorag", "fsearch", "fsearch.db");
		mkdirSync(dirname(dbPath), { recursive: true });
		writeFileSync(dbPath, "injected backend placeholder");
	});

	afterEach(() => {
		rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
	});

	function nativeClient(paths: readonly string[], excludeFolders?: readonly string[]) {
		const searches: string[][] = [];
		const client = new FSearchClient({
			root,
			folders: [docs],
			platform: "linux",
			excludeFolders,
			run: async (_command, args) => {
				if (args[0] === "--version") return { code: 0, stdout: "fsearch-cli 0.3", stderr: "" };
				if (args[0] !== "search") throw new Error(`Unexpected native command: ${args[0]}`);
				searches.push([...args]);
				const limit = Number(args[args.indexOf("--limit") + 1]);
				return {
					code: 0,
					stdout: [
						...paths.slice(0, limit).map((path) => JSON.stringify({ path, type: "file" })),
						JSON.stringify({ done: true, num_results: paths.length }),
					].join("\n"),
					stderr: "",
				};
			},
		});
		return { client, searches };
	}

	it("filters exact files and stale subtree hits while retaining sibling prefixes", async () => {
		const excludedFile = join(docs, "private.txt");
		const excludedFolder = join(docs, "archive");
		const kept = [join(docs, "private.txt.bak"), join(docs, "archive-copy", "report.txt")];
		const { client } = nativeClient(
			[excludedFile, excludedFolder, join(excludedFolder, "nested", "report.txt"), ...kept],
			[excludedFile, excludedFolder],
		);
		const result = await client.search({ query: "" });
		expect(result.ok).toBe(true);
		if (!result.ok) throw new Error(result.message);
		expect(result.backend).toBe("fsearch-cli");
		expect(result.results.map((entry) => entry.path)).toEqual(kept);
		expect(result.total).toBe(5);
		expect(result.truncated).toBeUndefined();
	});

	it("excludes native hits before applying the requested offset and result limit", async () => {
		const excluded = join(docs, "private.txt");
		const allowed = ["first.txt", "second.txt", "third.txt"].map((name) => join(docs, name));
		const { client, searches } = nativeClient([excluded, ...allowed], [excluded]);
		const result = await client.search({ query: ".txt", offset: 1, maxResults: 1 });
		expect(result.ok).toBe(true);
		if (!result.ok) throw new Error(result.message);
		expect(result.results.map((entry) => entry.path)).toEqual([allowed[1]]);
		expect(searches).toHaveLength(1);
		expect(searches[0].slice(searches[0].indexOf("--limit"))).toEqual(["--limit", "1000", "--", ".txt"]);
	});

	it("applies query path filtering together with exclusions before pagination", async () => {
		const scope = join(docs, "team");
		const excluded = join(scope, "private.txt");
		const wanted = join(scope, "second.txt");
		const { client } = nativeClient(
			[join(docs, "other.txt"), excluded, join(scope, "first.txt"), wanted],
			[excluded],
		);
		const result = await client.search({ query: ".txt", path: scope, offset: 1, maxResults: 1 });
		expect(result.ok).toBe(true);
		if (!result.ok) throw new Error(result.message);
		expect(result.results.map((entry) => entry.path)).toEqual([wanted]);
	});

	it.each([
		{ name: "omitted", exclusions: undefined },
		{ name: "empty", exclusions: [] },
	])("keeps the existing bounded native request when exclusions are $name", async ({ exclusions }) => {
		const paths = Array.from({ length: 10 }, (_, index) => join(docs, `${index}.txt`));
		const { client, searches } = nativeClient(paths, exclusions);
		const result = await client.search({ query: ".txt", offset: 2, maxResults: 3 });
		expect(result.ok).toBe(true);
		if (!result.ok) throw new Error(result.message);
		expect(result.results.map((entry) => entry.path)).toEqual(paths.slice(2, 5));
		expect(searches[0].slice(searches[0].indexOf("--limit"))).toEqual(["--limit", "5", "--", ".txt"]);
		expect(result.truncated).toBeUndefined();
	});

	it("caps the native candidate pool at 1000 and reports truncated exclusion-filtered results", async () => {
		const excluded = join(docs, "private.txt");
		const allowed = Array.from({ length: 1005 }, (_, index) => join(docs, `public-${index}.txt`));
		const { client, searches } = nativeClient([excluded, ...allowed], [excluded]);
		const result = await client.search({ query: ".txt", maxResults: 2000 });
		expect(result.ok).toBe(true);
		if (!result.ok) throw new Error(result.message);
		expect(result.results.map((entry) => entry.path)).toEqual(allowed.slice(0, 999));
		expect(result.total).toBe(1006);
		expect(result.truncated).toBe(true);
		expect(result.note).toEqual(expect.any(String));
		expect(searches).toHaveLength(1);
		expect(searches[0][searches[0].indexOf("--limit") + 1]).toBe("1000");
	});

	it("does not report truncation when the returned page is full despite more native matches", async () => {
		const excluded = join(docs, "private.txt");
		const allowed = Array.from({ length: 1500 }, (_, index) => join(docs, `public-${index}.txt`));
		const { client } = nativeClient([excluded, ...allowed], [excluded]);
		const result = await client.search({ query: ".txt", maxResults: 100 });
		expect(result.ok).toBe(true);
		if (!result.ok) throw new Error(result.message);
		expect(result.results.map((entry) => entry.path)).toEqual(allowed.slice(0, 100));
		expect(result.truncated).toBeUndefined();
		expect(result.note).toBeUndefined();
	});

	it("does not report truncation when the native pool exactly fits the cap", async () => {
		const excluded = join(docs, "private.txt");
		const paths = [excluded, ...Array.from({ length: 999 }, (_, index) => join(docs, `${index}.txt`))];
		const { client } = nativeClient(paths, [excluded]);
		const result = await client.search({ query: ".txt", maxResults: 1000 });
		expect(result.ok).toBe(true);
		if (!result.ok) throw new Error(result.message);
		expect(result.results).toHaveLength(999);
		expect(result.truncated).toBeUndefined();
		expect(result.note).toBeUndefined();
	});

	it("keeps exclusions consistent between native results and the missing-binary fallback", async () => {
		const excluded = [join(docs, "private.txt"), join(docs, "archive")];
		const kept = [join(docs, "archive-copy", "report.txt"), join(docs, "public.txt")];
		const paths = [excluded[0], join(excluded[1], "report.txt"), ...kept];
		for (const path of paths) {
			mkdirSync(dirname(path), { recursive: true });
			writeFileSync(path, "fixture");
		}
		const { client } = nativeClient([...paths].sort(), excluded);
		const fallback = new FSearchClient({
			root,
			folders: [docs],
			platform: "linux",
			excludeFolders: excluded,
			run: async () => ({ code: 127, stdout: "", stderr: "fsearch-cli unavailable" }),
		});
		const request = { query: ".txt", kind: "files", sort: "path-ascending" } as const;
		const nativeResult = await client.search(request);
		const walkResult = await fallback.search(request);
		expect(nativeResult.ok).toBe(true);
		expect(walkResult.ok).toBe(true);
		if (!nativeResult.ok || !walkResult.ok) throw new Error("Expected both retrieval backends to answer");
		expect(walkResult.backend).toBe("walk");
		expect(nativeResult.results.map((entry) => entry.path)).toEqual(kept);
		expect(walkResult.results.map((entry) => entry.path)).toEqual(kept);
	});
});
