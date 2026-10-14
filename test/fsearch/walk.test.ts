import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, parse, sep } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { FSearchClient } from "../../src/fsearch/client.ts";
import { walkFileSearch } from "../../src/fsearch/walk.ts";

// The fallback uses portable filesystem operations, even though the native
// fsearch backend is only exposed on macOS/Linux.
describe("fsearch walk exclusions", () => {
	let root: string;
	let docs: string;
	let excluded: string;
	let sibling: string;

	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-fsearch-exclusions-"));
		docs = join(root, "docs");
		excluded = join(docs, "archive");
		sibling = join(docs, "archive-copy");
		mkdirSync(join(excluded, "nested"), { recursive: true });
		mkdirSync(sibling, { recursive: true });
		writeFileSync(join(excluded, "hidden.txt"), "hidden");
		writeFileSync(join(excluded, "nested", "hidden.txt"), "hidden");
		writeFileSync(join(sibling, "kept.txt"), "kept");
		writeFileSync(join(docs, "kept.txt"), "kept");
		writeFileSync(join(docs, "private.txt"), "private");
	});

	afterEach(() => {
		rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
	});

	it("prunes excluded subtrees before they consume the walk budget", async () => {
		for (let index = 0; index < 20; index += 1) {
			writeFileSync(join(excluded, `hidden-${index}.txt`), "hidden");
		}
		const result = await walkFileSearch(
			[docs],
			{ query: "", kind: "files", sort: "path-ascending" },
			{ excludeFolders: [excluded, join(docs, "private.txt")], maxVisited: 6 },
		);
		expect(result.entries.map((entry) => entry.path)).toEqual([join(sibling, "kept.txt"), join(docs, "kept.txt")]);
		expect(result.visited).toBe(5);
		expect(result.truncated).toBe(false);
	});

	it("does not return excluded directories as folder matches", async () => {
		const result = await walkFileSearch(
			[docs],
			{ query: "archive", kind: "folders" },
			{ excludeFolders: [excluded] },
		);
		expect(result.entries.map((entry) => entry.path)).toEqual([sibling]);
	});

	it("neither visits nor returns an excluded search root or nested search root", async () => {
		const result = await walkFileSearch(
			[excluded, join(excluded, "nested")],
			{ query: "" },
			{ excludeFolders: [excluded] },
		);
		expect(result).toEqual({ entries: [], visited: 0, truncated: false });
	});

	it.each(["trailing separator", "filesystem root"])("handles an exclusion with a %s", async (kind) => {
		const exclusion = kind === "trailing separator" ? `${docs}${sep}` : parse(docs).root;
		const result = await walkFileSearch([docs], { query: "" }, { excludeFolders: [exclusion] });
		expect(result).toEqual({ entries: [], visited: 0, truncated: false });
	});

	it("does not let a query path filter restore excluded results", async () => {
		const result = await walkFileSearch([docs], { query: "hidden", path: excluded }, { excludeFolders: [excluded] });
		expect(result.entries).toEqual([]);
	});

	it("applies exclusions before sorting and pagination", async () => {
		const result = await walkFileSearch(
			[docs],
			{ query: ".txt", sort: "path-ascending", offset: 1, maxResults: 1 },
			{ excludeFolders: [excluded, join(docs, "private.txt")] },
		);
		expect(result.entries.map((entry) => entry.path)).toEqual([join(docs, "kept.txt")]);
	});

	it("keeps the existing walk behavior without configured exclusions", async () => {
		const result = await walkFileSearch([docs], { query: "hidden", kind: "files" });
		expect(result.entries.map((entry) => entry.path).sort()).toEqual(
			[join(excluded, "hidden.txt"), join(excluded, "nested", "hidden.txt")].sort(),
		);
	});

	it.each(["missing binary", "missing database"])("passes exclusions through the %s fallback", async (reason) => {
		const recorded: string[][] = [];
		const client = new FSearchClient({
			root,
			folders: [docs, excluded],
			excludeFolders: [excluded, join(docs, "private.txt")],
			platform: "linux",
			run: async (_command, args) => {
				recorded.push([...args]);
				return reason === "missing binary"
					? { code: null, stdout: "", stderr: "spawn fsearch-cli ENOENT" }
					: { code: 0, stdout: "fsearch-cli 0.1.0", stderr: "" };
			},
		});
		const result = await client.search({ query: "", kind: "files", sort: "path-ascending" });
		expect(result).toMatchObject({ ok: true, backend: "walk" });
		if (!result.ok) throw new Error(result.message);
		expect(result.results.map((entry) => entry.path)).toEqual([join(sibling, "kept.txt"), join(docs, "kept.txt")]);
		expect(recorded).toEqual([["--version"]]);
		expect(existsSync(join(root, ".autorag", "fsearch", "fsearch.db"))).toBe(false);
	});
});
