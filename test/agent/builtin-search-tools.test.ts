import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createBuiltinSearchTools, FIND_MAX_RESULTS } from "../../src/agent/builtin-search-tools.ts";
import { planSourceRoots, resolveVirtualSource } from "../../src/filesystem/source-paths.ts";

let root: string;
let docs: string;
let docs2: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-builtin-tools-"));
	docs = join(root, "docs");
	docs2 = join(root, "docs-2");
	mkdirSync(join(docs, "sub"), { recursive: true });
	mkdirSync(docs2, { recursive: true });
	writeFileSync(join(docs, "alpha.txt"), "alpha refund policy\nsecond line\n", "utf8");
	writeFileSync(join(docs, "sub", "beta.md"), "beta onboarding\n", "utf8");
	writeFileSync(join(docs2, "secret.txt"), "secret docs two\n", "utf8");
	writeFileSync(join(docs, "bin.dat"), Buffer.from([0, 1, 2, 3]));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function textOf(result: unknown): string {
	const first = (result as { content?: unknown[] }).content?.[0] as { text?: unknown } | undefined;
	return typeof first?.text === "string" ? first.text : "";
}

function tool(name: string) {
	const found = createBuiltinSearchTools({ root, searchPaths: [docs, docs2] }).find(
		(candidate) => candidate.name === name,
	);
	expect(found).toBeDefined();
	return found!;
}

describe("built-in search tools", () => {
	it("resolves opaque virtual paths with exact-prefix boundaries", () => {
		const roots = planSourceRoots([docs, docs2]);
		expect(resolveVirtualSource("/docs/alpha.txt", roots)?.sourceId).toBe("/docs/alpha.txt");
		expect(resolveVirtualSource("/docs-2/secret.txt", roots)?.sourceId).toBe("/docs-2/secret.txt");
		expect(resolveVirtualSource("/docs-2/secret.txt", roots)?.realPath).toContain("docs-2");
		expect(resolveVirtualSource("/docs-2/secret.txt", roots)?.realPath).not.toBe(
			resolveVirtualSource("/docs/secret.txt", roots)?.realPath,
		);
	});

	it("rejects traversal, URLs, fragments, queries, and symlink escapes", async () => {
		const outside = join(root, "outside.txt");
		writeFileSync(outside, "outside\n", "utf8");
		symlinkSync(outside, join(docs, "escape.txt"));
		const roots = planSourceRoots([docs]);
		for (const source of [
			"docs/alpha.txt",
			"/docs/../outside.txt",
			"file:///docs/alpha.txt",
			"/docs/alpha.txt#x",
			"/docs/alpha.txt?q=1",
			"/docs/escape.txt",
		]) {
			expect(resolveVirtualSource(source, roots)).toBeUndefined();
		}
		const response = await tool("read").execute("call", { source: "/docs/escape.txt" } as never);
		expect(JSON.stringify(response)).not.toContain(root);
		expect(textOf(response)).toContain("out of scope");
	});

	it("grep/find/read/ls/stat expose opaque sources and no real paths", async () => {
		const grep = await tool("grep").execute("call", { pattern: "refund" } as never);
		const find = await tool("find").execute("call", { query: "*.txt" } as never);
		const read = await tool("read").execute("call", { source: "/docs/alpha.txt", maxLines: 1 } as never);
		const ls = await tool("ls").execute("call", { source: "/docs" } as never);
		const stat = await tool("stat").execute("call", { source: "/docs/alpha.txt" } as never);
		const scopedGrep = await tool("grep").execute("call", { pattern: "onboarding", scope: "/docs" } as never);
		expect(JSON.stringify(scopedGrep)).toContain("/docs/sub/beta.md");
		const serialized = JSON.stringify([grep, find, read, ls, stat]);
		expect(serialized).toContain("/docs/alpha.txt");
		expect(serialized).not.toContain(root);
		expect(serialized).not.toContain(docs);
		expect(textOf(read)).toContain("alpha refund policy");
		expect(textOf(read)).not.toContain("second line");
	});

	it("caps find results and read bytes/lines", async () => {
		for (let i = 0; i < FIND_MAX_RESULTS + 5; i++) {
			writeFileSync(join(docs, `many-${i}.txt`), `row ${i}\n`, "utf8");
		}
		const find = await tool("find").execute("call", { query: "*many-*.txt" } as never);
		expect((find.details as { resultCount: number }).resultCount).toBe(FIND_MAX_RESULTS);
		expect((find.details as { truncated: boolean }).truncated).toBe(true);
		const read = await tool("read").execute("call", { source: "/docs/alpha.txt", maxBytes: 5 } as never);
		writeFileSync(
			join(docs, "long.txt"),
			`${Array.from({ length: 250 }, (_, i) => `line ${i}`).join("\n")}\n`,
			"utf8",
		);
		const oversizedRead = await tool("read").execute("call", {
			source: "/docs/long.txt",
			maxLines: 1_000_000,
			maxBytes: 1_000_000_000,
		} as never);
		expect((oversizedRead.details as { lines: number }).lines).toBe(200);
		expect((oversizedRead.details as { truncated: boolean }).truncated).toBe(true);
		expect((read.details as { truncated: boolean }).truncated).toBe(true);
	});

	it("reports binary files without leaking paths", async () => {
		const response = await tool("read").execute("call", { source: "/docs/bin.dat" } as never);
		expect(textOf(response)).toContain("binary");
		expect(JSON.stringify(response)).not.toContain(root);
	});
});
