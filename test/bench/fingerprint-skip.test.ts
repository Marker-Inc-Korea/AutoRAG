import { createHash } from "node:crypto";
import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	rmSync,
	statSync,
	utimesSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadMirrorIndex } from "../../src/mirror/index-store.ts";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";

/**
 * Proof harness for the BM25 fingerprint skip.
 *
 * Wall-clock timing cannot show *why* an unchanged refresh became cheaper, so these tests establish
 * the mechanism directly: an unchanged rebuild must not open a single parsed mirror document, and
 * must not rewrite the built artifact.
 */

const roots: string[] = [];

function makeWorkspace(documentCount: number): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-fp-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		const body = Array.from(
			{ length: 12 },
			(_, p) => `Paragraph ${p} of document ${i} about refund approval policy.`,
		).join("\n\n");
		writeFileSync(join(docs, `doc-${String(i).padStart(4, "0")}.md`), `# Document ${i}\n\n${body}\n`);
	}
	return { root, searchPaths: [docs] };
}

function mirrorFiles(root: string): string[] {
	// Mirrors are written to .autorag/parsed/files/<sha256>.md; the sibling index.json is metadata.
	const filesDir = join(root, ".autorag", "parsed", "files");
	if (!existsSync(filesDir)) return [];
	return readdirSync(filesDir).map((name) => join(filesDir, name));
}

/**
 * Run `fn` with every parsed mirror document unreadable, then restore permissions.
 *
 * This is the load-bearing measurement. Instrumenting `fs` does not work here because the modules
 * under test bind `readFileSync` at import time, so a patched property is never consulted. Denying
 * read permission proves the property directly: a sync that completes while the mirror documents
 * cannot be opened provably did not open them.
 */
async function withUnreadableMirrors(root: string, fn: () => Promise<void>): Promise<void> {
	const files = mirrorFiles(root);
	for (const file of files) chmodSync(file, 0o000);
	try {
		await fn();
	} finally {
		for (const file of files) chmodSync(file, 0o644);
	}
}

/** sha256 of the built artifact, or undefined when it does not exist. */
function artifactDigest(root: string): string | undefined {
	const path = join(root, ".autorag", "bm25", "fallback-index.json");
	if (!existsSync(path)) return undefined;
	return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function bm25(root: string): BM25Method {
	// Pin the pure-TypeScript engine so the measurement does not depend on the native binding.
	return new BM25Method({ root, forceEngine: "typescript-fallback" });
}

afterEach(() => {
	for (const root of roots.splice(0)) {
		try {
			for (const file of mirrorFiles(root)) chmodSync(file, 0o644);
		} catch {
			// Workspace may already be gone or never built; cleanup below still applies.
		}
		rmSync(root, { recursive: true, force: true });
	}
});

describe("BM25 fingerprint skip", () => {
	it("completes an unchanged rebuild without opening any mirror document", async () => {
		const { root, searchPaths } = makeWorkspace(40);
		await syncParsedMirrors({ root, searchPaths });
		const method = bm25(root);

		const first = await method.sync();
		expect(first.skipped).toBeUndefined();
		expect(first.indexedChunks).toBeGreaterThan(0);

		// If the skip really avoids mirror reads, revoking read permission changes nothing.
		await withUnreadableMirrors(root, async () => {
			const second = await method.sync();
			expect(second.skipped).toBe(true);
			expect(second.indexedChunks).toBe(first.indexedChunks);
		});
	});

	it("proves the first build does need those same documents", async () => {
		const { root, searchPaths } = makeWorkspace(8);
		await syncParsedMirrors({ root, searchPaths });
		const method = bm25(root);

		// Control for the test above. With no stored fingerprint there is nothing to skip, so the
		// build must reach for the mirror documents and cannot get past the denied read. The skip
		// path succeeding under the exact same permissions is therefore evidence about the skip, not
		// an artifact of the corpus being trivially cheap.
		await withUnreadableMirrors(root, async () => {
			await expect(method.sync()).rejects.toThrow(/EACCES|permission denied/i);
		});
	});

	it("does not rewrite the built artifact on a skipped rebuild", async () => {
		const { root, searchPaths } = makeWorkspace(12);
		await syncParsedMirrors({ root, searchPaths });
		const method = bm25(root);
		await method.sync();

		const digestBefore = artifactDigest(root);
		const mtimeBefore = statSync(join(root, ".autorag", "bm25", "fallback-index.json")).mtimeMs;
		expect(digestBefore).toBeDefined();

		const skipped = await method.sync();
		expect(skipped.skipped).toBe(true);
		expect(artifactDigest(root)).toBe(digestBefore);
		expect(statSync(join(root, ".autorag", "bm25", "fallback-index.json")).mtimeMs).toBe(mtimeBefore);
	});

	it("still answers queries identically after a skipped rebuild", async () => {
		const { root, searchPaths } = makeWorkspace(10);
		await syncParsedMirrors({ root, searchPaths });
		const method = bm25(root);
		await method.sync();
		const before = await method.retrieve("refund approval policy", { topK: 5 });

		expect((await method.sync()).skipped).toBe(true);
		const after = await method.retrieve("refund approval policy", { topK: 5 });

		expect(after.length).toBeGreaterThan(0);
		expect(after.map((r) => r.id)).toEqual(before.map((r) => r.id));
	});

	it("rebuilds when a document changes", async () => {
		const { root, searchPaths } = makeWorkspace(5);
		await syncParsedMirrors({ root, searchPaths });
		const method = bm25(root);
		await method.sync();
		expect((await method.sync()).skipped).toBe(true);

		writeFileSync(
			join(searchPaths[0] as string, "doc-0000.md"),
			"# Changed\n\nEntirely different chargeback content.\n",
		);
		await syncParsedMirrors({ root, searchPaths });

		expect((await method.sync()).skipped).toBeUndefined();
		expect((await method.retrieve("chargeback", { topK: 5 })).length).toBeGreaterThan(0);
	});

	it("rebuilds when content changes but size and mtime are restored", async () => {
		const { root, searchPaths } = makeWorkspace(3);
		const target = join(searchPaths[0] as string, "doc-0000.md");
		const originalBytes = readFileSync(target);
		const originalStat = statSync(target);

		await syncParsedMirrors({ root, searchPaths });
		const method = bm25(root);
		await method.sync();
		expect((await method.sync()).skipped).toBe(true);

		// Same byte length, restored mtime: size+mtime alone cannot tell these apart.
		const swapped = Buffer.from(originalBytes);
		swapped.fill(0x7a, swapped.length - 8, swapped.length - 1);
		writeFileSync(target, swapped);
		utimesSync(target, originalStat.atime, originalStat.mtime);

		await syncParsedMirrors({ root, searchPaths, force: true });
		expect((await method.sync()).skipped).toBeUndefined();
	});

	it("opens no mirror documents regardless of corpus size", async () => {
		const small = makeWorkspace(10);
		const large = makeWorkspace(80);
		await syncParsedMirrors({ root: small.root, searchPaths: small.searchPaths });
		await syncParsedMirrors({ root: large.root, searchPaths: large.searchPaths });

		const smallMethod = bm25(small.root);
		const largeMethod = bm25(large.root);
		await smallMethod.sync();
		await largeMethod.sync();

		// Eight times the corpus, and neither skip touches a mirror document.
		await withUnreadableMirrors(small.root, async () => {
			expect((await smallMethod.sync()).skipped).toBe(true);
		});
		await withUnreadableMirrors(large.root, async () => {
			expect((await largeMethod.sync()).skipped).toBe(true);
		});
	});

	it("backfills a legacy index that has no content digests", async () => {
		const { root, searchPaths } = makeWorkspace(6);
		await syncParsedMirrors({ root, searchPaths });

		// Simulate an index written before contentSha256 existed.
		const indexPath = join(root, ".autorag", "parsed", "index.json");
		const current = JSON.parse(readFileSync(indexPath, "utf8")) as {
			version: 1;
			entries: Record<string, Record<string, unknown>>;
		};
		for (const entry of Object.values(current.entries)) delete entry.contentSha256;
		writeFileSync(indexPath, JSON.stringify(current, null, 2));

		expect(Object.values(loadMirrorIndex(root).entries).every((e) => e.contentSha256 === undefined)).toBe(true);

		await syncParsedMirrors({ root, searchPaths });
		expect(Object.values(loadMirrorIndex(root).entries).every((e) => typeof e.contentSha256 === "string")).toBe(true);

		// The backfill is one-time: a further sync needs no mirror reads at all.
		await withUnreadableMirrors(root, async () => {
			await syncParsedMirrors({ root, searchPaths });
		});
		expect(Object.values(loadMirrorIndex(root).entries).every((e) => typeof e.contentSha256 === "string")).toBe(true);
	});
});
