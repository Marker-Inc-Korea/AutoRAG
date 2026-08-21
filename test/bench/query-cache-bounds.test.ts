import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

/**
 * Bounds and cross-process invalidation for the query-time cache.
 *
 * The cache is an optimisation, so it must be droppable at any moment without changing an answer,
 * and it must not survive a rebuild performed by a process this one knows nothing about.
 */

const roots: string[] = [];
const REPO_ROOT = join(import.meta.dirname, "..", "..");

function makeWorkspace(documentCount: number, marker = "alpha"): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-bound-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		writeFileSync(
			join(docs, `doc-${i}.md`),
			`# Document ${i}\n\nRefund approval policy and escalation threshold. Marker ${marker}.\n`,
		);
	}
	return { root, searchPaths: [docs] };
}

function shape(results: readonly RetrievalResult[]): string {
	return JSON.stringify(results.map((r) => [r.id, r.source, r.score]));
}

/** Read the private cache slot without exporting it from the production surface. */
function cacheSlot(method: BM25Method): unknown {
	return (method as unknown as { fallbackCache: unknown }).fallbackCache;
}

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("query cache bounds", () => {
	it("returns identical results whether or not the cache is retained", async () => {
		const { root, searchPaths } = makeWorkspace(20);
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await method.sync();

		const queries = ["refund approval policy", "escalation threshold", "alpha", "missingterm"];
		const cached = new Map<string, string>();
		for (const q of queries) {
			await method.retrieve(q, { topK: 10 });
			cached.set(q, shape(await method.retrieve(q, { topK: 10 })));
		}
		expect(cacheSlot(method)).toBeDefined();

		// Drop the cache before every query to emulate the over-bound path, where the prepared
		// structure is built for the query and then discarded.
		for (const q of queries) {
			(method as unknown as { fallbackCache: unknown }).fallbackCache = undefined;
			const uncached = shape(await method.retrieve(q, { topK: 10 }));
			expect(uncached).toBe(cached.get(q));
		}
	});

	it("keeps the cache for a corpus below the retention bound", async () => {
		const { root, searchPaths } = makeWorkspace(20);
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await method.sync();

		await method.retrieve("refund", { topK: 5 });
		expect(cacheSlot(method)).toBeDefined();
	});

	it("drops the cache when the retained-byte estimate exceeds the bound", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-bound-"));
		roots.push(root);
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		// ~200k short tokens: a quarter of the old token bound, but content strings plus term
		// objects plus postings exceed the byte budget, so the cache must not be retained.
		const words: string[] = [];
		for (let i = 0; i < 200_000; i += 1) {
			words.push(`term${String(i % 400).padStart(3, "0")}${["able", "ing", "tion"][i % 3]}`);
		}
		const body = words.join(" ");
		const paragraphs: string[] = [];
		for (let i = 0; i < body.length; i += 1800) paragraphs.push(body.slice(i, i + 1800));
		writeFileSync(join(docs, "doc-0.md"), `# Doc\n\n${paragraphs.join("\n\n")}\n`);

		await syncParsedMirrors({ root, searchPaths: [docs] });
		const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await method.sync();

		await method.retrieve("term000", { topK: 5 });
		expect(cacheSlot(method)).toBeUndefined();
	});

	it("never serves stale results while the fingerprint sidecar is missing", async () => {
		const { root, searchPaths } = makeWorkspace(6, "alpha");
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await method.sync();

		// Warm the cache against the original artifact.
		expect((await method.retrieve("alpha", { topK: 10 })).length).toBeGreaterThan(0);
		expect(cacheSlot(method)).toBeDefined();

		// Delete the sidecar and replace the artifact with different content, the way a writer
		// that records no fingerprint (old version, failed sidecar write) would.
		const fingerprintPath = join(root, ".autorag", "bm25", "index-fingerprint.json");
		rmSync(fingerprintPath);
		const docs = searchPaths[0] as string;
		for (let i = 0; i < 6; i += 1) {
			writeFileSync(join(docs, `doc-${i}.md`), `# Document ${i}\n\nRefund approval policy. Marker beta.\n`);
		}
		await syncParsedMirrors({ root, searchPaths });
		const writer = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await writer.sync();
		rmSync(fingerprintPath);

		// The warm instance observes the replacement and must not pin an unkeyed cache entry.
		expect(await method.retrieve("alpha", { topK: 10 })).toHaveLength(0);
		expect((await method.retrieve("beta", { topK: 10 })).length).toBeGreaterThan(0);
		expect(cacheSlot(method)).toBeUndefined();

		// A second replacement while the sidecar stays missing must be visible too.
		for (let i = 0; i < 6; i += 1) {
			writeFileSync(join(docs, `doc-${i}.md`), `# Document ${i}\n\nEscalation threshold. Marker gamma.\n`);
		}
		await syncParsedMirrors({ root, searchPaths });
		const secondWriter = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await secondWriter.sync();
		rmSync(fingerprintPath);

		expect(await method.retrieve("beta", { topK: 10 })).toHaveLength(0);
		expect((await method.retrieve("gamma", { topK: 10 })).length).toBeGreaterThan(0);
	});
	it("observes a rebuild performed by a separate operating system process", async () => {
		const { root, searchPaths } = makeWorkspace(6, "original");
		await syncParsedMirrors({ root, searchPaths });
		const reader = new BM25Method({ root, forceEngine: "typescript-fallback" });
		await reader.sync();

		// Warm the cache against the original corpus.
		expect((await reader.retrieve("original", { topK: 10 })).length).toBeGreaterThan(0);
		expect(cacheSlot(reader)).toBeDefined();

		// A genuinely separate process mutates the corpus and rebuilds the index. This process is
		// never told about it, so only the on-disk fingerprint can reveal the change.
		const script = `
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { syncParsedMirrors } from ${JSON.stringify(join(REPO_ROOT, "src/mirror/sync.ts"))};
import { BM25Method } from ${JSON.stringify(join(REPO_ROOT, "src/retrieval/methods/bm25.ts"))};
const root = ${JSON.stringify(root)};
const docs = ${JSON.stringify(searchPaths[0])};
for (let i = 0; i < 6; i += 1) {
	writeFileSync(join(docs, "doc-" + i + ".md"), "# Document " + i + "\\n\\nRefund approval policy. Marker mutated.\\n");
}
await syncParsedMirrors({ root, searchPaths: [docs] });
const writer = new BM25Method({ root, forceEngine: "typescript-fallback" });
await writer.sync();
`;
		const scriptPath = join(root, "rebuild.ts");
		writeFileSync(scriptPath, script);
		execFileSync("bun", ["run", scriptPath], { cwd: REPO_ROOT, stdio: "pipe" });

		// The warm reader must not answer from the superseded index.
		expect(await reader.retrieve("original", { topK: 10 })).toHaveLength(0);
		expect((await reader.retrieve("mutated", { topK: 10 })).length).toBeGreaterThan(0);
	});
});
