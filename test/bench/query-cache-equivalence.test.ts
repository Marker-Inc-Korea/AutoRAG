import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

/**
 * Correctness gate for the query-time reuse of derived index state.
 *
 * Reusing work is only legitimate if it is invisible: the same query against the same artifact must
 * return the same results, and the reuse must disappear the moment the artifact changes. A cache
 * that survives a rebuild would be indistinguishable from a speedup obtained by answering from stale
 * data, so both properties are asserted rather than assumed.
 */

const roots: string[] = [];
const ENGINES = ["typescript-fallback", "tantivy"] as const;

const QUERIES = [
	"refund approval policy",
	"chargeback evidence retention",
	"escalation threshold",
	"quarterly settlement report",
	"document 3 paragraph 7",
	"nonexistent term xyzzy",
	"",
];

function makeWorkspace(documentCount: number, marker = "baseline"): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-qc-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		const body = Array.from(
			{ length: 10 },
			(_, p) =>
				`Paragraph ${p} of document ${i}. Refund approval policy, chargeback evidence retention, escalation threshold, quarterly settlement report. Marker ${marker}.`,
		).join("\n\n");
		writeFileSync(join(docs, `doc-${String(i).padStart(4, "0")}.md`), `# Document ${i}\n\n${body}\n`);
	}
	return { root, searchPaths: [docs] };
}

/** Full comparable shape of a result set: identity, order, and score. */
function shape(results: readonly RetrievalResult[]): string {
	return JSON.stringify(results.map((r) => [r.id, r.source, r.score]));
}

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe.each(ENGINES)("query-time reuse (%s)", (engine) => {
	it("returns identical results on repeated queries", async () => {
		const { root, searchPaths } = makeWorkspace(30);
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: engine });
		await method.sync();

		for (const query of QUERIES) {
			const first = await method.retrieve(query, { topK: 10 });
			const second = await method.retrieve(query, { topK: 10 });
			const third = await method.retrieve(query, { topK: 10 });
			expect(shape(second)).toBe(shape(first));
			expect(shape(third)).toBe(shape(first));
		}
	});

	it("matches a cold instance that has no derived state", async () => {
		const { root, searchPaths } = makeWorkspace(30);
		await syncParsedMirrors({ root, searchPaths });
		const warm = new BM25Method({ root, forceEngine: engine });
		await warm.sync();

		for (const query of QUERIES) {
			// Warm the derived state, then compare against an instance built from scratch. Any
			// divergence would mean the reused state is not equivalent to recomputing it.
			await warm.retrieve(query, { topK: 10 });
			const warmResults = await warm.retrieve(query, { topK: 10 });

			const cold = new BM25Method({ root, forceEngine: engine });
			await cold.sync();
			const coldResults = await cold.retrieve(query, { topK: 10 });

			expect(shape(warmResults)).toBe(shape(coldResults));
		}
	});

	it("preserves scope semantics", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-qc-"));
		roots.push(root);
		const alpha = join(root, "alpha");
		const beta = join(root, "beta");
		mkdirSync(alpha, { recursive: true });
		mkdirSync(beta, { recursive: true });
		for (let i = 0; i < 12; i += 1) {
			writeFileSync(
				join(alpha, `a-${i}.md`),
				`# Alpha ${i}\n\nRefund approval policy and escalation threshold review.\n`,
			);
			writeFileSync(
				join(beta, `b-${i}.md`),
				`# Beta ${i}\n\nRefund approval policy and settlement reconciliation.\n`,
			);
		}
		await syncParsedMirrors({ root, searchPaths: [alpha, beta] });
		const method = new BM25Method({ root, forceEngine: engine });
		await method.sync();

		for (const scope of [undefined, "alpha", "beta"]) {
			const first = await method.retrieve("refund approval policy", { topK: 10, ...(scope ? { scope } : {}) });
			const second = await method.retrieve("refund approval policy", { topK: 10, ...(scope ? { scope } : {}) });
			expect(shape(second)).toBe(shape(first));

			const cold = new BM25Method({ root, forceEngine: engine });
			await cold.sync();
			const coldResults = await cold.retrieve("refund approval policy", { topK: 10, ...(scope ? { scope } : {}) });
			expect(shape(first)).toBe(shape(coldResults));
		}

		// Scoped statistics really are scoped: the two scopes must not return each other's documents.
		const alphaHits = await method.retrieve("escalation threshold", { topK: 10, scope: "alpha" });
		expect(alphaHits.length).toBeGreaterThan(0);
		expect(alphaHits.every((hit) => hit.source.includes("/alpha/"))).toBe(true);
		expect(alphaHits.some((hit) => hit.source.includes("/beta/"))).toBe(false);
	});

	it("never serves stale results after the corpus changes", async () => {
		const { root, searchPaths } = makeWorkspace(10, "original");
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: engine });
		await method.sync();

		// Warm the derived state against the original corpus.
		const before = await method.retrieve("original", { topK: 10 });
		expect(before.length).toBeGreaterThan(0);
		expect(await method.retrieve("mutated", { topK: 10 })).toHaveLength(0);

		// Replace every document, reindex, and query the same instance again.
		const docs = searchPaths[0] as string;
		for (let i = 0; i < 10; i += 1) {
			writeFileSync(
				join(docs, `doc-${String(i).padStart(4, "0")}.md`),
				`# Document ${i}\n\nRefund approval policy. Marker mutated.\n`,
			);
		}
		await syncParsedMirrors({ root, searchPaths });
		await method.sync();

		// If derived state survived the rebuild, the old term would still match and the new one would not.
		expect(await method.retrieve("original", { topK: 10 })).toHaveLength(0);
		expect((await method.retrieve("mutated", { topK: 10 })).length).toBeGreaterThan(0);
	});
	it("never serves stale results while the fingerprint sidecar is missing", async () => {
		const { root, searchPaths } = makeWorkspace(6, "alpha");
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: engine });
		await method.sync();

		// Warm the derived state against the original artifact.
		expect((await method.retrieve("alpha", { topK: 10 })).length).toBeGreaterThan(0);

		// Delete the sidecar and replace the artifact with different content, the way a writer
		// that records no fingerprint (old version, failed sidecar write) would.
		const fingerprintPath = join(root, ".autorag", "bm25", "index-fingerprint.json");
		rmSync(fingerprintPath);
		const docs = searchPaths[0] as string;
		for (let i = 0; i < 6; i += 1) {
			writeFileSync(
				join(docs, `doc-${String(i).padStart(4, "0")}.md`),
				`# Document ${i}\n\nRefund approval policy. Marker beta.\n`,
			);
		}
		await syncParsedMirrors({ root, searchPaths });
		const writer = new BM25Method({ root, forceEngine: engine });
		await writer.sync();
		rmSync(fingerprintPath);

		// The warm instance observes the replacement...
		expect(await method.retrieve("alpha", { topK: 10 })).toHaveLength(0);
		expect((await method.retrieve("beta", { topK: 10 })).length).toBeGreaterThan(0);

		// ...and a second replacement while the sidecar stays missing must be visible too.
		for (let i = 0; i < 6; i += 1) {
			writeFileSync(
				join(docs, `doc-${String(i).padStart(4, "0")}.md`),
				`# Document ${i}\n\nEscalation threshold. Marker gamma.\n`,
			);
		}
		await syncParsedMirrors({ root, searchPaths });
		const secondWriter = new BM25Method({ root, forceEngine: engine });
		await secondWriter.sync();
		rmSync(fingerprintPath);

		expect(await method.retrieve("beta", { topK: 10 })).toHaveLength(0);
		expect((await method.retrieve("gamma", { topK: 10 })).length).toBeGreaterThan(0);
	});

	it("picks up a rebuild performed by a different instance", async () => {
		const { root, searchPaths } = makeWorkspace(8, "original");
		await syncParsedMirrors({ root, searchPaths });
		const reader = new BM25Method({ root, forceEngine: engine });
		await reader.sync();
		expect((await reader.retrieve("original", { topK: 10 })).length).toBeGreaterThan(0);

		// A separate instance rebuilds the artifact; the reader holds warm derived state and must
		// still observe the new content, because invalidation keys on the on-disk fingerprint.
		const docs = searchPaths[0] as string;
		for (let i = 0; i < 8; i += 1) {
			writeFileSync(
				join(docs, `doc-${String(i).padStart(4, "0")}.md`),
				`# Document ${i}\n\nRefund approval policy. Marker mutated.\n`,
			);
		}
		await syncParsedMirrors({ root, searchPaths });
		const writer = new BM25Method({ root, forceEngine: engine });
		await writer.sync();

		expect(await reader.retrieve("original", { topK: 10 })).toHaveLength(0);
		expect((await reader.retrieve("mutated", { topK: 10 })).length).toBeGreaterThan(0);
	});
});
