import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";

/**
 * Recovery when the on-disk index state is damaged.
 *
 * Skipping a rebuild means trusting an artifact without opening it, so every way that trust can be
 * misplaced needs a defined outcome. The rule these tests pin is simple: when anything about the
 * stored state is missing or does not match what was recorded, rebuild rather than skip.
 */

const roots: string[] = [];

function makeWorkspace(documentCount = 6): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-integrity-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		writeFileSync(join(docs, `doc-${i}.md`), `# Document ${i}\n\nRefund approval policy and escalation threshold.\n`);
	}
	return { root, searchPaths: [docs] };
}

function paths(root: string): { fingerprint: string; artifact: string } {
	return {
		fingerprint: join(root, ".autorag", "bm25", "index-fingerprint.json"),
		artifact: join(root, ".autorag", "bm25", "fallback-index.json"),
	};
}

async function buildFallback(root: string, searchPaths: string[]): Promise<BM25Method> {
	await syncParsedMirrors({ root, searchPaths });
	const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
	await method.sync();
	return method;
}

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("artifact integrity", () => {
	it("rebuilds when only the fingerprint sidecar is deleted", async () => {
		const { root, searchPaths } = makeWorkspace();
		const method = await buildFallback(root, searchPaths);
		expect((await method.sync()).skipped).toBe(true);

		rmSync(paths(root).fingerprint);

		const rebuilt = await method.sync();
		expect(rebuilt.skipped).toBeUndefined();
		expect(rebuilt.indexedChunks).toBeGreaterThan(0);
		expect((await method.retrieve("refund", { topK: 5 })).length).toBeGreaterThan(0);
	});

	it("rebuilds when only the artifact is deleted", async () => {
		const { root, searchPaths } = makeWorkspace();
		const method = await buildFallback(root, searchPaths);
		expect((await method.sync()).skipped).toBe(true);

		rmSync(paths(root).artifact);

		const rebuilt = await method.sync();
		expect(rebuilt.skipped).toBeUndefined();
		expect((await method.retrieve("refund", { topK: 5 })).length).toBeGreaterThan(0);
	});

	it("rebuilds when the artifact is truncated", async () => {
		const { root, searchPaths } = makeWorkspace();
		const method = await buildFallback(root, searchPaths);
		expect((await method.sync()).skipped).toBe(true);

		// Truncation leaves a file that exists but cannot be parsed. Existence alone would wave it
		// through and the failure would only surface at query time.
		const artifact = paths(root).artifact;
		const original = readFileSync(artifact, "utf8");
		writeFileSync(artifact, original.slice(0, Math.floor(original.length / 2)));

		const rebuilt = await method.sync();
		expect(rebuilt.skipped).toBeUndefined();
		expect((await method.retrieve("refund", { topK: 5 })).length).toBeGreaterThan(0);
	});

	it("rebuilds when the fingerprint sidecar is corrupt", async () => {
		const { root, searchPaths } = makeWorkspace();
		const method = await buildFallback(root, searchPaths);
		expect((await method.sync()).skipped).toBe(true);

		writeFileSync(paths(root).fingerprint, "{ not valid json");

		const rebuilt = await method.sync();
		expect(rebuilt.skipped).toBeUndefined();
		expect((await method.retrieve("refund", { topK: 5 })).length).toBeGreaterThan(0);
	});

	it("rebuilds when the recorded engine differs from the requested one", async () => {
		const { root, searchPaths } = makeWorkspace();
		await buildFallback(root, searchPaths);

		// A different engine means a different artifact, so the stored fingerprint must not be
		// honoured even though the mirror content is unchanged.
		const tantivy = new BM25Method({ root, forceEngine: "tantivy" });
		const rebuilt = await tantivy.sync();
		expect(rebuilt.skipped).toBeUndefined();
	});
	it("never leaves a stale fingerprint beside a freshly built artifact", async () => {
		const { root, searchPaths } = makeWorkspace();
		const method = await buildFallback(root, searchPaths);
		expect((await method.sync()).skipped).toBe(true);

		// Sabotage the fingerprint path: replace the sidecar with a directory so the rename that
		// commits a new fingerprint is refused (rename over a directory fails). A fingerprint
		// write failure is a commit failure, not an engine failure, so the rebuild fails loudly
		// instead of rewriting the artifact and silently retrying the fingerprint.
		rmSync(paths(root).fingerprint);
		mkdirSync(paths(root).fingerprint);

		writeFileSync(
			join(searchPaths[0] as string, "doc-0.md"),
			"# Changed\n\nEntirely different chargeback content.\n",
		);
		await syncParsedMirrors({ root, searchPaths });

		await expect(method.sync()).rejects.toThrow();
		// The failed commit removed the sabotage and left no sidecar behind, so no stale identity
		// can claim the fresh artifact; the next sync retries and records the real fingerprint.
		expect(existsSync(paths(root).fingerprint)).toBe(false);

		const rebuilt = await method.sync();
		expect(rebuilt.skipped).toBeUndefined();
		// A real sidecar now describes the artifact.
		expect(statSync(paths(root).fingerprint).isFile()).toBe(true);
		expect((await method.retrieve("chargeback", { topK: 5 })).length).toBeGreaterThan(0);
		// The fingerprint now matches the corpus: the next sync skips instead of rebuilding.
		expect((await method.sync()).skipped).toBe(true);
	});

	it("does not commit a new artifact into a read-only index directory", async () => {
		const { root, searchPaths } = makeWorkspace();
		const method = await buildFallback(root, searchPaths);
		expect((await method.sync()).skipped).toBe(true);

		const { fingerprint, artifact } = paths(root);
		const artifactBefore = readFileSync(artifact);
		const fingerprintBefore = readFileSync(fingerprint, "utf8");

		// A fresh corpus forces a rebuild. With the directory read-only the rebuild must fail
		// instead of overwriting the existing artifact in place and leaving the old fingerprint to
		// claim a new artifact it was never computed for.
		writeFileSync(
			join(searchPaths[0] as string, "doc-0.md"),
			"# Changed\n\nEntirely different chargeback content.\n",
		);
		await syncParsedMirrors({ root, searchPaths });
		chmodSync(join(root, ".autorag", "bm25"), 0o555);
		try {
			await expect(method.sync()).rejects.toThrow(/EACCES|permission denied/i);
		} finally {
			chmodSync(join(root, ".autorag", "bm25"), 0o755);
		}
		// Neither the artifact nor the fingerprint moved: no half-built state was committed.
		expect(readFileSync(artifact)).toEqual(artifactBefore);
		expect(readFileSync(fingerprint, "utf8")).toBe(fingerprintBefore);

		// Once writable again, the same refresh completes and serves the new corpus.
		const rebuilt = await method.sync();
		expect(rebuilt.skipped).toBeUndefined();
		expect((await method.retrieve("chargeback", { topK: 5 })).length).toBeGreaterThan(0);
	});
});
