import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

/**
 * Transaction guarantees for `refresh()`.
 *
 * The fingerprint skip makes interleaved refreshes dangerous: if one run commits the BM25 artifact
 * while another commits the fingerprint, the fingerprint can end up describing a newer mirror than
 * the artifact it points at, and the next refresh then matches the fingerprint and keeps the stale
 * artifact without any error. These tests pin the guarantees that make that impossible.
 */

const roots: string[] = [];

function makeWorkspace(documentCount: number, marker = "alpha"): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-lock-"));
	roots.push(root);
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		writeFileSync(join(docs, `doc-${i}.md`), `# Document ${i}\n\nRefund approval policy. Marker ${marker}.\n`);
	}
	return { root, searchPaths: [docs] };
}

function makeAgent(root: string, searchPaths: string[]): AutoRAGAgent {
	return new AutoRAGAgent({
		// `workspacePath` is what the agent uses as its project root; passing anything else silently
		// falls back to process.cwd() and writes indexes into the current repository.
		workspacePath: root,
		searchPaths,
		bm25: { forceEngine: "typescript-fallback" },
		minSync: false,
	});
}

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("refresh transaction lock", () => {
	it("runs the pipeline once for concurrent calls with the same parameters", async () => {
		const { root, searchPaths } = makeWorkspace(6);
		const agent = makeAgent(root, searchPaths);

		// Count actual index builds by watching how often the parsed stage does work.
		let builds = 0;
		const original = agent.syncParsedMirrors.bind(agent);
		(agent as unknown as { syncParsedMirrors: typeof agent.syncParsedMirrors }).syncParsedMirrors = async (
			force?: boolean,
		) => {
			builds += 1;
			return original(force);
		};

		const [first, second] = await Promise.all([agent.refresh(true), agent.refresh(true)]);

		expect(builds).toBe(1);
		expect(first.outcome).toBe("completed");
		expect(second.outcome).toBe("completed");
	});

	it("gives joined callers the identical result object", async () => {
		const { root, searchPaths } = makeWorkspace(4);
		const agent = makeAgent(root, searchPaths);

		const [first, second] = await Promise.all([agent.refresh(true), agent.refresh(true)]);

		// Same run, so the same object: not merely equal values.
		expect(second).toBe(first);
	});

	it("refuses a concurrent call with different parameters and runs no downstream stage", async () => {
		const { root, searchPaths } = makeWorkspace(6);
		const agent = makeAgent(root, searchPaths);

		// Different `force` produces a different lock key, so the second call must be refused.
		const running = agent.refresh(true);
		const refused = await agent.refresh(false);

		expect(refused.outcome).toBe("busy");
		expect(refused.bm25).toBeUndefined();
		expect(refused.minsync).toBeUndefined();
		expect(refused.datasources).toEqual([]);
		expect(refused.scanned).toBe(0);
		expect(refused.written).toBe(0);
		expect(refused.diagnostics).toEqual([]);

		const completed = await running;
		expect(completed.outcome).toBe("completed");
	});

	it("keeps artifact and fingerprint consistent under repeated interleaving attempts", async () => {
		const { root, searchPaths } = makeWorkspace(5, "original");
		const agent = makeAgent(root, searchPaths);
		await agent.refresh(true);

		const fingerprintPath = join(root, ".autorag", "bm25", "index-fingerprint.json");
		const artifactPath = join(root, ".autorag", "bm25", "fallback-index.json");

		// Twenty rounds of racing refreshes against a mutating corpus. After each round the stored
		// fingerprint must describe the artifact that is actually on disk, which is checked by
		// asking for a fresh refresh and confirming it does not skip a stale artifact: the indexed
		// content must always contain the marker currently in the corpus.
		for (let round = 0; round < 20; round += 1) {
			const marker = `round${round}`;
			for (let i = 0; i < 5; i += 1) {
				writeFileSync(
					join(searchPaths[0] as string, `doc-${i}.md`),
					`# Document ${i}\n\nRefund approval policy. Marker ${marker}.\n`,
				);
			}
			await Promise.all([agent.refresh(false), agent.refresh(false), agent.refresh(true)]);

			// Settle any refusal by running one uncontended refresh.
			await agent.refresh(false);

			const artifact = readFileSync(artifactPath, "utf8");
			expect(artifact).toContain(marker);
			// A fingerprint must exist for the artifact that is present.
			expect(() => readFileSync(fingerprintPath, "utf8")).not.toThrow();

			const hits = await agent.searchAllDocuments(marker, { topK: 5 });
			expect(hits.results.length).toBeGreaterThan(0);
		}
	});

	it("releases the lock when a refresh throws", async () => {
		const { root, searchPaths } = makeWorkspace(3);
		const agent = makeAgent(root, searchPaths);

		const original = agent.syncParsedMirrors.bind(agent);
		let shouldFail = true;
		(agent as unknown as { syncParsedMirrors: typeof agent.syncParsedMirrors }).syncParsedMirrors = async (
			force?: boolean,
		) => {
			if (shouldFail) {
				shouldFail = false;
				throw new Error("injected failure");
			}
			return original(force);
		};

		await expect(agent.refresh(true)).rejects.toThrow("injected failure");

		// The lock must not be wedged by the rejection.
		const after = await agent.refresh(true);
		expect(after.outcome).toBe("completed");
	});

	it("still serves a sequential refresh normally", async () => {
		const { root, searchPaths } = makeWorkspace(4);
		const agent = makeAgent(root, searchPaths);

		const first = await agent.refresh(true);
		const second = await agent.refresh(false);

		expect(first.outcome).toBe("completed");
		expect(second.outcome).toBe("completed");
		expect(second.bm25?.skipped).toBe(true);
	});
});
