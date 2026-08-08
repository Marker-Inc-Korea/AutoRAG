import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";

/**
 * How many queries a rebuild must be followed by before query-time reuse pays for itself.
 *
 * Reusing derived state moves work from every query onto the first one after a rebuild, so the
 * honest question is not "are later queries faster" but "is the total lower". With `cold` the first
 * query after a rebuild, `warm` the steady-state query, and `old` the pre-change per-query cost,
 * reuse wins once `n * old > cold + n * warm`, i.e. `n > cold / (old - warm)`.
 *
 * `old` is measured on the same corpus by dropping the derived state before every query, which is
 * exactly the work the pre-change code did per query.
 *
 * The answer depends on how selective the query is, so both extremes are reported. A broad query
 * matches every chunk, so reuse saves only the tokenization. A selective query matches a handful,
 * so reuse also skips scoring the rest of the corpus. Quoting one number without the other would
 * misstate the trade.
 */

const BROAD = "refund approval policy";
const SELECTIVE = "zephyrine";

function buildCorpus(documentCount: number): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-be-"));
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		const body = Array.from(
			{ length: 40 },
			(_, p) =>
				`Paragraph ${p} of document ${i}. Refund approval policy, chargeback evidence retention, escalation threshold review.`,
		).join("\n\n");
		// The selective term is planted in exactly two documents regardless of corpus size, which is
		// the best case for candidate narrowing and is labelled as such in the output.
		const rare = i < 2 ? "\n\nzephyrine appears here.\n" : "";
		writeFileSync(join(docs, `doc-${i}.md`), `# Document ${i}\n\n${body}${rare}\n`);
	}
	return { root, searchPaths: [docs] };
}

function median(values: number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[Math.floor(sorted.length / 2)] as number;
}

/** Force the next query to rebuild derived state, reproducing the pre-change per-query cost. */
function dropCache(method: BM25Method): void {
	(method as unknown as { fallbackCache: unknown }).fallbackCache = undefined;
}

const rows: Record<string, number | string>[] = [];

for (const size of [100, 400, 1600]) {
	const { root, searchPaths } = buildCorpus(size);
	try {
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: "typescript-fallback" });
		const built = await method.sync();

		for (const [label, query] of [
			["broad", BROAD],
			["selective", SELECTIVE],
		] as const) {
			// cold: the first query after a rebuild, which also builds the derived state.
			dropCache(method);
			let started = performance.now();
			await method.retrieve(query, { topK: 10 });
			const cold = performance.now() - started;

			// warm: steady state with the derived state already built.
			const warmSamples: number[] = [];
			for (let i = 0; i < 20; i += 1) {
				started = performance.now();
				await method.retrieve(query, { topK: 10 });
				warmSamples.push(performance.now() - started);
			}
			const warm = median(warmSamples);

			// old: derived state discarded before every query, reproducing the pre-change cost.
			const oldSamples: number[] = [];
			for (let i = 0; i < 10; i += 1) {
				dropCache(method);
				started = performance.now();
				await method.retrieve(query, { topK: 10 });
				oldSamples.push(performance.now() - started);
			}
			const old = median(oldSamples);

			const breakEven = cold / (old - warm);
			rows.push({
				documents: size,
				chunks: built.indexedChunks,
				query: label,
				coldMs: +cold.toFixed(2),
				warmMs: +warm.toFixed(3),
				oldMs: +old.toFixed(2),
				breakEvenQueries: +breakEven.toFixed(2),
				breakEvenWhole: Math.max(1, Math.ceil(breakEven)),
			});
		}
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
}

console.log("\n| documents | chunks | query | cold (ms) | warm (ms) | old (ms) | break-even | whole queries |");
console.log("|---|---|---|---|---|---|---|---|");
for (const r of rows) {
	console.log(
		`| ${r.documents} | ${r.chunks} | ${r.query} | ${r.coldMs} | ${r.warmMs} | ${r.oldMs} | ${r.breakEvenQueries} | ${r.breakEvenWhole} |`,
	);
}
console.log("\nThe selective row is the best case for candidate narrowing: the term is planted in");
console.log("exactly two documents regardless of corpus size. The broad row matches every chunk.");
console.log(`JSON: ${JSON.stringify(rows)}`);
