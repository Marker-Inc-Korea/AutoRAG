import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";

/**
 * Measures per-query retrieval cost against corpus size and query selectivity.
 *
 * Selectivity is a required dimension. A query that matches every document exercises a different
 * code path cost than one that matches a handful, and reporting only one of them would misstate the
 * effect of any change that narrows the candidate set.
 */

/** Matches essentially every chunk: shared boilerplate present in all documents. */
const BROAD = ["refund approval policy", "chargeback evidence retention", "escalation threshold review"];

/** Matches a handful of chunks: identifiers planted in a small number of documents. */
const SELECTIVE = ["zephyrine", "quokkaledger", "vermillionclause"];

function buildCorpus(documentCount: number): { root: string; searchPaths: string[] } {
	const root = mkdtempSync(join(tmpdir(), "autorag-q-"));
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let i = 0; i < documentCount; i += 1) {
		const shared = Array.from(
			{ length: 40 },
			(_, p) =>
				`Paragraph ${p} of document ${i}. Refund approval policy, chargeback evidence retention, escalation threshold review, quarterly settlement report.`,
		).join("\n\n");
		// Plant each rare term in exactly two documents, independent of corpus size.
		const rare =
			i < 2
				? "\n\nzephyrine appears here.\n"
				: i < 4
					? "\n\nquokkaledger appears here.\n"
					: i < 6
						? "\n\nvermillionclause appears here.\n"
						: "";
		writeFileSync(join(docs, `doc-${String(i).padStart(5, "0")}.md`), `# Document ${i}\n\n${shared}${rare}\n`);
	}
	return { root, searchPaths: [docs] };
}

async function medianQueryMs(method: BM25Method, queries: readonly string[], rounds: number): Promise<number> {
	const times: number[] = [];
	for (let r = 0; r < rounds; r += 1) {
		for (const q of queries) {
			const started = performance.now();
			await method.retrieve(q, { topK: 10 });
			times.push(performance.now() - started);
		}
	}
	times.sort((a, b) => a - b);
	return times[Math.floor(times.length / 2)] as number;
}

const engine = (process.argv[2] ?? "tantivy") as "tantivy" | "typescript-fallback";
const sizes = engine === "typescript-fallback" ? [50, 100, 200, 400] : [50, 200, 800, 2000];
const rows: { documents: number; chunks: number; broadMs: number; selectiveMs: number }[] = [];

for (const size of sizes) {
	const { root, searchPaths } = buildCorpus(size);
	try {
		await syncParsedMirrors({ root, searchPaths });
		const method = new BM25Method({ root, forceEngine: engine });
		const built = await method.sync();
		await method.retrieve(BROAD[0] as string, { topK: 10 });
		rows.push({
			documents: size,
			chunks: built.indexedChunks,
			broadMs: await medianQueryMs(method, BROAD, 4),
			selectiveMs: await medianQueryMs(method, SELECTIVE, 4),
		});
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
}

console.log(`\nengine: ${engine}`);
console.log("| documents | chunks | broad (ms) | selective (ms) |");
console.log("|---|---|---|---|");
for (const r of rows) {
	console.log(`| ${r.documents} | ${r.chunks} | ${r.broadMs.toFixed(2)} | ${r.selectiveMs.toFixed(3)} |`);
}
const first = rows[0];
const last = rows[rows.length - 1];
if (first && last) {
	const growth = last.chunks / first.chunks;
	console.log(
		`\ncorpus x${growth.toFixed(0)} -> broad x${(last.broadMs / first.broadMs).toFixed(1)}, selective x${(last.selectiveMs / first.selectiveMs).toFixed(1)}`,
	);
}
console.log(`JSON: ${JSON.stringify(rows)}`);
